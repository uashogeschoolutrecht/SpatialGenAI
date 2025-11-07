# Geospatial Agent Toolkit for Utrecht

**Multi-agent LLM system for spatial constraint analysis using geospatial data and RAG-based regulatory knowledge.**

This repository is the results of a collaboration between the Hogeschool Utrecht (University of Applied Science Utrecht) and the Province of Utrecht. It contains the result of a feasibility study in which we tried to gain knowledge on how well current AI system (LLMs specifically) are able to perform spatial planning and reasoning tasks. As such it houses geospatial utilities and an agent-based workflow for analyzing spatial constraints (e.g., wind turbine placement restrictions) within the Province of Utrecht, Netherlands. The system combines:
- **RAG (Retrieval-Augmented Generation)** for retrieving relevant legal/regulatory constraints
- **Multi-agent orchestration** (reasoning + validation agents) powered by Gemini 2.5 Pro
- **Geospatial processing** with GML/GPKG data (EPSG:28992 RD New coordinate system <-- Dutch coordinate system)

For questions and/or comments, see Support at the bottom of the page.

---

## 🎯 Key Features

- **Agent-based reasoning workflow**: Sequential pipeline of LLM agents (reasoning → validation → optional extensions)
- **Geospatial constraint mapping**: Query and process Dutch PDOK spatial datasets (Top10NL, administrative boundaries)
- **RAG-powered legal analysis**: FAISS vector search over regulatory documents for spatial planning
- **GML/GPKG utilities**: CRS-aware polygon processing (WGS84 ↔ RD New conversion)
- **Automated geometry operations**: Buffered cuts, polygon simplification, multi-layer GPKG output

---

## 📂 Project Structure

```
.
├── agent_toolkit.py          # Agent orchestration (ReasoningAgent, ValidationAgent, AgentOrchestrator)
├── agent_prompts.py          # LLM system prompts for agents
├── agent_workflow.ipynb      # Main workflow: RAG query → agent analysis → geometry processing
├── RAG_setup.py              # Build FAISS index from PDF documents (one-time setup)
├── utils.py                  # Core geospatial helpers (CRS transforms, GML I/O, polygon ops)
├── requirements.txt          # Python dependencies
├── data/                     # Data directory (not in repo, see Setup)
│   └── spatial_genai_storage/
│       ├── data_RAG/         # PDF source documents for RAG
│       ├── database_RAG/     # FAISS index + metadata
│       └── data_PDOK/        # GPKG files (e.g., Top10NL)
├── archive/                  # Historical code examples from previous attempts
│   ├── code/                 # Sample scripts (WFS fetching, data entry)
│   └── results/              # Archived boundary files
```

---

## Quick Start

### Prerequisites

1. **System Dependencies**:
   - **GDAL** (required for GML/GPKG I/O):
     ```bash
     # Ubuntu/Debian
     sudo apt-get install gdal-bin libgdal-dev
     
     # macOS
     brew install gdal
     ```
   - Python 3.10+

2. **API Keys**:
   - OpenAI API key (for embeddings: `text-embedding-3-large`)
   - Google Gemini API key (for agent reasoning: `gemini-2.5-pro`)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd <repository-name>

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add:
# OPENAI_API_KEY=your_key_here
# GOOGLE_API_KEY=your_key_here
```

### Setup Data

Our data was stored in `data/spatial_genai_storage/`:
- `data_RAG/`: PDF documents with legal/regulatory text
- `data_PDOK/`: GeoPackage files (we used [TOP10NL Geopackage - top10nl_Compleet-2024.gpkg ](https://nationaalgeoregister.nl/geonetwork/srv/dut/catalog.search#/metadata/29d5310f-dd0d-45ba-abad-b4ffc6b8785f), see e.g., PDOK for lots of free public resources through convenient API access)
- `database_RAG/`: Will be created by RAG setup

---

## Usage

### 1. Build RAG Index (One-Time)

```bash
python RAG_setup.py \
  --data-dir data/spatial_genai_storage/data_RAG \
  --index-dir data/spatial_genai_storage/database_RAG \
  --query "windenergie restricties"  # Optional test query
```

Creates FAISS index from PDFs for semantic search over regulatory documents.

### 2. Run Agent Workflow

Open and execute `agent_workflow.ipynb`:

```bash
jupyter notebook agent_workflow.ipynb
```

**Workflow steps:**
1. Query RAG database for relevant legal constraints (e.g., wind turbine restrictions)
2. Reasoning agent analyzes constraints → proposes spatial filters (table/column/value)
3. Validation agent reviews proposal → provides feedback
4. Iterate for N rounds with feedback loop
5. Apply filters to GeoPackage data → cut geometries from base polygon
6. Export results to multi-layer GPKG with categorized constraints

**Example output:**
- `utrecht_cut_with_categories.gpkg`: Remaining suitable area + cut layers per constraint category

---

## Agent System Architecture

### Agent Types

1. **ReasoningAgent** (`agent_toolkit.py`):
   - Analyzes thematic objects (e.g., "windturbine") + legal context
   - Outputs structured JSON: `[{tabel, kolom, waarde, categorie, reden}, ...]`
   - Refines proposals based on validation feedback

2. **ValidationAgent** (`agent_toolkit.py`):
   - Reviews reasoning agent's output for completeness/correctness
   - Returns: `{approved: bool, comments: [...], issues: [...]}`
   - Drives feedback loop for iterative refinement

3. **AgentOrchestrator**:
   - Sequential pipeline: `agents=[reasoning, validation, ...]`
   - Manages multi-round execution with feedback propagation
   - Saves round results to JSON for audit trail

### Multi-Agent Extension

Add custom agents to pipeline:

```python
from agent_toolkit import Agent, AgentOrchestrator

class OptimizationAgent(Agent):
    def build_prompts(self, context):
        # Access previous agents' outputs
        reasoning_output = context.get("filters")
        return system_prompt, user_prompt
    
    def parse_response(self, raw):
        return parse_llm_json(raw)

# 3-agent pipeline
orchestrator = AgentOrchestrator(agents=[
    reasoning_agent,
    validation_agent,
    OptimizationAgent(name="optimizer", system_prompt_template=PROMPT3)
])
results = orchestrator.run_rounds(context, num_rounds=2)
```

---

## 🔧 Configuration

### Environment Variables (`.env`)

```bash
OPENAI_API_KEY=sk-...           # For embeddings (text-embedding-3-large)
GOOGLE_API_KEY=AIza...          # For Gemini 2.5 Pro (agent reasoning)
```

### Key Parameters (`agent_workflow.ipynb`)

```python
NUM_ROUNDS = 2                   # Agent feedback loop iterations
EMBEDDER = "text-embedding-3-large"
EMB_DIM = 3072                   # Embedding dimensions
BBOX_UTRECHT_PROV = (109311, 430032, 169326, 479261)  # RD coordinates
```

---

## Example Use Case: Wind Turbine Suitability

**Input:**
- Thematic object: "windturbine" (160m hub height, 162m rotor diameter, 241m tip height)
- Base polygon: Province of Utrecht
- Query: "Legal restrictions for wind turbine placement"

**Process:**
1. RAG retrieves 30 relevant document chunks (laws, policies, spatial plans)
2. Reasoning agent identifies constraints:
   - `{tabel: "top10nl_gebouw_vlak", kolom: "typebouw", waarde: "woning", categorie: "harde belemmering"}`
   - `{tabel: "top10nl_waterdeel_vlak", kolom: "typewater", waarde: "meer", categorie: "complexe belemmering"}`
   - ... (more filters)
3. Validation agent reviews, requests clarifications
4. Round 2: Reasoning agent refines based on feedback
5. Geometry processing: Cut buildings, water, roads from base polygon
6. Output: Multi-layer GPKG with remaining suitable areas + cut zones by category

---

## Testing

```bash
# Test agent orchestrator (mock agents, no LLM calls)
python test_multi_agent.py

# Verify geospatial imports
python -c "import geopandas; import faiss; import openai; print('✓ All imports OK')"

# Test CRS transforms
python -c "from utils import wgs84_to_rd, rd_to_wgs84; print('✓ Utils imported')"
```

---

## Support

For questions about this repository, contact:
- Fabian Kok [fabian.kok@hu.nl], Hogeschool Utrecht, main developer
- Rob Peters [], Provincie Utrecht, project owner

---

## 🔗 Related Resources

- [GeoPandas Documentation](https://geopandas.org/)
- [PDOK Services](https://www.pdok.nl/) (Dutch geospatial data)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Google Gemini API](https://ai.google.dev/)
