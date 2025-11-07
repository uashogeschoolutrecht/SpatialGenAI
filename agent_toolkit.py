"""
Utility functions and agent orchestration helpers for the agent workflow

AgentOrchestrator accepts a list of agents that execute sequentially:
- Usage: AgentOrchestrator(agents=[reasoning_agent, validation_agent, ...])
- Each agent's output is passed to the next agent in the pipeline via context
"""

from __future__ import annotations

from pathlib import Path

import ast
import json
import re

import faiss
import geopandas as gpd
import google.generativeai as genai
import numpy as np
import openai
import tiktoken

DEFAULT_EMBEDDER = "text-embedding-3-large"
MAX_BATCH_TOKENS = 300_000
DEFAULT_FAISS_INDEX_DIR = Path("data/spatial_genai_storage/database_RAG")
DEFAULT_FAISS_META_PATH = DEFAULT_FAISS_INDEX_DIR / "metadata.json"
DEFAULT_FAISS_INDEX_PATH = DEFAULT_FAISS_INDEX_DIR / "faiss.index"
DEFAULT_GPKG_FILE_PATH = Path("data/spatial_genai_storage/data_PDOK/top10nl_Compleet.gpkg")
DEFAULT_BBOX_UTRECHT_PROV = (109311, 430032, 169326, 479261)


def embed_texts(texts, batch_size=1000, embedder=DEFAULT_EMBEDDER):
    """Generate embeddings for a list of texts using OpenAI's API.

    Args:
        texts (Iterable[str]): The texts to embed.
        batch_size (int): Number of texts per API call (default 1000).
        embedder (str): The embedding model name (default DEFAULT_EMBEDDER).

    Returns:
        List[List[float]]: List of embedding vectors.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    all_embs = []
    texts_list = list(texts)
    for start in range(0, len(texts_list), batch_size):
        batch = texts_list[start : start + batch_size]
        total_tokens = sum(len(enc.encode(item)) for item in batch)
        if total_tokens > MAX_BATCH_TOKENS:
            print(
                f"Batch {start // batch_size} skipped; contains {total_tokens} tokens,"
                f" above limit {MAX_BATCH_TOKENS}."
            )
            continue
        resp = openai.embeddings.create(model=embedder, input=batch)
        all_embs.extend([record.embedding for record in resp.data])
    return all_embs


def faiss_search(query, k, location=None, meta_path=DEFAULT_FAISS_META_PATH, index_path=DEFAULT_FAISS_INDEX_PATH, embedder=DEFAULT_EMBEDDER):
    """Search the FAISS index for similar text chunks to the query.

    Args:
        query (str): The search query text.
        k (int): Number of top results to return.
        location (Optional[str]): Optional location filter for results.
        meta_path (Path): Path to the metadata JSON file.
        index_path (Path): Path to the FAISS index file.
        embedder (str): Embedding model name.

    Returns:
        List[Dict[str, Any]]: List of search results with scores and metadata.
    """
    meta_path = Path(meta_path)
    index_path = Path(index_path)
    print(f"Searching for: '{query}' (location filter: {location})")
    with meta_path.open() as handle:
        metas = json.load(handle)
    index = faiss.read_index(str(index_path))
    query_vector = embed_texts([query], embedder=embedder)[0]
    distances, indices = index.search(
        np.array([query_vector], dtype="float32"), min(k * 4, len(metas))
    )
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        metadata = metas[idx]
        if location and metadata.get("location") != location:
            continue
        results.append({**metadata, "score": float(dist)})
        if len(results) >= k:
            break
    return results


def faiss_search_flatten(search_results):
    """Format search results into a human-readable string.

    Args:
        search_results (Iterable[Dict[str, Any]]): The search results to format.

    Returns:
        str: Formatted string of results.
    """
    lines = []
    for res in search_results:
        lines.append(
            f"Score: {res['score']}, Location: {res['location']}, Text: {res['text']}"
        )
    return "\n".join(lines)


def gpkg_query(table_name, bbox=DEFAULT_BBOX_UTRECHT_PROV, filters=None, limit=None, gpkg_file=DEFAULT_GPKG_FILE_PATH):
    """Query a GeoPackage file for spatial data with optional filters.

    Args:
        table_name (str): Name of the table/layer to query.
        bbox (Tuple[float, float, float, float]): Bounding box (minx, miny, maxx, maxy).
        filters (Optional[Dict[str, Any]]): Column filters to apply.
        limit (Optional[int]): Maximum number of rows to return.
        gpkg_file (Path): Path to the GeoPackage file.

    Returns:
        gpd.GeoDataFrame: The queried GeoDataFrame.
    """
    gdf = gpd.read_file(gpkg_file, layer=table_name, bbox=bbox)
    if filters:
        for column, value in filters.items():
            if column in gdf.columns:
                gdf = gdf[gdf[column] == value]
    if limit:
        gdf = gdf.head(limit)
    return gdf


def call_llm(prompt, system_prompt=None, model="gemini-2.5-pro", temperature=0.7, max_output_tokens=16_000):
    """Send a prompt to a large language model and get the response.

    Args:
        prompt (str): The user prompt to send.
        system_prompt (Optional[str]): Optional system prompt.
        model (str): The model name to use.
        temperature (float): Sampling temperature.
        max_output_tokens (int): Maximum tokens in response.

    Returns:
        str: The model's text response.
    """
    generation_config = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    model_client = genai.GenerativeModel(
        model_name=model,
        generation_config=generation_config,
        system_instruction=system_prompt if system_prompt else None,
    )
    chat = model_client.start_chat(history=[])
    response = chat.send_message(prompt)
    return response.text


def prepare_system_prompt(thematic_object, object_description, judicial_reference, database_reference, base_polygon, template=None):
    """Fill in a system prompt template with context variables.

    Args:
        thematic_object (str): The thematic object name.
        object_description (str): Description of the object.
        judicial_reference (str): Judicial reference text.
        database_reference (str): Database reference text.
        base_polygon (str): Base polygon name.
        template (Optional[str]): The template string to fill.

    Returns:
        str: The filled system prompt.
    """
    if template is None:
        raise ValueError("System prompt template must be provided.")
    return (
        template.replace("{thematic_object}", thematic_object)
        .replace("{object_description}", object_description)
        .replace("{judicial_reference}", judicial_reference)
        .replace("{database_reference}", database_reference)
        .replace("{base_polygon}", base_polygon)
    )


def parse_llm_json(raw):
    """Parse LLM response into JSON, with fallbacks for various formats.

    Args:
        raw (str): The raw LLM response string.

    Returns:
        Any: Parsed JSON object or equivalent.
    """
    if not raw or not raw.strip():
        raise ValueError("Empty LLM response")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    candidate = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.S)
    if fenced:
        candidate = fenced.group(1).strip()
    if (candidate.startswith("\"") and candidate.endswith("\"")) or (
        candidate.startswith("'") and candidate.endswith("'")
    ):
        try:
            candidate = ast.literal_eval(candidate)
        except Exception:
            candidate = candidate[1:-1].encode("utf-8").decode("unicode_escape")
    first_json = re.search(r"(\[.*?\]|\{.*?\})", candidate, flags=re.S)
    if first_json:
        candidate = first_json.group(1)
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as json_err:
        try:
            import yaml  # type: ignore
        except ImportError as import_err:
            raise json_err from import_err
        try:
            parsed_yaml = yaml.safe_load(candidate)
        except yaml.YAMLError as yaml_err:
            raise json_err from yaml_err
        return parsed_yaml


class Agent:
    """Base class for agents that interact with LLMs.

    Provides a framework for building prompts, parsing responses,
    and calling the LLM.

    Attributes:
        name (str): The agent's name.
        system_prompt_template (str): Template for system prompts.
        model (str): The LLM model to use.
    """
    def __init__(self, name, system_prompt_template, model="gemini-2.5-pro"):
        self.name = name
        self.system_prompt_template = system_prompt_template
        self.model = model

    def build_prompts(self, context):
        raise NotImplementedError

    def parse_response(self, raw):
        raise NotImplementedError

    def __call__(self, context):
        system_prompt, user_prompt = self.build_prompts(context)
        raw_response = call_llm(user_prompt, system_prompt=system_prompt, model=self.model)
        parsed_response = self.parse_response(raw_response)
        return {"agent": self.name, "raw": raw_response, "parsed": parsed_response}


class ReasoningAgent(Agent):
    """Agent specialized in reasoning and generating filter proposals.

    Extends Agent to build prompts for thematic object analysis
    and parse JSON responses containing filter lists.
    """
    def build_prompts(self, context):
        feedback = context.get("feedback") or []
        system_prompt = prepare_system_prompt(
            thematic_object=context["thematic_object"],
            object_description=context["object_description"],
            judicial_reference=context["judicial_reference"],
            database_reference=context["database_reference"],
            base_polygon=context["base_polygon_name"],
            template=self.system_prompt_template,
        )
        base_prompt = (
            "Identificeer alle relevante belemmeringen en geef deze in het gevraagde JSON-formaat."
            " Antwoord uitsluitend met geldige JSON, zonder extra tekst of uitleg."
        )
        if feedback:
            feedback_lines = "\n".join(f"- {item}" for item in feedback)
            user_prompt = (
                "Herzie het voorstel op basis van onderstaande feedback en lever opnieuw een JSON-lijst van belemmeringen.\n"
                "Geef enkel verbeterde json output, geen extra uitleg.\n"
                f"Feedback:\n{feedback_lines}\n\n"
                f"Originele opdracht:\n{base_prompt}"
            )
        else:
            user_prompt = base_prompt
        return system_prompt, user_prompt

    def parse_response(self, raw):
        parsed = parse_llm_json(raw)
        if not isinstance(parsed, list):
            raise ValueError("Reasoning agent expected een lijst van belemmeringen.")
        return parsed


class ValidationAgent(Agent):
    """Agent specialized in validating and providing feedback on proposals.

    Extends Agent to evaluate proposals and return structured
    feedback with approval, comments, and issues.
    """
    def build_prompts(self, context):
        system_prompt = self.system_prompt_template
        filters = context.get("filters") or []
        filters_json = json.dumps(filters, indent=2)
        previous_feedback = context.get("feedback") or []
        intro = (
            "Beoordeel het onderstaande voorstel van de inhoudsexpert voor het thematische object '%s'"
            " binnen de basispolygoon '%s'."
            % (context["thematic_object"], context["base_polygon_name"])
        )
        user_prompt = (
            f"{intro}\n\n"
            f"Objectbeschrijving:\n{context['object_description']}\n\n"
            f"Voorstel (JSON):\n```json\n{filters_json}\n```\n\n"
            "Geef een JSON-antwoord met de velden 'approved', 'comments' en 'issues'."
            " Antwoord uitsluitend met geldige JSON, zonder extra tekst of uitleg."
        )
        if previous_feedback:
            prev_feedback_block = "\n".join(f"- {item}" for item in previous_feedback)
            user_prompt += f"\n\nFeedback uit eerdere rondes voor referentie:\n{prev_feedback_block}"
        return system_prompt, user_prompt

    def parse_response(self, raw):
        parsed = parse_llm_json(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Validation agent verwacht een JSON-object met goedkeuring en opmerkingen.")
        parsed.setdefault("approved", False)
        parsed.setdefault("comments", [])
        parsed.setdefault("issues", [])
        if not isinstance(parsed["comments"], list):
            parsed["comments"] = [str(parsed["comments"])]
        if not isinstance(parsed["issues"], list):
            parsed["issues"] = [str(parsed["issues"])]
        return parsed


class AgentOrchestrator:
    """Orchestrates multi-round interactions between agents in a sequential pipeline.

    Agents execute in sequence, with each agent's output passed to the next in context.
    For two-agent workflows, maintains 'reasoning' and 'validation' keys for compatibility.

    Attributes:
        agents (List[Agent]): List of agents to execute in sequence.
        save_dir (Path): Directory to save round results (optional).
    """
    def __init__(self, agents, save_dir=Path("scratch/agent_rounds")):
        self.agents = agents
        self.save_dir = save_dir
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def run_rounds(self, agent_context, num_rounds=1):
        results = []
        feedback = agent_context.get("feedback") or []
        
        for round_index in range(1, num_rounds + 1):
            round_payload = {"round": round_index}
            context = dict(agent_context)
            context["feedback"] = feedback
            
            # Execute agents in sequence
            for idx, agent in enumerate(self.agents):
                agent_result = agent(context)
                
                # Store result - use 'reasoning'/'validation' keys for 2-agent setup
                if len(self.agents) == 2 and idx == 0:
                    round_payload["reasoning"] = agent_result
                    # Update context with reasoning output for next agent
                    context.update({
                        "filters": agent_result["parsed"],
                        "filters_raw": agent_result["raw"],
                    })
                elif len(self.agents) == 2 and idx == 1:
                    round_payload["validation"] = agent_result
                else:
                    # Multi-agent mode: use agent name as key
                    round_payload[agent.name] = agent_result
                    # Pass parsed output to next agent in context
                    context[f"{agent.name}_output"] = agent_result["parsed"]
                    context[f"{agent.name}_raw"] = agent_result["raw"]
            
            results.append(round_payload)
            
            # Save round results
            if self.save_dir is not None:
                round_path = self.save_dir / f"round_{round_index}.json"
                with round_path.open("w", encoding="utf-8") as handle:
                    json.dump(round_payload, handle, indent=2)
            
            # Extract feedback for next round from last agent
            last_agent_key = [k for k in round_payload.keys() if k != "round"][-1]
            last_result = round_payload[last_agent_key]["parsed"] or {}
            
            if isinstance(last_result, dict):
                feedback = last_result.get("comments") or []
                if not isinstance(feedback, list):
                    feedback = [str(feedback)]
                # Check approval to break early
                if last_result.get("approved"):
                    break
        
        return results
