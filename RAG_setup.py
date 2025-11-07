"""Turns a folder of PDFs into a FAISS index you can search via embeddings.
Requires an OpenAI API key (or drop-in compatible embedder) exposed as OPENAI_API_KEY.
Run it once to build the index, optionally pass --query to test lookups right away.

Example: python RAG_setup.py --data-dir data/spatial_genai_storage/data_RAG --index-dir data/spatial_genai_storage/database_RAG --query "energie"
--query is an optional argument if you want to test the index after building it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Sequence

import faiss
import numpy as np
import openai
import tiktoken
from pypdf import PdfReader


def embed_texts(texts: Sequence[str], *, model: str, batch_size: int = 1000, max_batch_tokens: int = 300_000) -> List[List[float]]:
    """Create embeddings for a list of texts using the specified OpenAI model."""
    encoder = tiktoken.get_encoding("cl100k_base")
    embeddings: List[List[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        total_tokens = sum(len(encoder.encode(item)) for item in batch)
        if total_tokens > max_batch_tokens:
            # Guardrail against request limits by dropping oversized batches.
            print(
                f"Batch {start // batch_size} skipped; {total_tokens} tokens exceeds {max_batch_tokens}."
            )
            continue
        response = openai.embeddings.create(model=model, input=batch)
        embeddings.extend([record.embedding for record in response.data])
    return embeddings


def read_pdf(path: Path) -> str:
    """Extract text from a PDF file, normalising whitespace."""
    reader = PdfReader(path)
    pages: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            pages.append(text)
    return "\n".join(pages)


def chunk_text(text: str, *, target_tokens: int = 300, overlap: int = 50, encoding_name: str = "cl100k_base") -> List[str]:
    """Chunk text into overlapping segments measured by token count."""
    encoder = tiktoken.get_encoding(encoding_name)
    tokens = encoder.encode(text)
    chunks: List[str] = []
    start = 0
    while start < len(tokens):
        end = start + target_tokens
        slice_tokens = tokens[start:end]
        chunk = encoder.decode(slice_tokens)
        chunks.append(chunk)
        start += max(target_tokens - overlap, 1)
    return chunks


def infer_metadata(filename: str) -> dict:
    """Infer simple metadata fields from a filename."""
    location = None
    doc_type = None
    match_location = re.search(r"Omgevingsplan_(\w+)\.pdf", filename)
    match_doc = re.search(r"(\w+)_*", filename)
    if match_location:
        location = match_location.group(1)
    if match_doc:
        doc_type = match_doc.group(1)
    return {
        "source_file": filename,
        "location": location,
        "doc_type": doc_type,
        "language": "nl",
    }


def build_index(data_dir: Path, index_dir: Path, *, embedder: str, batch_size: int, target_tokens: int, overlap: int) -> None:
    """Build the FAISS index and metadata JSON from PDF documents."""
    index_dir.mkdir(parents=True, exist_ok=True)
    meta_path = index_dir / "metadata.json"
    index_path = index_dir / "faiss.index"

    documents = sorted(data_dir.glob("*.pdf"))
    if not documents:
        raise FileNotFoundError(f"No PDF files found in {data_dir}")

    metadata_records: List[dict] = []
    text_corpus: List[str] = []

    for pdf_path in documents:
        print(f"Processing PDF: {pdf_path.name}")
        raw_text = read_pdf(pdf_path)
        if not raw_text:
            print("  - No extractable text; skipping.")
            continue
        # Convert each document into overlapping token chunks for embedding.
        chunks = chunk_text(raw_text, target_tokens=target_tokens, overlap=overlap)
        print(f"  - Extracted {len(chunks)} chunks")
        base_metadata = infer_metadata(pdf_path.name)
        for idx, chunk in enumerate(chunks):
            record = {**base_metadata, "chunk_id": f"{pdf_path.name}::${idx}", "text": chunk}
            metadata_records.append(record)
            text_corpus.append(chunk)

    if not text_corpus:
        raise ValueError("No text chunks generated; cannot build index.")

    print("Generating embeddings...")
    # Single-shot embedding to keep FAISS vectors aligned with metadata order.
    embeddings = embed_texts(text_corpus, model=embedder, batch_size=batch_size)
    if not embeddings:
        raise ValueError("Embedding generation returned no vectors.")

    embedding_matrix = np.array(embeddings, dtype="float32")
    dimension = embedding_matrix.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_matrix)

    # Persist both the FAISS index and the accompanying metadata lookup.
    faiss.write_index(index, str(index_path))
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata_records, handle, ensure_ascii=False, indent=2)

    print(f"Index written to {index_path}")
    print(f"Metadata written to {meta_path}")


def search_index(query: str, *, k: int, index_path: Path, meta_path: Path, embedder: str, location: str | None = None) -> List[dict]:
    """Run a simple similarity search against the built index."""
    with meta_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not metadata:
        raise ValueError("Metadata is empty; build the index first.")

    index = faiss.read_index(str(index_path))
    # Embed the query to reuse the same vector space as the corpus.
    query_vector = embed_texts([query], model=embedder)[0]
    distances, indices = index.search(
        np.array([query_vector], dtype="float32"), min(k * 4, len(metadata))
    )

    results: List[dict] = []
    for distance, idx in zip(distances[0], indices[0]):
        record = metadata[idx]
        if location and record.get("location") != location:
            continue
        # Append matches with their score, stopping once k results are collected.
        results.append({**record, "score": float(distance)})
        if len(results) >= k:
            break
    return results


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Build a FAISS RAG index from PDFs.")
    parser.add_argument(
        "--data-dir",
        default="data/spatial_genai_storage/data_RAG",
        help="Directory containing PDF source documents.",
    )
    parser.add_argument(
        "--index-dir",
        default="data/spatial_genai_storage/database_RAG",
        help="Directory to store the FAISS index and metadata.",
    )
    parser.add_argument(
        "--embedder",
        default="text-embedding-3-large",
        help="OpenAI embedding model to use.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1000, help="Embedding batch size." 
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=300,
        help="Token count per chunk before overlap is applied.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Token overlap between consecutive chunks.",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Optional query to run after building the index.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of search results to return when --query is provided.",
    )
    parser.add_argument(
        "--location",
        default=None,
        help="Optional location filter for search results.",
    )
    return parser.parse_args()


def main() -> None:
    """Build the index and optionally run a query."""
    args = parse_args()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")
    openai.api_key = api_key

    data_dir = Path(args.data_dir)
    index_dir = Path(args.index_dir)

    build_index(
        data_dir,
        index_dir,
        embedder=args.embedder,
        batch_size=args.batch_size,
        target_tokens=args.target_tokens,
        overlap=args.overlap,
    )

    if args.query:
        meta_path = index_dir / "metadata.json"
        index_path = index_dir / "faiss.index"
        results = search_index(
            args.query,
            k=args.k,
            index_path=index_path,
            meta_path=meta_path,
            embedder=args.embedder,
            location=args.location,
        )
        print("\nSearch results:")
        for result in results:
            snippet = result["text"][:160].replace("\n", " ")
            print(f"Score: {result['score']:.4f} | Location: {result.get('location')} | {snippet}")


if __name__ == "__main__":
    main()
