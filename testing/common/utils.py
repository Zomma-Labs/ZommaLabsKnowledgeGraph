"""
MODULE: Utils
DESCRIPTION: Common utility functions for the RAG evaluation framework.
             Includes chunk loading, Q&A dataset loading, results saving, and embedding caching.
"""

import json
import hashlib
import pickle
import os
from pathlib import Path
from datetime import datetime
from typing import Callable, Any

import numpy as np
from dotenv import load_dotenv

from testing.common.schemas import EvalResult, EvalSummary

load_dotenv()

# Constants
EMBEDDING_DIMENSION = 3072  # text-embedding-3-large dimension


def load_chunks(filepath: str) -> list[dict]:
    """Load chunks from a JSONL file.

    Args:
        filepath: Path to the JSONL file containing chunks.

    Returns:
        List of dicts with keys: chunk_id, doc_id, body, header_path, metadata.
        Also includes breadcrumbs if present in the source file.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Chunk file not found: {filepath}")

    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
                # Normalize expected fields
                normalized = {
                    "chunk_id": chunk.get("chunk_id", f"chunk_{line_num}"),
                    "doc_id": chunk.get("doc_id", "unknown"),
                    "body": chunk.get("body", ""),
                    "header_path": chunk.get("header_path", ""),
                    "metadata": chunk.get("metadata", {}),
                }
                # Include breadcrumbs if present
                if "breadcrumbs" in chunk:
                    normalized["breadcrumbs"] = chunk["breadcrumbs"]
                chunks.append(normalized)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Invalid JSON on line {line_num}: {e.msg}",
                    e.doc,
                    e.pos,
                )
    return chunks


def load_qa_dataset(filepath: str) -> list[dict]:
    """Load Q&A dataset from JSON file (Beige_OA.json format).

    Args:
        filepath: Path to the JSON file containing Q&A pairs.

    Returns:
        List of dicts with keys: id, category, question, answer, entities.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
        KeyError: If the expected structure is not found.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Q&A dataset file not found: {filepath}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both direct list format and nested format with metadata
    if isinstance(data, list):
        qa_pairs = data
    elif isinstance(data, dict) and "qa_pairs" in data:
        qa_pairs = data["qa_pairs"]
    else:
        raise KeyError(
            "Expected either a list of Q&A pairs or a dict with 'qa_pairs' key"
        )

    # Normalize each Q&A pair
    normalized = []
    for qa in qa_pairs:
        normalized.append(
            {
                "id": qa.get("id", len(normalized) + 1),
                "category": qa.get("category", "uncategorized"),
                "question": qa.get("question", ""),
                "answer": qa.get("answer", ""),
                "entities": qa.get("entities", []),
            }
        )
    return normalized


def save_eval_results(
    results: list[EvalResult],
    summary: EvalSummary,
    config: dict,
    output_dir: str = "eval",
) -> str:
    """Save evaluation results to JSON file.

    Args:
        results: List of EvalResult objects from evaluation.
        summary: EvalSummary object with aggregated metrics.
        config: Configuration dict used for the evaluation run.
        output_dir: Directory to save results (default: "eval").

    Returns:
        The filepath of the saved file.

    Format: eval_{system_name}_{timestamp}.json
    """
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    system_name = summary.system_name.replace(" ", "_").lower()
    filename = f"eval_{system_name}_{timestamp}.json"
    filepath = output_path / filename

    # Build output structure
    output_data = {
        "metadata": {
            "system_name": summary.system_name,
            "timestamp": datetime.now().isoformat(),
            "config": config,
        },
        "summary": {
            "total_questions": summary.total_questions,
            "correct": summary.correct,
            "partially_correct": summary.partially_correct,
            "abstained": summary.abstained,
            "incorrect": summary.incorrect,
            "correct_pct": summary.correct_pct,
            "partially_correct_pct": summary.partially_correct_pct,
            "abstained_pct": summary.abstained_pct,
            "incorrect_pct": summary.incorrect_pct,
            "accuracy_pct": summary.accuracy_pct,
            "avg_time_ms": summary.avg_time_ms,
        },
        "results": [r.to_dict() for r in results],
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    return str(filepath)


def load_eval_results(filepath: str) -> tuple[list[EvalResult], EvalSummary, dict]:
    """Load evaluation results from a previously saved JSON file.

    Args:
        filepath: Path to the saved evaluation JSON file.

    Returns:
        Tuple of (results, summary, config).

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation results file not found: {filepath}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = [EvalResult.from_dict(r) for r in data.get("results", [])]
    config = data.get("metadata", {}).get("config", {})

    # Reconstruct summary from results
    summary = EvalSummary.from_results(
        system_name=data.get("metadata", {}).get("system_name", "unknown"),
        results=results,
    )

    return results, summary, config


class EmbeddingCache:
    """Cache embeddings to avoid recomputing for the same texts.

    Provides disk-based caching with content-based hashing to ensure
    consistent results across runs.

    Note: Uses pickle for efficient numpy array serialization. This cache
    is for local use only - do not load cache files from untrusted sources.

    Usage:
        cache = EmbeddingCache("eval/embeddings_cache.pkl")
        embeddings = cache.get_or_compute(texts, embedding_fn)
    """

    def __init__(self, cache_file: str | None = None):
        """Initialize the embedding cache.

        Args:
            cache_file: Optional path to a pickle file for persistent caching.
                       If None, cache is in-memory only.
        """
        self.cache_file = Path(cache_file) if cache_file else None
        self._cache: dict[str, np.ndarray] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cache from disk if it exists."""
        if self.cache_file and self.cache_file.exists():
            try:
                with open(self.cache_file, "rb") as f:
                    self._cache = pickle.load(f)
            except (pickle.PickleError, EOFError, KeyError):
                # Cache file corrupted, start fresh
                self._cache = {}

    def _save_cache(self) -> None:
        """Save cache to disk."""
        if self.cache_file:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "wb") as f:
                pickle.dump(self._cache, f)

    def _hash_text(self, text: str) -> str:
        """Generate a hash key for a text string."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def get_or_compute(
        self,
        texts: list[str],
        embedding_fn: Callable[[list[str]], list[list[float]]],
    ) -> np.ndarray:
        """Get embeddings from cache or compute them.

        Args:
            texts: List of text strings to embed.
            embedding_fn: Function that takes a list of texts and returns
                         a list of embedding vectors (as lists of floats).

        Returns:
            numpy array of shape (len(texts), embedding_dimension).
        """
        if not texts:
            return np.array([])

        # Check which texts need computing
        result = []
        texts_to_compute = []
        indices_to_compute = []

        for i, text in enumerate(texts):
            key = self._hash_text(text)
            if key in self._cache:
                result.append((i, self._cache[key]))
            else:
                texts_to_compute.append(text)
                indices_to_compute.append(i)

        # Compute missing embeddings
        if texts_to_compute:
            new_embeddings = embedding_fn(texts_to_compute)
            for text, emb, idx in zip(
                texts_to_compute, new_embeddings, indices_to_compute
            ):
                key = self._hash_text(text)
                emb_array = np.array(emb, dtype=np.float32)
                self._cache[key] = emb_array
                result.append((idx, emb_array))

            # Save updated cache
            self._save_cache()

        # Sort by original index and stack
        result.sort(key=lambda x: x[0])
        return np.vstack([emb for _, emb in result])

    def get_cached(self, text: str) -> np.ndarray | None:
        """Get a single cached embedding if it exists.

        Args:
            text: The text to look up.

        Returns:
            The cached embedding array or None if not found.
        """
        key = self._hash_text(text)
        return self._cache.get(key)

    def clear(self) -> None:
        """Clear the cache (both in-memory and on disk)."""
        self._cache = {}
        if self.cache_file and self.cache_file.exists():
            self.cache_file.unlink()

    def __len__(self) -> int:
        """Return the number of cached embeddings."""
        return len(self._cache)


def get_openai_embedding_fn(model: str = "text-embedding-3-large") -> Callable:
    """Get an OpenAI embedding function for use with EmbeddingCache.

    Args:
        model: The OpenAI embedding model to use.

    Returns:
        A function that takes a list of texts and returns embeddings.
    """
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(model=model)

    def embed(texts: list[str]) -> list[list[float]]:
        return embeddings.embed_documents(texts)

    return embed


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector (1D numpy array).
        b: Second vector (1D numpy array).

    Returns:
        Cosine similarity score between -1 and 1.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def batch_cosine_similarity(
    query: np.ndarray, documents: np.ndarray
) -> np.ndarray:
    """Compute cosine similarity between a query and multiple documents.

    Args:
        query: Query vector of shape (embedding_dim,).
        documents: Document vectors of shape (num_docs, embedding_dim).

    Returns:
        Array of similarity scores of shape (num_docs,).
    """
    if len(documents) == 0:
        return np.array([])

    # Normalize vectors
    query_norm = query / (np.linalg.norm(query) + 1e-10)
    doc_norms = np.linalg.norm(documents, axis=1, keepdims=True) + 1e-10
    documents_norm = documents / doc_norms

    # Compute similarities
    return np.dot(documents_norm, query_norm)


def format_chunk_for_display(
    chunk: dict, max_body_length: int = 200, include_metadata: bool = False
) -> str:
    """Format a chunk dictionary for human-readable display.

    Args:
        chunk: Chunk dictionary with body, header_path, etc.
        max_body_length: Maximum length of body text to display.
        include_metadata: Whether to include metadata in output.

    Returns:
        Formatted string representation of the chunk.
    """
    body = chunk.get("body", "")
    if len(body) > max_body_length:
        body = body[:max_body_length] + "..."

    header_path = chunk.get("header_path", "")
    chunk_id = chunk.get("chunk_id", "unknown")

    lines = [
        f"[{chunk_id}]",
        f"  Path: {header_path}" if header_path else "",
        f"  Body: {body}",
    ]

    if include_metadata and chunk.get("metadata"):
        lines.append(f"  Metadata: {chunk.get('metadata')}")

    return "\n".join(line for line in lines if line)


def print_eval_summary(summary: EvalSummary) -> None:
    """Print a formatted evaluation summary to stdout.

    Args:
        summary: EvalSummary object to display.
    """
    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY: {summary.system_name}")
    print("=" * 60)
    print(f"Total Questions: {summary.total_questions}")
    print(f"")
    print(f"Results:")
    print(f"  Correct:           {summary.correct:3d} ({summary.correct_pct:5.1f}%)")
    print(
        f"  Partially Correct: {summary.partially_correct:3d} ({summary.partially_correct_pct:5.1f}%)"
    )
    print(f"  Abstained:         {summary.abstained:3d} ({summary.abstained_pct:5.1f}%)")
    print(f"  Incorrect:         {summary.incorrect:3d} ({summary.incorrect_pct:5.1f}%)")
    print(f"")
    print(f"Overall Accuracy (correct + partial): {summary.accuracy_pct:.1f}%")
    print(f"Average Time per Question: {summary.avg_time_ms:.0f}ms")
    print("=" * 60 + "\n")
