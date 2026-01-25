"""Common utilities and base classes for RAG evaluation."""

from testing.common.schemas import EvalResult, EvalSummary, JudgeVerdict
from testing.common.utils import (
    load_chunks,
    load_qa_dataset,
    save_eval_results,
    load_eval_results,
    EmbeddingCache,
    get_openai_embedding_fn,
    compute_cosine_similarity,
    batch_cosine_similarity,
    format_chunk_for_display,
    print_eval_summary,
    EMBEDDING_DIMENSION,
)
from testing.common.judge import LLMJudge

__all__ = [
    # Schemas
    "EvalResult",
    "EvalSummary",
    "JudgeVerdict",
    # Judge
    "LLMJudge",
    # Data loading
    "load_chunks",
    "load_qa_dataset",
    # Results I/O
    "save_eval_results",
    "load_eval_results",
    # Embeddings
    "EmbeddingCache",
    "get_openai_embedding_fn",
    "EMBEDDING_DIMENSION",
    # Similarity
    "compute_cosine_similarity",
    "batch_cosine_similarity",
    # Display
    "format_chunk_for_display",
    "print_eval_summary",
]
