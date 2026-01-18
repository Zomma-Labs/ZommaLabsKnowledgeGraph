"""Chunker module for PDF → Markdown → Chunks conversion using Gemini."""

from .chunk_types import Chunk, DocumentPayload
from .loader import iter_document_payloads, document_id_from_path
from .markdown_chunker import (
    parse_markdown_to_chunks,
    chunk_markdown_file,
    chunks_to_jsonl,
)
from .pdf_to_markdown import pdf_to_markdown

__all__ = [
    # Data types
    "Chunk",
    "DocumentPayload",
    # Loader utilities
    "iter_document_payloads",
    "document_id_from_path",
    # Gemini PDF conversion
    "pdf_to_markdown",
    # Markdown chunking
    "parse_markdown_to_chunks",
    "chunk_markdown_file",
    "chunks_to_jsonl",
]
