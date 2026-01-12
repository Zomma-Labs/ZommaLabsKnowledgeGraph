"""Shared data structures for the GraphRAG workflow."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field


class DocumentPayload(BaseModel):
    """Metadata and optional bytes for a document to be processed."""

    doc_id: str
    path: Optional[Path] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    bytes: Optional[bytes] = None

    class Config:
        arbitrary_types_allowed = True


class Paragraph(BaseModel):
    """A single paragraph extracted from a document."""

    paragraph_id: str
    text: str
    page: Optional[int] = None
    doc_item_ref: Optional[str] = None


class Subsection(BaseModel):
    """A subsection containing multiple paragraphs."""

    subheading: str
    paragraphs: List[Paragraph] = Field(default_factory=list)


class Section(BaseModel):
    """A top-level section with optional subsections."""

    heading: str
    order: int
    subsections: List[Subsection] = Field(default_factory=list)


class TableRowPayload(BaseModel):
    """A single row extracted from a table."""

    table_ref: str
    row_index: int
    headers: List[str]
    values: Dict[str, str]
    page: Optional[int] = None
    label: Optional[str] = None


class ScrapeArtifact(BaseModel):
    """The result of scraping a document with structural information."""

    doc_id: str
    sections: List[Section] = Field(default_factory=list)
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    figures: List[Dict[str, Any]] = Field(default_factory=list)
    source_path: Optional[Path] = None
    docling_json: Optional[Dict[str, Any]] = None
    doc_item_breadcrumbs: Dict[str, List[str]] = Field(default_factory=dict)
    table_rows: List[TableRowPayload] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


class Chunk(BaseModel):
    """A chunk of text ready for embedding and ingestion."""

    chunk_id: str
    doc_id: str
    heading: str
    subheading: Optional[str] = None
    body: str
    breadcrumbs: List[str] = Field(default_factory=list)
    relationships: List[Any] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChunkStats(BaseModel):
    """Statistics about chunking results."""

    doc_id: str
    total_paragraphs: int
    total_chunks: int
    min_tokens: int
    max_tokens: int
    mean_tokens: float


__all__ = [
    "Chunk",
    "ChunkStats",
    "DocumentPayload",
    "Paragraph",
    "ScrapeArtifact",
    "Section",
    "Subsection",
    "TableRowPayload",
]
