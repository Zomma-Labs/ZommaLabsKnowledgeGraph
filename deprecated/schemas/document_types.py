"""Shared data structures for the GraphRAG workflow."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

from typing_extensions import TypedDict

from pydantic import BaseModel, Field


class DocumentPayload(BaseModel):
    doc_id: str
    path: Path
    bytes: Optional[bytes] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    @property
    def size_bytes(self) -> Optional[int]:
        """Return the byte size extracted during document discovery, if any."""

        return self.metadata.get("size")


class Paragraph(BaseModel):
    paragraph_id: str
    text: str
    page: Optional[int] = None
    doc_item_ref: Optional[str] = None


class Subsection(BaseModel):
    subheading: str
    paragraphs: List[Paragraph] = Field(default_factory=list)


class Section(BaseModel):
    heading: str
    order: int
    subsections: List[Subsection] = Field(default_factory=list)


class ScrapeArtifact(BaseModel):
    doc_id: str
    sections: List[Section] = Field(default_factory=list)
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    figures: List[Dict[str, Any]] = Field(default_factory=list)
    source_path: Optional[Path] = None
    docling_json: Optional[Dict[str, Any]] = None
    doc_item_breadcrumbs: Dict[str, List[str]] = Field(default_factory=dict)
    table_rows: List["TableRowPayload"] = Field(default_factory=list)

    model_config = {"json_encoders": {Path: str}}


class TableRowPayload(BaseModel):
    table_ref: str
    row_index: int
    headers: List[str]
    values: Dict[str, Any]
    page: Optional[int] = None
    label: Optional[str] = None


class ChunkMetadata(TypedDict, total=False):
    doc_item_refs: List[str]
    doc_items_count: int
    page_numbers: List[int]
    headings: List[str]
    docling_headings: List[str]
    origin_filename: Optional[str]
    content_type: Literal["text", "table_row"]
    table_ref: Optional[str]
    row_index: Optional[int]
    table_headers: List[str]
    table_row: Dict[str, Any]
    table_label: Optional[str]
    doc_date: Optional[str]


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    heading: str
    subheading: Optional[str] = None
    body: str
    breadcrumbs: Sequence[str] = Field(default_factory=list)
    relationships: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: ChunkMetadata = Field(default_factory=dict)


class ChunkStats(BaseModel):
    doc_id: str
    total_paragraphs: int
    total_chunks: int
    min_tokens: int
    max_tokens: int
    mean_tokens: float



