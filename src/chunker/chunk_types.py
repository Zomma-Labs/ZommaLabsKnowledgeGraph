"""Shared data structures for the chunking workflow."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentPayload(BaseModel):
    """Metadata and optional bytes for a document to be processed."""

    doc_id: str
    path: Optional[Path] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    bytes: Optional[bytes] = None

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


__all__ = [
    "Chunk",
    "DocumentPayload",
]
