"""Chunk builder backed by Docling's HybridChunker."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from docling.chunking import HybridChunker
from docling.datamodel.document import DoclingDocument
from docling_core.transforms.chunker.hierarchical_chunker import DocChunk




CHUNK_DIR = Path("./SAVED")
SAVE_SCRAPE = True
SAVE_CHUNKS = True
from .loader import iter_document_payloads
from .scraper import StructuralScraper
from .types import Chunk, ChunkStats, ScrapeArtifact, TableRowPayload

logger = logging.getLogger(__name__)


class ContextualChunker:
    """Use Docling HybridChunker to build breadcrumb-aware chunks."""

    def __init__(self, chunker: Optional[HybridChunker] = None) -> None:
        """Configure the Docling chunker wrapper.

        Args:
            chunker: Optional pre-built HybridChunker for testing overrides.
        """

        self._hybrid_chunker = chunker or HybridChunker()

    def build_chunks(self, artifact: ScrapeArtifact) -> List[Chunk]:
        """Convert a ScrapeArtifact into breadcrumb-aware Chunk objects."""

        if not artifact.docling_json:
            raise ValueError(
                "ScrapeArtifact missing Docling payload. Ensure StructuralScraper stores docling_json."
            )

        dl_document = DoclingDocument.model_validate(artifact.docling_json)
        docling_chunks = list(self._hybrid_chunker.chunk(dl_doc=dl_document))
        breadcrumb_lookup = artifact.doc_item_breadcrumbs or {}
        table_row_lookup = self._build_table_row_lookup(artifact.table_rows)
        table_rows_by_ref = self._group_rows_by_table_ref(artifact.table_rows)

        chunks: List[Chunk] = []
        processed_tables: Set[str] = set()
        chunk_counter = 0
        for docling_chunk in docling_chunks:
            meta = getattr(docling_chunk, "meta", None)
            if meta is None:
                raise ValueError("Docling chunk missing meta field")

            doc_items = list(getattr(meta, "doc_items", []) or [])
            if not doc_items:
                raise ValueError("Docling chunk missing doc_items list")

            docling_headings = [
                str(h).strip() for h in (getattr(meta, "headings", []) or []) if str(h).strip()
            ]
            doc_item_refs = [item.self_ref for item in doc_items if getattr(item, "self_ref", None)]
            table_row_refs = [ref for ref in doc_item_refs if ref and "::row::" in ref]
            origin_filename = getattr(getattr(meta, "origin", None), "filename", None)

            if table_row_refs:
                emitted: Set[str] = set()
                for row_ref in table_row_refs:
                    if row_ref in emitted:
                        continue
                    emitted.add(row_ref)
                    row_payload = table_row_lookup.get(row_ref)
                    if not row_payload:
                        continue
                    chunk_counter += 1
                    chunk_id = f"{artifact.doc_id}_chunk_{chunk_counter:04d}"
                    chunks.append(
                        self._build_table_row_chunk(
                            chunk_id,
                            artifact.doc_id,
                            row_payload,
                            breadcrumb_lookup,
                            docling_headings,
                            origin_filename,
                        )
                    )
                continue

            table_refs = [ref for ref in doc_item_refs if ref and ref.startswith("#/tables/")]
            unprocessed = [ref for ref in table_refs if ref not in processed_tables]
            if unprocessed:
                for table_ref in unprocessed:
                    processed_tables.add(table_ref)
                    rows = table_rows_by_ref.get(table_ref, [])
                    if not rows:
                        continue
                    for row_payload in rows:
                        chunk_counter += 1
                        chunk_id = f"{artifact.doc_id}_chunk_{chunk_counter:04d}"
                        chunks.append(
                            self._build_table_row_chunk(
                                chunk_id,
                                artifact.doc_id,
                                row_payload,
                                breadcrumb_lookup,
                                docling_headings,
                                origin_filename,
                            )
                        )
                continue

            chunk_counter += 1
            chunk_id = f"{artifact.doc_id}_chunk_{chunk_counter:04d}"
            chunks.append(
                self._build_text_chunk(
                    doc_id=artifact.doc_id,
                    chunk_id=chunk_id,
                    docling_chunk=docling_chunk,
                    breadcrumb_lookup=breadcrumb_lookup,
                )
            )
        logger.info("Built %d chunks for %s via Docling HybridChunker", len(chunks), artifact.doc_id)
        return chunks

    def save_chunks(self, doc_id: str, chunks: Iterable[Chunk]) -> str:
        """Persist chunk JSONL files for inspection and debugging."""

        CHUNK_DIR.mkdir(parents=True, exist_ok=True)
        output_path = CHUNK_DIR / f"{doc_id}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(json.dumps(chunk.model_dump()) + "\n")
        logger.info("Persisted %s", output_path)
        return str(output_path)

    def _build_text_chunk(
        self,
        *,
        doc_id: str,
        chunk_id: str,
        docling_chunk: DocChunk,
        breadcrumb_lookup: Dict[str, List[str]],
    ) -> Chunk:
        """Translate a Docling chunk containing prose/text into our schema."""

        enriched_text = self._hybrid_chunker.contextualize(docling_chunk)
        meta = getattr(docling_chunk, "meta", None)
        if meta is None:
            raise ValueError("Docling chunk missing meta field")

        doc_items = list(getattr(meta, "doc_items", []) or [])
        if not doc_items:
            raise ValueError("Docling chunk missing doc_items list")

        docling_headings = [
            str(h).strip() for h in (getattr(meta, "headings", []) or []) if str(h).strip()
        ]

        doc_item_refs = [item.self_ref for item in doc_items if getattr(item, "self_ref", None)]
        page_numbers = sorted(
            {
                prov.page_no
                for item in doc_items
                for prov in getattr(item, "prov", []) or []
                if getattr(prov, "page_no", None) is not None
            }
        )

        breadcrumbs = self._resolve_breadcrumbs(doc_item_refs, docling_headings, breadcrumb_lookup)
        heading = breadcrumbs[0] if breadcrumbs else "Document"
        subheading = breadcrumbs[1] if len(breadcrumbs) > 1 else None

        metadata = {
            "doc_item_refs": doc_item_refs,
            "doc_items_count": len(doc_items),
            "page_numbers": page_numbers,
            "headings": breadcrumbs,
            "docling_headings": docling_headings,
            "origin_filename": getattr(getattr(meta, "origin", None), "filename", None),
            "content_type": "text",
        }

        return Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            heading=heading,
            subheading=subheading,
            body=enriched_text,
            breadcrumbs=breadcrumbs,
            relationships=[],
            metadata=metadata,
        )

    @staticmethod
    def _resolve_breadcrumbs(
        doc_item_refs: List[str],
        docling_headings: List[str],
        breadcrumb_lookup: Dict[str, List[str]],
    ) -> List[str]:
        """Resolve the most specific breadcrumb trail for a doc_item."""

        for ref in doc_item_refs:
            crumbs = breadcrumb_lookup.get(ref)
            if crumbs:
                return crumbs
        return docling_headings or ["Document"]

    @staticmethod
    def summarize(doc_id: str, chunks: Iterable[Chunk]) -> ChunkStats:
        """Compute basic statistics to validate chunking quality."""

        chunk_list = list(chunks)
        token_counts = [ContextualChunker._estimate_tokens(c.body) for c in chunk_list]
        total_units = 0
        for chunk in chunk_list:
            doc_item_refs = chunk.metadata.get("doc_item_refs") if isinstance(chunk.metadata, dict) else None
            if doc_item_refs is None:
                raise ValueError(f"Chunk {chunk.chunk_id} missing doc_item_refs metadata")
            total_units += len(doc_item_refs)
        return ChunkStats(
            doc_id=doc_id,
            total_paragraphs=total_units,
            total_chunks=len(chunk_list),
            min_tokens=min(token_counts) if token_counts else 0,
            max_tokens=max(token_counts) if token_counts else 0,
            mean_tokens=(sum(token_counts) / len(token_counts)) if token_counts else 0.0,
        )

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rudimentary token estimator based on whitespace splitting."""

        if not text:
            return 0
        return max(1, len(text.split()))

    @staticmethod
    def _build_table_row_lookup(rows: Sequence[TableRowPayload]) -> Dict[str, TableRowPayload]:
        """Return a doc_item_ref → TableRowPayload map for quick lookups."""

        lookup: Dict[str, TableRowPayload] = {}
        for row in rows or []:
            if not row.table_ref:
                continue
            ref = f"{row.table_ref}::row::{row.row_index}"
            lookup[ref] = row
        return lookup

    @staticmethod
    def _group_rows_by_table_ref(rows: Sequence[TableRowPayload]) -> Dict[str, List[TableRowPayload]]:
        """Group table rows by their originating table reference."""

        grouped: Dict[str, List[TableRowPayload]] = {}
        for row in rows or []:
            if not row.table_ref:
                continue
            grouped.setdefault(row.table_ref, []).append(row)
        for row_list in grouped.values():
            row_list.sort(key=lambda r: r.row_index)
        return grouped

    @staticmethod
    def _format_table_row_body(row: TableRowPayload) -> str:
        """Format table row chunks as the mandated two-line structure."""

        label = (row.label or row.table_ref or "Table Row").strip()
        if row.page is not None:
            header = f"Table: {label} (Page {row.page})"
        else:
            header = f"Table: {label}"
        row_json = json.dumps(dict(row.values), separators=(",", ":"))
        return f"{header}\n{row_json}"

    def _build_table_row_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        row_payload: TableRowPayload,
        breadcrumb_lookup: Dict[str, List[str]],
        docling_headings: Sequence[str],
        origin_filename: Optional[str],
    ) -> Chunk:
        """Construct a Chunk object for a single table row."""

        row_ref = f"{row_payload.table_ref}::row::{row_payload.row_index}"
        breadcrumbs = self._resolve_breadcrumbs([row_ref], list(docling_headings), breadcrumb_lookup)
        heading = breadcrumbs[0] if breadcrumbs else "Document"
        subheading = breadcrumbs[1] if len(breadcrumbs) > 1 else None
        page_numbers = [row_payload.page] if row_payload.page is not None else []
        metadata: Dict[str, Any] = {
            "doc_item_refs": [row_ref],
            "doc_items_count": 1,
            "page_numbers": page_numbers,
            "headings": breadcrumbs,
            "docling_headings": list(docling_headings),
            "origin_filename": origin_filename,
            "content_type": "table_row",
            "table_ref": row_payload.table_ref or None,
            "row_index": row_payload.row_index,
            "table_headers": list(row_payload.headers),
            "table_row": dict(row_payload.values),
            "table_label": row_payload.label,
        }

        return Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            heading=heading,
            subheading=subheading,
            body=self._format_table_row_body(row_payload),
            breadcrumbs=breadcrumbs,
            relationships=[],
            metadata=metadata,
        )


__all__ = ["ContextualChunker", "Chunk", "ChunkStats"]


if __name__ == "__main__":
    scraper = StructuralScraper()
    chunker = ContextualChunker()
    for payload in iter_document_payloads(load_bytes=True):
        artifact = scraper.scrape(payload)
        if SAVE_SCRAPE:
            scraper.save_artifact(artifact)
        chunks = chunker.build_chunks(artifact)
        if SAVE_CHUNKS:
            chunker.save_chunks(payload.doc_id, chunks)
        stats = ContextualChunker.summarize(payload.doc_id, chunks)
        logger.info(
            "Chunked %s: chunks=%d min=%d max=%d mean=%.1f",
            payload.doc_id,
            stats.total_chunks,
            stats.min_tokens,
            stats.max_tokens,
            stats.mean_tokens,
        )
