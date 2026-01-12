"""Document loader utilities."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator


PDF_DIR = Path("data")
from .chunk_types import DocumentPayload

logger = logging.getLogger(__name__)


def document_id_from_path(path: Path) -> str:
    """Generate the canonical doc_id slug used throughout the pipeline.

    Args:
        path: Filesystem path to the PDF asset.

    Returns:
        Lowercase identifier derived from the filename so downstream
        components reference the same document key.
    """

    return path.stem.replace(" ", "_").lower()


def iter_pdf_paths() -> Iterator[Path]:
    """Yield every PDF under the configured input directory.

    Yields:
        Paths for each readable PDF so scrapers/chunkers can iterate over
        the corpus without reimplementing directory traversal.
    """

    if not PDF_DIR.exists():
        logger.warning("PDF directory %s does not exist", PDF_DIR)
        return
    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        if pdf_path.is_file():
            yield pdf_path


def build_payload(path: Path) -> DocumentPayload:
    """Construct a DocumentPayload with metadata inferred from the file.

    Args:
        path: PDF path selected by the loader.

    Returns:
        DocumentPayload describing the file location, doc_id, and basic
        metadata (size/modified timestamps) so later stages can enrich it.
    """

    stat = path.stat()
    metadata = {"size": stat.st_size, "modified": stat.st_mtime}
    return DocumentPayload(doc_id=document_id_from_path(path), path=path, metadata=metadata)


def iter_document_payloads(load_bytes: bool = False) -> Iterator[DocumentPayload]:
    """Yield DocumentPayload objects for every discovered PDF.

    Args:
        load_bytes: When True, eagerly read the PDF bytes for consumers that
            require in-memory payloads (e.g., Docling converters).

    Yields:
        Fully-initialized DocumentPayload instances ready for scraping.
    """

    seen: set[str] = set()
    for path in iter_pdf_paths():
        doc_id = document_id_from_path(path)
        if doc_id in seen:
            raise RuntimeError(
                f"Duplicate doc_id derived from filename stem '{doc_id}'. "
                "Rename the PDF to ensure unique stems before ingest."
            )
        seen.add(doc_id)

        payload = build_payload(path)
        payload.doc_id = doc_id

        if load_bytes:
            payload.bytes = path.read_bytes()
        yield payload


if __name__ == "__main__":
    docs = list(iter_document_payloads())
    logger.info("Discovered %d documents under %s", len(docs), PDF_DIR)
    for doc in docs:
        print(doc.doc_id, doc.path)
