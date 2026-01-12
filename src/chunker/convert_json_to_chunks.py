"""Convert page-based JSON (from LlamaParse or similar) to JSONL chunk format."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .chunk_types import Chunk

logger = logging.getLogger(__name__)


def convert_json_to_chunks(
    input_path: Path,
    doc_id: Optional[str] = None,
    min_body_chars: int = 50,
) -> List[Chunk]:
    """Convert page-based JSON to list of Chunks.

    Args:
        input_path: Path to the JSON file with pages structure.
        doc_id: Document ID to use. If None, derived from filename.
        min_body_chars: Minimum characters for a chunk body.

    Returns:
        List of Chunk objects ready for JSONL output.
    """
    if doc_id is None:
        doc_id = input_path.stem.replace(" ", "_").lower()

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pages = data.get("pages", [])
    if not pages:
        raise ValueError(f"No pages found in {input_path}")

    chunks: List[Chunk] = []
    current_h1: str = "Document"
    current_h2: Optional[str] = None
    pending_texts: List[str] = []
    pending_pages: List[int] = []
    chunk_counter = 0

    def flush_chunk() -> None:
        """Emit accumulated text as a chunk."""
        nonlocal chunk_counter, pending_texts, pending_pages

        if not pending_texts:
            return

        body = "\n".join(pending_texts).strip()
        if len(body) < min_body_chars:
            pending_texts = []
            pending_pages = []
            return

        chunk_counter += 1
        chunk_id = f"{doc_id}_chunk_{chunk_counter:04d}"

        heading = current_h1
        subheading = current_h2 or "Body"
        breadcrumbs = [heading]
        if current_h2:
            breadcrumbs.append(current_h2)

        metadata: Dict[str, Any] = {
            "page_numbers": sorted(set(pending_pages)),
            "headings": breadcrumbs,
            "content_type": "text",
            "origin_filename": input_path.name,
        }

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                heading=heading,
                subheading=subheading,
                body=body,
                breadcrumbs=breadcrumbs,
                relationships=[],
                metadata=metadata,
            )
        )
        pending_texts = []
        pending_pages = []

    for page_data in pages:
        page_num = page_data.get("page", 0)
        items = page_data.get("items", [])

        for item in items:
            item_type = item.get("type", "")
            value = item.get("value", "").strip()
            level = item.get("lvl", 0)

            if not value:
                continue

            # Skip page numbers and short headers/footers
            if len(value) < 10 and value.isdigit():
                continue
            if value.lower().startswith("the beige book") and len(value) < 20:
                continue

            if item_type == "heading":
                if level == 1:
                    # New H1 section - flush current and start new
                    flush_chunk()
                    current_h1 = value
                    current_h2 = None
                elif level == 2:
                    # New H2 subsection - flush current
                    flush_chunk()
                    current_h2 = value
            elif item_type == "text":
                # Accumulate text content
                if value and len(value) > 10:
                    pending_texts.append(value)
                    pending_pages.append(page_num)

    # Flush any remaining content
    flush_chunk()

    logger.info("Converted %s → %d chunks", input_path.name, len(chunks))
    return chunks


def save_chunks_jsonl(chunks: List[Chunk], output_path: Path) -> None:
    """Save chunks to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.model_dump()) + "\n")
    logger.info("Saved %d chunks to %s", len(chunks), output_path)


def main():
    """CLI entry point for conversion."""
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Convert JSON to JSONL chunks")
    parser.add_argument("input_file", type=Path, help="Input JSON file")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output JSONL file (default: SAVED/<doc_id>.jsonl)",
    )
    parser.add_argument("--doc-id", type=str, default=None, help="Document ID")

    args = parser.parse_args()

    doc_id = args.doc_id or args.input_file.stem.replace(" ", "_").lower()
    output_path = args.output or Path("src/chunker/SAVED") / f"{doc_id}.jsonl"

    chunks = convert_json_to_chunks(args.input_file, doc_id=doc_id)
    save_chunks_jsonl(chunks, output_path)

    print(f"Converted {args.input_file} → {output_path}")
    print(f"Total chunks: {len(chunks)}")


if __name__ == "__main__":
    main()
