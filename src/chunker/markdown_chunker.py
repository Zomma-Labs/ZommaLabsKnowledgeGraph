"""
Markdown chunker that preserves header hierarchy.

Splits by paragraphs, each paragraph inherits its parent header path as breadcrumbs.
"""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    body: str
    breadcrumbs: list[str]
    header_path: str  # breadcrumbs joined with " > "


def parse_markdown_to_chunks(
    markdown_text: str,
    doc_id: str,
    skip_short_paragraphs: int = 50,  # Skip paragraphs shorter than this
) -> list[Chunk]:
    """
    Parse markdown into chunks, one per paragraph, with header hierarchy preserved.

    Args:
        markdown_text: The full markdown content
        doc_id: Document identifier for chunk IDs
        skip_short_paragraphs: Minimum paragraph length to include (chars)

    Returns:
        List of Chunk objects with breadcrumbs
    """
    lines = markdown_text.split('\n')
    chunks = []
    chunk_counter = 0

    # Track current header stack: [(level, title), ...]
    header_stack: list[tuple[int, str]] = []

    # Accumulate paragraph lines
    current_paragraph_lines: list[str] = []

    def get_breadcrumbs() -> list[str]:
        """Get current breadcrumb list from header stack."""
        return [title for _, title in header_stack]

    def flush_paragraph():
        """Save current paragraph as a chunk if non-empty."""
        nonlocal chunk_counter

        if not current_paragraph_lines:
            return

        paragraph_text = '\n'.join(current_paragraph_lines).strip()

        # Skip if too short or just whitespace/separators
        if len(paragraph_text) < skip_short_paragraphs:
            current_paragraph_lines.clear()
            return

        # Skip if it's just a horizontal rule
        if paragraph_text.startswith('---'):
            current_paragraph_lines.clear()
            return

        chunk_counter += 1
        breadcrumbs = get_breadcrumbs()

        chunks.append(Chunk(
            chunk_id=f"{doc_id}_chunk_{chunk_counter:04d}",
            doc_id=doc_id,
            body=paragraph_text,
            breadcrumbs=breadcrumbs,
            header_path=" > ".join(breadcrumbs) if breadcrumbs else "Document"
        ))

        current_paragraph_lines.clear()

    # Track if we're inside an HTML table
    inside_table = False

    for line in lines:
        # Check if line is a header
        header_match = re.match(r'^(#{1,6})\s+(.+)$', line)

        # Check for HTML table boundaries
        if '<table' in line.lower():
            # Flush any accumulated paragraph before starting table
            flush_paragraph()
            inside_table = True
            current_paragraph_lines.append(line)
            # Check if table also closes on same line
            if '</table>' in line.lower():
                inside_table = False
                flush_paragraph()
            continue

        if inside_table:
            # Accumulate table lines without splitting on blank lines
            current_paragraph_lines.append(line)
            if '</table>' in line.lower():
                inside_table = False
                flush_paragraph()
            continue

        if header_match:
            # Flush any accumulated paragraph before changing headers
            flush_paragraph()

            level = len(header_match.group(1))
            title = header_match.group(2).strip()

            # Pop headers at same or deeper level
            while header_stack and header_stack[-1][0] >= level:
                header_stack.pop()

            # Push new header
            header_stack.append((level, title))

        elif line.strip() == '':
            # Blank line = end of paragraph
            flush_paragraph()

        else:
            # Regular content line
            current_paragraph_lines.append(line)

    # Flush any remaining paragraph
    flush_paragraph()

    return chunks


def chunk_markdown_file(
    filepath: str | Path,
    doc_id: str | None = None,
    skip_short_paragraphs: int = 50,
) -> list[Chunk]:
    """
    Chunk a markdown file.

    Args:
        filepath: Path to the markdown file
        doc_id: Document ID (defaults to filename stem)
        skip_short_paragraphs: Minimum paragraph length

    Returns:
        List of Chunk objects
    """
    filepath = Path(filepath)

    if doc_id is None:
        doc_id = filepath.stem

    markdown_text = filepath.read_text(encoding='utf-8')
    return parse_markdown_to_chunks(markdown_text, doc_id, skip_short_paragraphs)


def chunks_to_jsonl(chunks: list[Chunk], output_path: str | Path) -> None:
    """Write chunks to JSONL format compatible with pipeline."""
    import json

    output_path = Path(output_path)

    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            record = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "body": chunk.body,
                "breadcrumbs": chunk.breadcrumbs,
                "header_path": chunk.header_path,
            }
            f.write(json.dumps(record) + '\n')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chunk markdown files into JSONL format")
    parser.add_argument("input", help="Path to markdown file to chunk")
    parser.add_argument("--output", "-o", help="Output JSONL path (default: SAVED/<doc_id>.jsonl)")
    parser.add_argument("--doc-id", help="Document ID (default: input filename stem)")
    parser.add_argument("--min-length", type=int, default=50, help="Minimum paragraph length (default: 50)")
    parser.add_argument("--preview", action="store_true", help="Show sample chunks after processing")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        raise SystemExit(1)

    doc_id = args.doc_id or input_path.stem.replace(" ", "_").lower()
    chunks = chunk_markdown_file(input_path, doc_id=doc_id, skip_short_paragraphs=args.min_length)

    print(f"Generated {len(chunks)} chunks")

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(__file__).parent / "SAVED" / f"{doc_id}.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_to_jsonl(chunks, output_path)
    print(f"Saved to: {output_path}")

    # Show preview if requested
    if args.preview and chunks:
        print("\n" + "=" * 80)
        print("SAMPLE CHUNKS:")
        print("=" * 80)
        for chunk in chunks[:5]:
            print(f"\nHeader path: {chunk.header_path}")
            print(f"Body preview: {chunk.body[:200]}...")
            print("-" * 40)
