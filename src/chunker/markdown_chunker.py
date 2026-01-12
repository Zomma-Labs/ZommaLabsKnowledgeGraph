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

        # Skip if it's just a horizontal rule or list items (table of contents)
        if paragraph_text.startswith('---') or paragraph_text.startswith('* '):
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

    for line in lines:
        # Check if line is a header
        header_match = re.match(r'^(#{1,6})\s+(.+)$', line)

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
    import sys

    # Test with the beige book markdown
    test_file = Path(__file__).parent / "test_pdfs" / "biege.md"

    if test_file.exists():
        chunks = chunk_markdown_file(test_file, doc_id="beigebook_nov2025")

        print(f"Generated {len(chunks)} chunks\n")

        # Show some example chunks
        print("=" * 80)
        print("SAMPLE CHUNKS:")
        print("=" * 80)

        # Show chunks from different sections
        samples = [
            ("National Summary", 3),
            ("Highlights", 5),
            ("Federal Reserve Bank of Cleveland", 3),
        ]

        for section_keyword, count in samples:
            print(f"\n--- {section_keyword} chunks ---")
            matching = [c for c in chunks if section_keyword in c.header_path]
            for chunk in matching[:count]:
                print(f"\nHeader path: {chunk.header_path}")
                print(f"Body preview: {chunk.body[:200]}...")
                print("-" * 40)
    else:
        print(f"Test file not found: {test_file}")
        sys.exit(1)
