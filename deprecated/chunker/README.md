# Contextual Chunker & Scraper

The **Chunker** module is a high-fidelity ingestion engine designed to transform complex PDFs into semantically rich, context-aware chunks. Unlike naive splitters, it preserves the document's hierarchical structure (Sections > Subsections) and "atomizes" tables into queryable row-level units.

Powered by [Docling](https://github.com/DS4SD/docling), it handles OCR, layout analysis, and table structure recognition automatically.

## Key Features

- **Hierarchical Scraping**: Parses PDFs into a strict tree of Sections, Subsections, and Paragraphs.
- **Contextual Grounding**: Every chunk carries its "Breadcrumb Trail" (e.g., `Risk Factors > Market Risks > Interest Rates`), ensuring LLMs know *where* a fact came from.
- **Table Atomization**: Tables are exploded into individual rows. Each row chunk contains its headers and values as structured JSON, enabling precise SQL-like reasoning over tabular data.
- **Hybrid Chunking**: Combines structural boundaries with semantic grouping to prevent mid-sentence splits.

## Pipeline Flow

1.  **Ingestion (Scraper)**:
    -   The `StructuralScraper` uses Docling to parse the PDF.
    -   It builds a `ScrapeArtifact` containing the document tree and extracted tables.
    -   **OCR**: Automatically applies RapidOCR for scanned documents.

2.  **Contextualization**:
    -   A "Breadcrumb Index" is built, mapping every paragraph and table row to its parent headings.

3.  **Chunking (Chunker)**:
    -   **Text**: Processed via `HybridChunker` to respect sentence and paragraph boundaries.
    -   **Tables**: Converted into `TableRowPayloads`. Each row becomes a standalone chunk with:
        -   `body`: A human-readable summary (e.g., "Table: Revenue 2023...").
        -   `metadata`: The raw JSON key-value pairs of the row.

## Data Structure

### The Chunk Object
Every chunk produced is a strict Pydantic model containing:

- `chunk_id`: Unique identifier.
- `body`: The actual text content (or table row representation).
- `breadcrumbs`: List of headings leading to this chunk.
- `metadata`: Rich provenance info (page numbers, bounding boxes, original filename).

### Table Handling
Instead of treating tables as large text blocks, we treat them as collections of records:

**Input Table:**
| Year | Revenue |
|------|---------|
| 2023 | $10M    |

**Output Chunk (Row 1):**
```json
{
  "body": "Table: Financials\n{\"Year\":\"2023\",\"Revenue\":\"$10M\"}",
  "breadcrumbs": ["Financial Results", "Annual Summary"],
  "metadata": {
    "content_type": "table_row",
    "table_row": {"Year": "2023", "Revenue": "$10M"}
  }
}
```

## Usage

```python
from chunker.scraper import StructuralScraper
from chunker.chunker import ContextualChunker
from chunker.loader import iter_document_payloads

# 1. Initialize
scraper = StructuralScraper()
chunker = ContextualChunker()

# 2. Process
for payload in iter_document_payloads(load_bytes=True):
    # Scrape PDF -> Structured Artifact
    artifact = scraper.scrape(payload)
    
    # Artifact -> Contextual Chunks
    chunks = chunker.build_chunks(artifact)
    
    print(f"Generated {len(chunks)} chunks for {payload.doc_id}")
```
