# ZommaLabsKG

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A knowledge graph ingestion pipeline that transforms financial documents (10-Ks, Earnings Calls, Press Releases, Fed Reports) into a queryable knowledge graph stored in Neo4j.

## Features

- **Gemini-Powered Chunking**: PDF → Markdown conversion using Gemini 2.5 Pro with header-based splitting
- **Chain-of-Thought Extraction**: Two-step LLM extraction that first enumerates all entities, then generates relationships
- **Hybrid Entity Deduplication**: Embedding clustering + LLM verification to resolve references ("Tim Cook" = "Timothy Cook")
- **Subsidiary Awareness**: Keeps related entities separate (AWS ≠ Amazon, YouTube ≠ Google)
- **Topic Resolution**: Matches extracted topics against a curated financial ontology
- **Chunk-Centric Provenance**: All facts link back to source text for traceability
- **Multi-Tenant Support**: `group_id` isolation for multiple data sources
- **V6 Query Pipeline**: Fast threshold-only retrieval (~1-2 min/query) with configurable precision/recall tradeoffs

---

## Architecture

### High-Level Overview

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    PDF      │───►│  Chunking   │───►│  Pipeline   │───►│   Neo4j     │
│  Documents  │    │  (Gemini)   │    │  (3-Phase)  │    │   Graph     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                          │                                      │
                          ▼                                      ▼
                   ┌─────────────┐                        ┌─────────────┐
                   │   JSONL     │                        │ V6 Querier  │
                   │   Chunks    │                        │  Pipeline   │
                   └─────────────┘                        └─────────────┘
```

---

## Chunking System

The chunking system uses **Gemini 2.5 Pro** for high-quality PDF → Markdown conversion, followed by header-based splitting.

### Two-Step Process

**Step 1: PDF → Markdown** (`src/chunker/pdf_to_markdown.py`)
- Sends entire PDF to Gemini 2.5 Pro
- Preserves structural elements (headings, lists, tables)
- Tables converted to HTML format for accurate preservation
- Headers/footers automatically removed

**Step 2: Markdown → Chunks** (`src/chunker/markdown_chunker.py`)
- Splits on paragraph boundaries (blank lines)
- Tracks header hierarchy for breadcrumb context
- HTML tables kept intact as single chunks
- Each chunk carries its `header_path` (e.g., "Federal Reserve Bank of Boston > Labor Markets")

### Usage

```bash
# Convert PDF to markdown
uv run src/chunker/pdf_to_markdown.py data/document.pdf -o output.md

# Chunk markdown to JSONL (programmatic)
python -c "
from src.chunker.markdown_chunker import chunk_markdown_file, chunks_to_jsonl
chunks = chunk_markdown_file('output.md', doc_id='my_document')
chunks_to_jsonl(chunks, 'src/chunker/SAVED/my_document.jsonl')
"
```

### Chunk Schema

```python
@dataclass
class Chunk:
    chunk_id: str          # "doc_chunk_0001"
    doc_id: str            # Document identifier
    body: str              # The text content
    breadcrumbs: list[str] # ["Section", "Subsection", ...]
    header_path: str       # "Section > Subsection > ..."
```

---

## Ingestion Pipeline

### Three-Phase Architecture (`src/pipeline.py`)

```
Phase 1: PARALLEL EXTRACTION (LLM-heavy)
├── ExtractorV2: Chain-of-thought extraction
│   1. Enumerate ALL entities in text
│   2. Generate relationships between entities
│   3. Critique step (reflexion) catches missed facts
│   4. Re-extract if critique found issues
└── Semaphore(100) limits concurrent LLM calls

Phase 2: RESOLUTION (Per-document deduplication)
├── 2a-c: IN-DOCUMENT DEDUP
│   • Build similarity graph (cosine > 0.70)
│   • Find connected components (Union-Find)
│   • LLM verifies which entities are the same
├── 2d: GRAPH RESOLUTION
│   • Vector search Neo4j for existing entities
│   • LLM verifies matches
│   • Merge summaries if match found
└── 2e: TOPIC RESOLUTION
    • Vector search Qdrant ontology
    • LLM verifies candidate matches

Phase 3: ASSEMBLY (Bulk write to Neo4j)
├── Collect all operations into BulkWriteBuffer
├── Batch generate embeddings (avoids rate limits)
└── Bulk write with UNWIND queries
```

### Graph Schema (Chunk-Centric Provenance)

All relationships flow through chunks for full traceability:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ FACT PATTERN (Subject → Chunk → Object)                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  (EntityNode OR TopicNode)                    (EntityNode OR TopicNode)     │
│         Subject                                       Object                │
│            │                                            ▲                   │
│            │ [REL {fact_id, description, date_context}] │                   │
│            ▼                                            │                   │
│      (EpisodicNode) ─────────────────────────────────────                   │
│            │         [REL_TARGET {fact_id, ...}]                            │
│            │                                                                │
│            ├─[CONTAINS_FACT]─► (FactNode)                                   │
│            │                                                                │
│            └─[DISCUSSES]─► (TopicNode)  ← Thematic topics from fact.topics  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ DOCUMENT STRUCTURE                                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  (DocumentNode) ─[CONTAINS_CHUNK]─► (EpisodicNode)                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Points:**
- **Subject/Object can be EntityNode OR TopicNode** — Topics like "Economic Activity" or "Inflation" can be fact subjects/objects
- **DISCUSSES is separate** — Links chunks to thematic topics from `fact.topics` list (e.g., "Labor Markets", "Consumer Spending")
- **REL types are dynamic** — Relationship names are normalized from free-form text (e.g., "reported growth in" → `REPORTED_GROWTH_IN`)

**Node Types:**
| Node | Description |
|------|-------------|
| **DocumentNode** | Parent container for source file metadata |
| **EpisodicNode** | Text chunk with `header_path` context (the provenance hub) |
| **EntityNode** | Deduplicated real-world entities (Companies, People, Organizations, Locations) |
| **FactNode** | Atomic facts with embeddings for semantic search |
| **TopicNode** | Financial themes/topics — can be fact subjects/objects OR thematic tags via DISCUSSES |

### LLM Usage by Step

| Step | Model | Purpose |
|------|-------|---------|
| Extraction | `gpt-5.2` | Entity & relationship extraction |
| Critique | `gpt-5.1` | Quality check, catch missed facts |
| In-doc Dedup | `gpt-5.1` | Verify entity clusters within document |
| Graph Match | `gpt-5-mini` | Fast entity lookup verification |
| Summary Merge | `gemini-2.5-flash-lite` | Merge entity descriptions |
| Topic Resolution | `gemini-2.5-flash-lite` | Topic matching & verification |

### Running the Pipeline

```bash
# Basic run (processes all chunks in src/chunker/SAVED/)
uv run src/pipeline.py

# Limit chunks for testing
uv run src/pipeline.py --limit 5

# Adjust extraction parallelism (default: 100)
uv run src/pipeline.py --concurrency 50

# Adjust entity resolution parallelism (default: 50)
uv run src/pipeline.py --resolve-concurrency 30

# Adjust deduplication parallelism (default: 20)
uv run src/pipeline.py --dedup-concurrency 10

# Adjust similarity threshold for dedup (default: 0.70)
uv run src/pipeline.py --similarity-threshold 0.75

# Filter by filename
uv run src/pipeline.py --filter beige

# Multi-tenant isolation
uv run src/pipeline.py --group-id tenant1

# Verbose logging
VERBOSE=true uv run src/pipeline.py
```

---

## V6 Query Pipeline

The V6 query system uses **threshold-only retrieval** for fast, accurate question answering over the knowledge graph.

### Key Innovation

V6 eliminates LLM-based fact scoring by using OpenAI's `text-embedding-3-large` (3072 dimensions), which provides ~3x better score separation than previous embeddings. A simple threshold filter (0.65) replaces expensive LLM scoring calls, achieving **5-10x faster queries**.

### Three-Phase Query Architecture

```
Phase 1: DECOMPOSITION
└── Break question into focused sub-queries
    • Returns: SubQueries + EntityHints + TopicHints + QuestionType

Phase 2: PARALLEL RESEARCH (per sub-query)
└── Each Researcher handles one sub-query end-to-end:

    Step 1: Resolution
            • Vector search + LLM verify for entities/topics

    Step 2: Dual-Path Retrieval (parallel)
            • Scoped searches (per resolved entity/topic)
            • Global search (always runs for coverage)

    Step 3: Threshold Filter (NO LLM - fast!)
            • Filter: vector_score >= 0.65
            • Cross-source boost: +0.15 per additional retrieval path

    Step 4: Gap Expansion (optional)
            • 1-hop graph expansion if gaps detected

    Step 5: Entity Drilldown (for ENUMERATION questions)
            • LLM selects entities needing more context

    Step 5.5: LLM Fact Filter (optional, for precision)
            • LLM filters to directly relevant facts only

    Step 6: Synthesis
            • LLM synthesizes focused sub-answer

    Step 7: Refinement (if confidence < threshold)
            • Detect vague references
            • Targeted searches for specifics
            • Re-synthesize with new evidence

Phase 3: FINAL SYNTHESIS
└── Combine sub-answers into comprehensive final answer
    • Deduplicate evidence across sub-answers
```

### Configuration Options

All V6 behavior is configurable via `ResearcherConfig`:

```python
from src.querying_system.v6 import ResearcherConfig

config = ResearcherConfig(
    # ─────────────────────────────────────────────────────────────
    # FEATURE TOGGLES
    # ─────────────────────────────────────────────────────────────
    enable_global_search=True,        # Always run global search alongside scoped
    enable_gap_expansion=True,        # LLM-guided 1-hop expansion when gaps detected
    enable_entity_drilldown=True,     # Extra retrieval for ENUMERATION questions
    enable_refinement_loop=True,      # Refine vague answers with targeted searches
    enable_llm_fact_filter=True,      # LLM-based fact filtering for precision
    enable_fact_filter_critique=True, # Critique loop for the fact filter

    # ─────────────────────────────────────────────────────────────
    # RETRIEVAL PARAMETERS
    # ─────────────────────────────────────────────────────────────
    relevance_threshold=0.65,         # OpenAI embedding score cutoff (higher = more precision)
    global_top_k=30,                  # Max facts from global search
    scoped_threshold=0.3,             # Min similarity for scoped entity/topic search
    drilldown_max_entities=10,        # Max entities to drill down on
    max_facts_to_score=50,            # Max facts to filter/score per sub-query

    # ─────────────────────────────────────────────────────────────
    # REFINEMENT PARAMETERS
    # ─────────────────────────────────────────────────────────────
    max_refinement_loops=2,           # Max iterations of refinement
    refinement_search_top_k=20,       # Facts per refinement search
    refinement_confidence_threshold=0.85,  # Skip refinement if confidence >= this

    # ─────────────────────────────────────────────────────────────
    # SYNTHESIS PARAMETERS
    # ─────────────────────────────────────────────────────────────
    top_k_evidence=15,                # Facts to include in synthesis
    top_k_evidence_enumeration=40,    # More facts for ENUMERATION questions
)
```

### Feature Toggle Details

| Feature | Default | Description |
|---------|---------|-------------|
| `enable_global_search` | `True` | Run global vector search alongside entity-scoped searches. Ensures coverage even if resolution fails. |
| `enable_gap_expansion` | `True` | When gaps detected in retrieved facts, expand 1-hop from identified entities. Adds ~20-50% query time when triggered. |
| `enable_entity_drilldown` | `True` | For ENUMERATION questions, LLM selects entities that might have more relevant info and expands from them. |
| `enable_refinement_loop` | `True` | Detects vague references in answers (e.g., "three Districts" without names) and runs targeted searches to resolve them. Only triggers when confidence < 0.85. |
| `enable_llm_fact_filter` | `True` | Uses LLM to filter facts to only those directly relevant, dramatically improving precision. Adds ~500-1500ms per sub-query. |
| `enable_fact_filter_critique` | `True` | Critique loop catches filter errors (missed facts or wrong inclusions). Adds ~300-800ms additional. |

### Configuration Trade-offs

Each configuration parameter involves a trade-off between **precision**, **recall**, **latency**, and **cost**:

| Parameter | Higher Value → | Lower Value → |
|-----------|----------------|---------------|
| `relevance_threshold` | More precision, less recall, faster | More recall, less precision, slower LLM filter |
| `global_top_k` | More coverage, more noise | Faster, may miss relevant facts |
| `scoped_threshold` | More precision in scoped search | More recall in scoped search |
| `top_k_evidence` | More context for synthesis, higher cost | Faster synthesis, may miss nuance |
| `refinement_confidence_threshold` | Refinement triggers more often | Refinement triggers less often |

### Understanding Key Parameters

#### `relevance_threshold` (default: 0.65)

This is the **most important tuning parameter**. It controls which facts pass the initial filter.

```
threshold=0.55  →  More facts pass  →  Higher recall, more noise
threshold=0.65  →  Balanced (default)
threshold=0.75  →  Fewer facts pass  →  Higher precision, may miss relevant facts
```

The threshold works with OpenAI's `text-embedding-3-large` embeddings (3072 dimensions), which provide better score separation than smaller models. A 0.65 threshold filters out ~80% of retrieved facts while keeping most relevant ones.

#### `enable_llm_fact_filter` (default: True)

When enabled, an LLM reviews facts that passed the threshold filter and removes tangentially related ones.

**Example**: For "What were Boston's unemployment trends?", threshold filtering might include:
- ✅ "Boston reported unemployment declined to 3.2%" (relevant)
- ❌ "Boston Fed President met with local businesses" (tangentially related)

The LLM filter catches the second case. **Trade-off**: Adds ~500-1500ms per sub-query but significantly improves precision.

#### `enable_gap_expansion` (default: True)

When retrieved facts seem insufficient, V6 expands 1-hop in the graph from key entities.

**When it triggers**:
1. LLM detects missing information compared to the question
2. Fewer than 5 facts pass threshold filtering

**What it does**:
```
Fact about "Boston" → Expand → Find related facts about "Federal Reserve Bank of Boston"
                            → Find facts about entities mentioned alongside Boston
```

**Trade-off**: Adds ~20-50% latency when triggered, but catches facts missed by direct retrieval.

#### `enable_refinement_loop` (default: True)

Detects vague answers and runs targeted searches to fill specifics.

**What it catches**:
- Quantified but unnamed: "three Districts reported growth" → Which districts?
- Generic references: "several companies", "various factors"
- Count mismatches: Evidence mentions 4 items, answer lists 3

**How it works**:
1. LLM analyzes answer for vague references
2. Generates targeted search queries for each
3. Runs searches in parallel
4. Re-synthesizes with new evidence

**Trade-off**: Only triggers when `confidence < refinement_confidence_threshold` (0.85). Adds ~500-2000ms when triggered.

### Configuration Presets

#### 🚀 Fast Mode (minimize latency)
Best for: Simple factual questions, real-time applications

```python
config = ResearcherConfig(
    enable_gap_expansion=False,        # Skip expansion
    enable_entity_drilldown=False,     # Skip drilldown
    enable_refinement_loop=False,      # Skip refinement
    enable_llm_fact_filter=False,      # Skip LLM filter
    enable_fact_filter_critique=False, # Skip critique
    global_top_k=15,                   # Fewer facts
    top_k_evidence=10,                 # Smaller synthesis context
)
# Expected: ~30-60 seconds per query
```

#### 🎯 High Precision Mode (minimize noise)
Best for: Questions requiring specific facts, avoiding hallucination

```python
config = ResearcherConfig(
    relevance_threshold=0.70,          # Higher threshold
    enable_llm_fact_filter=True,       # LLM filtering
    enable_fact_filter_critique=True,  # Critique catches errors
    max_facts_to_score=30,             # Fewer facts to filter
    top_k_evidence=12,                 # Focused synthesis
)
# Expected: ~90-150 seconds per query
```

#### 📚 High Recall Mode (maximize coverage)
Best for: Enumeration questions, comprehensive research

```python
config = ResearcherConfig(
    relevance_threshold=0.55,          # Lower threshold
    enable_gap_expansion=True,         # Expand when gaps
    enable_entity_drilldown=True,      # Extra retrieval
    global_top_k=50,                   # More global facts
    top_k_evidence=30,                 # More evidence
    top_k_evidence_enumeration=60,     # Even more for enumerations
)
# Expected: ~120-180 seconds per query
```

#### ⚖️ Balanced Mode (default)
Best for: General-purpose querying

```python
config = ResearcherConfig()  # All defaults
# Expected: ~60-120 seconds per query
```

### Question Type Handling

V6 automatically adjusts behavior based on question type:

| Question Type | Detection | Special Handling |
|--------------|-----------|------------------|
| **FACTUAL** | "What was...", "How much..." | Standard retrieval, 15 evidence facts |
| **COMPARISON** | "Compare...", "vs", "difference" | Generates combinatorial sub-queries for each comparison target |
| **ENUMERATION** | "Which...", "List...", "What are all..." | Uses `top_k_evidence_enumeration=40`, triggers entity drilldown |
| **TEMPORAL** | "How has X changed...", dates | Time-focused retrieval, emphasizes `document_date` in synthesis |
| **CAUSAL** | "Why...", "What caused..." | Multi-hop expansion more likely, looks for causal relationships |

### Performance Characteristics

| Configuration | Query Time | Precision | Recall | Cost |
|--------------|------------|-----------|--------|------|
| Fast Mode | ~30-60s | Medium | Medium | Low |
| Balanced (default) | ~60-120s | High | High | Medium |
| High Precision | ~90-150s | Very High | Medium | Medium-High |
| High Recall | ~120-180s | Medium-High | Very High | High |

### Debugging Configuration Issues

**Problem: Too many irrelevant facts in answer**
- Increase `relevance_threshold` (try 0.70)
- Enable `enable_llm_fact_filter=True`
- Decrease `global_top_k` (try 20)

**Problem: Missing relevant information**
- Decrease `relevance_threshold` (try 0.55)
- Enable `enable_gap_expansion=True`
- Increase `global_top_k` (try 50)

**Problem: Vague answers ("several Districts" without names)**
- Enable `enable_refinement_loop=True`
- Lower `refinement_confidence_threshold` (try 0.80)
- Increase `top_k_evidence` (try 25)

**Problem: Queries too slow**
- Disable optional features (gap expansion, drilldown, refinement)
- Reduce `global_top_k` and `top_k_evidence`
- Disable LLM fact filter (use threshold only)

### Usage Examples

**Simple Query:**
```python
import asyncio
from src.querying_system.v6 import query_v6

async def main():
    result = await query_v6("What economic conditions did Boston report?")
    print(result.answer)
    print(f"Confidence: {result.confidence}")
    print(f"Evidence count: {len(result.evidence)}")

asyncio.run(main())
```

**With Custom Configuration:**
```python
from src.querying_system.v6 import V6Pipeline, ResearcherConfig

# High-precision configuration
config = ResearcherConfig(
    relevance_threshold=0.70,         # Higher threshold = fewer, more relevant facts
    enable_llm_fact_filter=True,      # LLM precision filtering
    enable_fact_filter_critique=True, # Critique catches filter errors
)

pipeline = V6Pipeline(group_id="my_tenant", config=config)
result = await pipeline.query("Compare inflation trends across Fed districts")
```

**High-Recall Configuration:**
```python
# When you want more comprehensive results
config = ResearcherConfig(
    relevance_threshold=0.55,          # Lower threshold = more facts
    enable_gap_expansion=True,         # Expand when few results
    enable_entity_drilldown=True,      # Extra retrieval passes
    top_k_evidence=30,                 # More evidence in synthesis
)
```

**Fast Configuration (disable optional steps):**
```python
# Minimize latency for simple questions
config = ResearcherConfig(
    enable_gap_expansion=False,
    enable_entity_drilldown=False,
    enable_refinement_loop=False,
    enable_llm_fact_filter=False,
)
```

### LLM Usage in V6

| Step | Model | Purpose |
|------|-------|---------|
| Decomposition | `gpt-5.1` | Break question into sub-queries |
| Entity Resolution | `gpt-5-mini` | Verify entity candidates |
| Topic Resolution | `gpt-5-mini` | Verify topic candidates |
| Gap Detection | `gpt-5-mini` | Identify missing information |
| Drilldown Selection | `gpt-5-mini` | Select entities to expand |
| LLM Fact Filter | `gpt-5.2` | Precision filtering of facts |
| Fact Filter Critique | `gpt-5.2` | Review filter decisions |
| Sub-Answer Synthesis | `gpt-5.1` | Synthesize sub-answer from facts |
| Vagueness Detection | `gpt-5-mini` | Detect vague references |
| Final Synthesis | `gpt-5.1` | Combine sub-answers into final answer |

---

## Installation

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Neo4j database (local or [Aura](https://neo4j.com/cloud/aura/))
- OpenAI API key (for embeddings and GPT models)
- Google API key (for Gemini models)

### Setup

```bash
# Clone the repository
git clone https://github.com/Zomma-Labs/ZommaLabsKnowledgeGraph.git
cd ZommaLabsKnowledgeGraph

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Initialize Neo4j vector indices
uv run src/scripts/setup_graph_index.py
```

---

## Configuration

### Required Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key (embeddings + GPT models) |
| `GOOGLE_API_KEY` | Google API key (Gemini models) |
| `NEO4J_URI` | Neo4j connection URI (e.g., `bolt://localhost:7687`) |
| `NEO4J_USERNAME` | Neo4j username |
| `NEO4J_PASSWORD` | Neo4j password |

### Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_DATABASE` | `neo4j` | Neo4j database name |
| `VERBOSE` | `false` | Enable detailed logging |
| `EXTRACTION_CONCURRENCY` | `100` | Max parallel LLM extractions |

### Vector Stores

- **Neo4j Vector Indices** (3072 dimensions, OpenAI text-embedding-3-large):
  - `entity_name_embeddings` - Entity name + summary matching
  - `entity_name_only_embeddings` - Name-only matching
  - `fact_embeddings` - Semantic fact search
  - `topic_embeddings` - Topic matching

- **Qdrant** (local):
  - `./qdrant_topics` - Topic ontology vectors
  - `./qdrant_facts` - Fact vector cache

---

## Project Structure

```
src/
├── pipeline.py              # Main ingestion orchestrator
├── agents/                  # LLM-based extraction agents
│   ├── extractor_v2.py      # Chain-of-thought extraction
│   ├── entity_registry.py   # Entity deduplication against graph
│   ├── topic_librarian.py   # Topic resolution against ontology
│   └── temporal_extractor.py # Document date extraction
├── chunker/                 # Document processing
│   ├── pdf_to_markdown.py   # Gemini PDF conversion
│   ├── markdown_chunker.py  # Header-based chunking
│   ├── chunk_types.py       # Chunk data structures
│   └── SAVED/               # Processed chunks (JSONL)
├── schemas/                 # Pydantic models
│   ├── nodes.py             # Neo4j node definitions
│   └── extraction.py        # Extraction schemas
├── util/                    # Shared utilities
│   ├── services.py          # Singleton service container
│   ├── llm_client.py        # LLM provider wrappers
│   ├── neo4j_client.py      # Database operations
│   └── entity_dedup.py      # In-document deduplication
├── config/                  # Configuration
│   └── topics/              # Financial topic ontology
├── scripts/                 # Setup utilities
│   └── setup_graph_index.py # Initialize vector indices
└── querying_system/
    ├── v6/                  # Production query pipeline
    │   ├── pipeline.py      # V6 orchestrator
    │   ├── researcher.py    # Per-subquery research agent
    │   ├── graph_store.py   # Neo4j/Qdrant interface
    │   ├── schemas.py       # V6 data models + config
    │   └── prompts.py       # LLM prompts
    ├── deep_research/       # Multi-step research pipeline
    ├── shared/              # Shared query utilities
    │   ├── decomposer.py    # Query decomposition
    │   └── schemas.py       # Shared schemas
    └── mcp_server.py        # MCP protocol server

scripts/                     # Test and evaluation scripts
tests/                       # Unit tests
docs/                        # Design documents
eval/                        # Evaluation results
```

---

## MCP Server (External Tool Access)

```bash
# Start the Model Context Protocol server
./scripts/start_mcp.sh
```

---

## Development

### Running Tests

```bash
uv run pytest tests/ -v
```

### Project Documentation

- [CLAUDE.md](CLAUDE.md) - Detailed architecture and implementation guide
- [docs/](docs/) - Design documents and improvement notes

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup
- Code style guidelines
- Pull request process

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

Built with:
- [LangChain](https://langchain.com/) - LLM orchestration
- [Neo4j](https://neo4j.com/) - Graph database
- [Qdrant](https://qdrant.tech/) - Vector search
- [OpenAI](https://openai.com/) - Embeddings & GPT models
- [Google Gemini](https://deepmind.google/technologies/gemini/) - PDF conversion & cheap inference
- [Anthropic Claude](https://anthropic.com/) - Entity disambiguation
