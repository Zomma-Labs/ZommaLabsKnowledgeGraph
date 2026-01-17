# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ZommaLabsKG is a knowledge graph ingestion pipeline that transforms financial documents (10-Ks, Earnings Calls, Press Releases) into a typed knowledge graph stored in Neo4j.

## Commands

```bash
# Setup vector indices (required before first run)
uv run src/scripts/setup_graph_index.py

# Run the main ingestion pipeline
uv run src/pipeline.py

# With options
uv run src/pipeline.py --limit 5              # Limit chunks for testing
uv run src/pipeline.py --concurrency 10       # Adjust parallel extraction
uv run src/pipeline.py --filter beige         # Filter files by name
uv run src/pipeline.py --group-id tenant1     # Set tenant/group ID
VERBOSE=true uv run src/pipeline.py           # Verbose mode
```

## Architecture

### Pipeline Flow (`src/pipeline.py`)

Three-phase async pipeline:

**Phase 1: Parallel Extraction** (LLM-heavy)
- All chunks extracted concurrently with semaphore-limited concurrency
- Uses V2 Chain-of-Thought extraction for better entity coverage

**Phase 2: Resolution** (sequential per chunk)
- Entity resolution via EntityRegistry (vector search + LLM verification)
- Topic resolution via TopicLibrarian (Qdrant + LLM verification)

**Phase 3: Assembly** (sequential per chunk)
- Creates DocumentNode, EpisodicNode, EntityNode, FactNode, TopicNode in Neo4j
- Links entities through chunks for provenance

### Key Agents (`src/agents/`)

- **extractor_v2.py** - Chain-of-thought extraction: (1) enumerate ALL entities, (2) generate relationships. Better coverage than single-pass.
- **entity_registry.py** - LLM-based entity deduplication with vector search + LLM verification. Subsidiaries kept separate (AWS ≠ Amazon).
- **topic_librarian.py** - Topic ontology resolution using Qdrant vector search + LLM verification

### Extraction Patterns

**Chain-of-Thought Extraction**: Two-step structured output:
1. Step 1: Enumerate ALL entities in the text
2. Step 2: Generate relationships between enumerated entities

**Reflexion Loop**: Single critique step to catch missed facts and improve extraction quality.

**Free-form Relationships**: No strict enum - relationships preserved as natural language (e.g., "acquired majority stake in").

### Node Schema (`src/schemas/nodes.py`)

- **DocumentNode** - Parent container for all chunks from a source file
- **EpisodicNode** - The "hub" chunk of text with `header_path` for context
- **EntityNode** - Deduplicated real-world entities (People, Orgs, Places)
- **FactNode** - Atomic facts with embedding for semantic search
- **TopicNode** - Global themes linked to chunks

### Shared Services (`src/util/services.py`)

Singleton container providing lazy-initialized clients:
- `services.llm` - LLM client (Gemini by default)
- `services.embeddings` - OpenAI embeddings (text-embedding-3-large, 3072 dimensions)
- `services.neo4j` - Neo4j client

### Environment Variables

- `GOOGLE_API_KEY` - Required for Gemini models
- `OPENAI_API_KEY` - Required for embeddings (text-embedding-3-large)
- `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` - Neo4j connection
- `VERBOSE` - Set to "true" for detailed logging
- `EXTRACTION_CONCURRENCY` - Max concurrent extractions (default: 5)

### Graph Schema (Chunk-Centric Design)

Relationships flow through chunks for provenance:
```
(EntityNode) -[:RELATIONSHIP_TYPE]-> (EpisodicNode) -[:RELATIONSHIP_TYPE_TARGET]-> (EntityNode)
     Subject                              Chunk                                         Object
```

All nodes include `group_id` for multi-tenant isolation.

## Data Flow

1. Documents chunked by `src/chunker/` and saved as JSONL in `src/chunker/SAVED/`
2. Each chunk becomes an EpisodicNode linked to its DocumentNode
3. V2 CoT extraction: enumerate entities → generate relationships → critique → refine
4. Entity resolution via vector search + LLM verification (subsidiaries kept separate)
5. FactNodes link subjects/objects through the EpisodicNode for provenance

## LLM Usage by Pipeline Step

```
PIPELINE FLOW WITH LLM ASSIGNMENTS:

┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: EXTRACTION (per chunk)                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Chunk Text ──► EXTRACTION ──► CRITIQUE ──► RE-EXTRACT (if needed)        │
│                     │              │              │                         │
│                  gpt-5.2        gpt-5.1        gpt-5.2                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2a-2c: IN-DOCUMENT ENTITY DEDUP (bulk)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   All Entities ──► Embed ──► Cluster ──► LLM DEDUP (per component)          │
│                                              │                              │
│                                           gpt-5.2                           │
│                                                                             │
│   Example: "Tim Cook", "Timothy Cook" ──► Same person ──► 1 canonical       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2d: GRAPH RESOLUTION (per canonical entity)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Canonical Entity ──► Vector Search Neo4j ──► LLM MATCH ──► SUMMARY MERGE  │
│                                                    │              │         │
│                                                gpt-5-mini   gemini-flash    │
│                                                                             │
│   Example: "Tim Cook" ──► exists in graph? ──► merge summaries              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: TOPIC RESOLUTION (per chunk)                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Topics ──► BATCH DEFINE ──► Vector Search Qdrant ──► LLM VERIFY           │
│                   │                                        │                │
│            gemini-flash                              gemini-flash           │
│                                                                             │
│   Example: "M&A" ──► define ──► match "Mergers and Acquisitions"            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: ASSEMBLY (bulk write to Neo4j)                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   No LLM calls - pure database operations                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### LLM Summary Table

| Step | Model | Cost | File |
|------|-------|------|------|
| Extraction | `gpt-5.2` | $$$ | `extractor_v2.py` |
| Critique | `gpt-5.1` | $$ | `extractor_v2.py` |
| Re-extract | `gpt-5.2` | $$$ | `extractor_v2.py` |
| In-doc dedup | `gpt-5.2` | $$$ | `entity_dedup.py` |
| Graph match | `gpt-5-mini` | $ | `entity_registry.py` |
| Summary merge | `gemini-2.5-flash-lite` | $ | `entity_registry.py` |
| Topic define | `gemini-2.5-flash-lite` | $ | `topic_librarian.py` |
| Topic verify | `gemini-2.5-flash-lite` | $ | `topic_librarian.py` |

### Helper Functions (`src/util/llm_client.py`)

- `get_llm()` - Main LLM (gpt-5.2)
- `get_critique_llm()` - Critique LLM (gpt-5.1)
- `get_nano_gpt_llm()` - Cheap GPT (gpt-5-mini)
- `get_nano_llm()` - Cheap Gemini (gemini-2.5-flash-lite)

## Vector Stores

- **Neo4j** - `entity_name_embeddings`, `entity_name_only_embeddings`, `topic_embeddings`, `fact_embeddings` indexes (3072 dimensions)
- **Qdrant** - `./qdrant_topics` for topic ontology, `./qdrant_facts` for fact search

## Querying System

Located in `src/querying_system/`:

### V6 Pipeline (Production)
- Uses OpenAI text-embedding-3-large (3072 dims)
- Threshold-only retrieval (no LLM scoring)
- Query time: ~1-2 minutes per question
- Located in `src/querying_system/v6/`

```python
# V6 Usage
from src.querying_system.v6 import V6Pipeline, query_v6

# Option 1: Async function
result = await query_v6("What economic conditions did Boston report?")

# Option 2: Pipeline class
pipeline = V6Pipeline(group_id="default")
result = await pipeline.query("Compare inflation in Boston vs New York")
```

### V6 Key Configuration
```python
from src.querying_system.v6 import ResearcherConfig

config = ResearcherConfig(
    relevance_threshold=0.65,      # OpenAI score cutoff (no LLM scoring)
    enable_gap_expansion=True,     # 1-hop expansion if too few facts
    enable_entity_drilldown=True,  # Extra retrieval for ENUMERATION
    enable_refinement_loop=True,   # Refine vague answers
)
pipeline = V6Pipeline(config=config)
```

### Deep Research Pipeline
- Multi-step research with supervisor/worker pattern
- Located in `src/querying_system/deep_research/`

### MCP Server
- Model Context Protocol server for external tool access
- Located in `src/querying_system/mcp_server.py`
- Start with: `scripts/start_mcp.sh`

## Directory Structure

```
src/
├── pipeline.py              # Main ingestion pipeline
├── agents/                  # Extraction agents
├── chunker/                 # Document chunking
├── config/                  # Configuration files
├── schemas/                 # Data schemas
├── util/                    # Shared utilities
├── scripts/                 # Setup and maintenance scripts
└── querying_system/
    ├── v6/                  # Production query pipeline
    ├── deep_research/       # Deep research pipeline
    ├── shared/              # Shared query utilities
    ├── mcp_server.py        # MCP server
    └── _archived/           # Old versions (v1-v5, experiments)

scripts/                     # Active test/eval scripts
├── test_v6_*.py            # V6 pipeline tests
├── test_deep_research_*.py # Deep research tests
├── start_mcp.sh            # MCP server launcher
└── _archived/              # Old scripts

docs/                        # Design documents and planning
eval/                        # Evaluation results
tests/                       # Unit tests
```

## Archived Code

Old pipeline versions and experimental code are in `_archived/` directories:
- `src/querying_system/_archived/` - v1-v5, hybrid_cot_gnn, structured_pipeline
- `scripts/_archived/` - old debug/diagnostic scripts
