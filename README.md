# ZommaLabsKG

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A knowledge graph ingestion pipeline that transforms financial documents (10-Ks, Earnings Calls, Press Releases, Fed Reports) into a queryable knowledge graph stored in Neo4j.

## Features

- **Chain-of-Thought Extraction**: Two-step LLM extraction that first enumerates all entities, then generates relationships
- **Entity Deduplication**: Vector search + LLM verification to resolve references ("Tim Cook" = "Timothy Cook" = "Apple CEO")
- **Subsidiary Awareness**: Keeps related entities separate (AWS ≠ Amazon, YouTube ≠ Google)
- **Topic Resolution**: Matches extracted topics against a curated financial ontology
- **Chunk-Centric Provenance**: All facts link back to source text for traceability
- **Multi-Tenant Support**: `group_id` isolation for multiple data sources
- **V6 Query Pipeline**: Fast semantic search (~1-2 min/query) with 94% accuracy on financial QA

## Architecture

### Pipeline Overview

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Documents  │───►│  Chunking   │───►│ Extraction  │───►│ Resolution  │───► Neo4j
│  (PDF/MD)   │    │  by Header  │    │    (LLM)    │    │   (Dedup)   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### Three-Phase Pipeline (`src/pipeline.py`)

**Phase 1: Parallel Extraction** (LLM-heavy)
- All chunks processed concurrently with semaphore-limited parallelism
- Chain-of-Thought extraction: enumerate entities → generate relationships
- Reflexion loop: critique step catches missed facts

**Phase 2: Resolution** (sequential per chunk)
- **Entity Resolution**: Vector search in Neo4j + LLM verification
- **Topic Resolution**: Qdrant vector search + LLM verification against financial ontology

**Phase 3: Assembly** (database writes)
- Creates DocumentNode, EpisodicNode, EntityNode, FactNode, TopicNode
- Links entities through chunks for full provenance

### Key Agents (`src/agents/`)

| Agent | File | Purpose |
|-------|------|---------|
| **Extractor** | `extractor_v2.py` | Chain-of-thought extraction with critique/refinement |
| **Entity Registry** | `entity_registry.py` | Deduplication via vector search + LLM verification |
| **Topic Librarian** | `topic_librarian.py` | Topic resolution against financial ontology |

### Graph Schema (Chunk-Centric Design)

All relationships flow through chunks for provenance:

```
(EntityNode) ─[:RELATIONSHIP]─► (EpisodicNode) ─[:RELATIONSHIP_TARGET]─► (EntityNode)
   Subject                          Chunk                                   Object
```

**Node Types:**
- **DocumentNode** - Parent container for source file metadata
- **EpisodicNode** - Text chunk with `header_path` context (the provenance hub)
- **EntityNode** - Deduplicated real-world entities (People, Organizations, Places)
- **FactNode** - Atomic facts with embeddings for semantic search
- **TopicNode** - Financial themes/topics from curated ontology

### Data Flow

```
1. PDF/Markdown ──► Chunker splits by headers ──► JSONL files
                                                      │
2. JSONL chunks ──► Extractor V2 (CoT) ──► Entities + Relations
                                                      │
3. Raw entities ──► Entity Registry ──► Deduplicated canonical entities
                                                      │
4. Topics ──► Topic Librarian ──► Matched ontology topics
                                                      │
5. All data ──► Graph Assembler ──► Neo4j nodes + relationships
```

### LLM Usage

| Step | Model | Purpose |
|------|-------|---------|
| Extraction | GPT-4 class | Entity & relationship extraction |
| Critique | GPT-4 class | Quality check, catch missed facts |
| Entity Dedup | GPT-4 class | Verify entity matches |
| Graph Match | GPT-4-mini | Fast entity lookup verification |
| Summary Merge | Gemini Flash | Merge entity descriptions |
| Topic Resolution | Gemini Flash | Topic matching & verification |

## Installation

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Neo4j database (local or [Aura](https://neo4j.com/cloud/aura/))
- OpenAI API key (for embeddings)
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

## Usage

### Running the Ingestion Pipeline

```bash
# Basic run (processes all chunks in src/chunker/SAVED/)
uv run src/pipeline.py

# Limit chunks for testing
uv run src/pipeline.py --limit 5

# Adjust parallelism
uv run src/pipeline.py --concurrency 10

# Filter by filename
uv run src/pipeline.py --filter beige

# Multi-tenant isolation
uv run src/pipeline.py --group-id tenant1

# Verbose logging
VERBOSE=true uv run src/pipeline.py
```

### Querying the Knowledge Graph

```python
import asyncio
from src.querying_system.v6 import query_v6, V6Pipeline, ResearcherConfig

# Simple query
async def main():
    result = await query_v6("What economic conditions did Boston report?")
    print(result.answer)

asyncio.run(main())
```

**With custom configuration:**

```python
config = ResearcherConfig(
    relevance_threshold=0.65,      # Similarity cutoff for fact retrieval
    enable_gap_expansion=True,     # 1-hop graph expansion if few results
    enable_entity_drilldown=True,  # Extra retrieval for enumeration queries
    enable_refinement_loop=True,   # Refine vague initial answers
)

pipeline = V6Pipeline(group_id="my_tenant", config=config)
result = await pipeline.query("Compare inflation trends across Fed districts")
```

### MCP Server (External Tool Access)

```bash
# Start the Model Context Protocol server
./scripts/start_mcp.sh
```

## Project Structure

```
src/
├── pipeline.py              # Main ingestion orchestrator
├── agents/                  # LLM-based extraction agents
│   ├── extractor_v2.py      # Chain-of-thought extraction
│   ├── entity_registry.py   # Entity deduplication
│   └── topic_librarian.py   # Topic resolution
├── chunker/                 # Document processing
│   ├── markdown_chunker.py  # Header-based chunking
│   ├── pdf_to_markdown.py   # PDF conversion
│   └── SAVED/               # Processed chunks (JSONL)
├── schemas/                 # Pydantic models
│   ├── nodes.py             # Neo4j node definitions
│   └── extraction.py        # Extraction schemas
├── util/                    # Shared utilities
│   ├── services.py          # Singleton service container
│   ├── llm_client.py        # LLM provider wrappers
│   ├── neo4j_client.py      # Database operations
│   └── entity_dedup.py      # Deduplication logic
├── config/                  # Configuration
│   └── topics/              # Financial topic ontology
├── scripts/                 # Setup utilities
│   └── setup_graph_index.py # Initialize vector indices
└── querying_system/
    ├── v6/                  # Production query pipeline
    ├── deep_research/       # Multi-step research pipeline
    ├── shared/              # Shared query utilities
    └── mcp_server.py        # MCP protocol server

scripts/                     # Test and evaluation scripts
tests/                       # Unit tests
docs/                        # Design documents
eval/                        # Evaluation results
```

## Configuration

### Required Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for embeddings (text-embedding-3-large) |
| `GOOGLE_API_KEY` | Google API key for Gemini models |
| `NEO4J_URI` | Neo4j connection URI (e.g., `bolt://localhost:7687`) |
| `NEO4J_USERNAME` | Neo4j username |
| `NEO4J_PASSWORD` | Neo4j password |

### Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_DATABASE` | `neo4j` | Neo4j database name |
| `VERBOSE` | `false` | Enable detailed logging |
| `EXTRACTION_CONCURRENCY` | `5` | Max parallel LLM extractions |

See [.env.example](.env.example) for the complete template.

### Vector Stores

- **Neo4j Vector Indices** (3072 dimensions):
  - `entity_name_embeddings` - Full entity matching
  - `entity_name_only_embeddings` - Name-only matching
  - `fact_embeddings` - Semantic fact search

- **Qdrant** (local):
  - `./qdrant_topics` - Topic ontology vectors
  - `./qdrant_facts` - Fact vector cache

## Development

### Running Tests

```bash
uv run pytest tests/ -v
```

### Project Documentation

- [CLAUDE.md](CLAUDE.md) - Detailed architecture and implementation guide
- [docs/](docs/) - Design documents and improvement notes

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup
- Code style guidelines
- Pull request process

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Built with:
- [LangChain](https://langchain.com/) & [LangGraph](https://langchain-ai.github.io/langgraph/) - LLM orchestration
- [Neo4j](https://neo4j.com/) - Graph database
- [Qdrant](https://qdrant.tech/) - Vector search
- [OpenAI](https://openai.com/) - Embeddings
- [Google Gemini](https://deepmind.google/technologies/gemini/) - LLM inference
