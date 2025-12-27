# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ZommaLabsKG is a knowledge graph ingestion pipeline that transforms financial documents (10-Ks, Earnings Calls, Press Releases) into a typed, FIBO-standardized knowledge graph stored in Neo4j.

## Commands

```bash
# Setup vector indices (required before first run)
uv run src/scripts/setup_graph_index.py

# Run the main ingestion pipeline
uv run scripts/run_pipeline.py

# Test scripts
uv run src/scripts/test_large_pipeline.py      # End-to-end verification
uv run src/scripts/test_atomizer_reflexion.py  # Test atomizer with reflexion
uv run src/scripts/test_entity_resolution.py   # Test entity resolution

# QA Evaluation (tests KG agent against Q&A pairs with LLM judge)
uv run src/scripts/evaluate_qa.py                        # Run full evaluation
uv run src/scripts/evaluate_qa.py --limit 10 --verbose   # Test with 10 questions
uv run src/scripts/evaluate_qa.py --qa-file custom.json  # Use custom Q&A file
```

## Architecture

### Pipeline Flow (LangGraph StateGraph)

The pipeline in `src/workflows/main_pipeline.py` processes chunks through these stages:

1. **initialize_episode** - Creates DocumentNode and EpisodicNode in Neo4j, analyzes header dimensions
2. **atomizer** - Decomposes text into atomic propositions using LLM with reflexion loop
3. **entity_extraction** - Extracts entities/relationships from propositions in parallel
4. **parallel_resolution** - Resolves entities against FIBO ontology and deduplicates against existing graph
5. **assembler** - Creates FactNodes and relationships in Neo4j

### Key Agents (`src/agents/`)

- **atomizer.py** - Text decomposition with de-contextualization (pronoun resolution, temporal grounding, completeness preservation)
- **FIBO_librarian.py** - Hybrid entity resolution using Qdrant vector search + RapidFuzz fuzzy matching
- **entity_extractor.py** - Extracts subject/object/relationship triples from propositions with multi-relation pattern support (source attribution, list expansion)
- **analyst.py** - Classifies relationships into strict semantic types (e.g., ACQUIRED, SUED, RAISED_POLICY_RATE)
- **graph_assembler.py** - Batch writes to Neo4j with embedding-based deduplication
- **graph_enhancer.py** - Entity summarization and graph candidate matching

### Extraction Patterns

**Atomizer Completeness Rule**: Facts must be self-contained. When someone communicates information, the speaker and their message stay together as ONE fact:
- BAD: "The CEO announced something" (incomplete)
- GOOD: "The CEO announced that revenue grew 15% in Q3"

**Entity Extractor Multi-Relation Patterns**: A single fact can produce MULTIPLE relations:
- **List Expansion**: "Subsidiaries include Google, Waymo, DeepMind" → 3 relations
- **Source Attribution**: "Dr. Smith found that Drug X treats Disease Y" → 2 relations:
  - Content: Drug X → treats → Disease Y
  - Source: Dr. Smith → discovered → Drug X

### Node Schema (`src/schemas/nodes.py`)

- **DocumentNode** - Parent container for all chunks from a source file
- **EpisodicNode** - The "hub" chunk of text with `header_path` for context
- **EntityNode** - Deduplicated real-world entities (People, Orgs, Places)
- **FactNode** - Atomic facts with `fact_type` classification and embedding
- **TopicNode** - Global themes linked to chunks

### Shared Services (`src/util/services.py`)

Singleton container providing lazy-initialized clients:
- `services.llm` - LLM client (Gemini or OpenAI based on LLM_MODEL env var)
- `services.embeddings` - Voyage AI embeddings (voyage-finance-2)
- `services.neo4j` - Neo4j client
- `services.qdrant_fibo` - Qdrant client for FIBO entity vectors
- `services.qdrant_relationships` - Qdrant client for relationship definitions

### Environment Variables

- `LLM_MODEL` - Model to use (default: gemini-2.5-flash-lite)
- `GOOGLE_API_KEY` - Required for Gemini models
- `VOYAGE_API_KEY` - Required for embeddings
- `VERBOSE` - Set to "true" for detailed logging
- `LLM_CONCURRENCY` - Max concurrent LLM calls (default: 100)

### Graph Schema (V2 Chunk-Centric Design)

Relationships flow through chunks for provenance:
```
(EntityNode) -[:RELATIONSHIP_TYPE]-> (EpisodicNode) -[:RELATIONSHIP_TYPE_PASSIVE]-> (EntityNode)
     Subject                              Chunk                                          Object
```

Example: `(Apple) -[:SUED]-> (chunk_123) -[:GOT_SUED]-> (Intel)`

### Relationship Taxonomy

Strict enum to prevent schema drift:
- **Corporate**: ACQUIRED, SUED, PARTNERED, INVESTED, DIVESTED, HIRED, LAUNCHED_PRODUCT
- **Financial**: REPORTED_FINANCIALS, ISSUED_GUIDANCE, DECLARED_DIVIDEND, FILED_BANKRUPTCY
- **Regulatory**: REGULATED, SETTLED_LEGAL_DISPUTE, GRANTED_PATENT
- **Causation**: CAUSED, EFFECTED_BY, CONTRIBUTED_TO, PREVENTED

## Data Flow

1. Documents chunked by `src/chunker/` and saved as JSONL in `src/chunker/SAVED/`
2. Each chunk becomes an EpisodicNode linked to its DocumentNode
3. Atomic facts extracted and entities resolved against FIBO (90% confidence threshold)
4. Unmatched entities create new EntityNodes with embeddings
5. FactNodes link subjects/objects through the EpisodicNode for provenance
