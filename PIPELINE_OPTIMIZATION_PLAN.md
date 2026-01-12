# Pipeline Optimization Plan

## Current State Analysis

### Architecture Overview

The current pipeline has two files with overlapping responsibilities:

| File | Role | Issues |
|------|------|--------|
| `src/run_pipeline.py` | Multi-file orchestrator | Reimplements resolution logic, Phase 2 is sequential |
| `src/workflows/pipeline.py` | LangGraph state machine | Creates new agents per chunk, not used by run_pipeline |

### Performance Bottlenecks Identified

#### Bottleneck 1: Sequential Entity Resolution (CRITICAL)
**Location**: `run_pipeline.py:192-205`
```python
for entity in extraction.entities:  # Sequential loop
    resolution = entity_registry.resolve(...)  # Each call = embed + vector_search + LLM
```
- **Impact**: N entities × 3 API calls = 3N sequential calls per chunk
- **Current**: ~500ms-2s per entity

#### Bottleneck 2: Sequential Topic Resolution
**Location**: `run_pipeline.py:220-228`
```python
for topic in unique_topics:  # Sequential loop
    match = topic_librarian.resolve_with_definition(...)
```
- **Impact**: M topics × 2 API calls per chunk
- **Current**: ~300ms-1s per topic

#### Bottleneck 3: Individual Embedding Calls
**Location**: `run_pipeline.py:240-241, 269`
```python
embedding = embeddings.embed_query(embed_text)  # One at a time
fact_embedding = embeddings.embed_query(fact.fact)  # One at a time
```
- **Impact**: Voyage AI supports batches of 128, we're sending 1 at a time
- **Current**: ~100ms per embed call, could batch 50+ in same time

#### Bottleneck 4: Individual Neo4j Writes
**Location**: `pipeline.py` helper functions
```python
_create_entity_node(...)   # Individual Cypher query
_create_fact_node(...)     # Individual Cypher query
_create_relationship(...)  # Individual Cypher query
```
- **Impact**: Network round-trip (~5-20ms) per write
- **Current**: 50+ queries per chunk

#### Bottleneck 5: Sequential Phase 2 Processing
**Location**: `run_pipeline.py:458-471`
```python
for i, extraction_result in enumerate(extraction_results):
    result = resolve_and_assemble(...)  # Chunks processed one at a time
```
- **Impact**: No parallelism across chunks in resolution phase
- **Current**: Total time = sum of all chunk times

---

## Proposed Optimized Architecture

### Design Principles

1. **Accuracy First**: Entity deduplication must remain LLM-verified
2. **Maximum Parallelism**: Concurrent API calls with rate limiting
3. **Batched Operations**: Embeddings, searches, and writes in batches
4. **Single Source of Truth**: One consolidated pipeline file
5. **Cross-Chunk Deduplication**: Dedupe entities across all chunks before Neo4j writes

### New Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 1: Parallel Extraction (KEEP AS-IS)                          │
│   - All chunks extracted concurrently                               │
│   - Semaphore-limited (default: 5 concurrent)                       │
│   - Output: List[ChainOfThoughtResult]                              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 2: Batched Resolution (NEW)                                   │
│                                                                      │
│   2a. Collect all entities from all chunks                          │
│       └── Deduplicate by name (in-memory)                           │
│                                                                      │
│   2b. Batch embed all unique entities (1 API call per 128)          │
│       └── embeddings.embed_documents([...])                         │
│                                                                      │
│   2c. Parallel vector searches (asyncio.gather)                     │
│       └── All searches run concurrently                             │
│                                                                      │
│   2d. Parallel LLM entity verification (semaphore-limited)          │
│       └── Rate-limited concurrent LLM calls                         │
│                                                                      │
│   2e. Batch embed + resolve all topics                              │
│       └── Same pattern as entities                                  │
│                                                                      │
│   Output: entity_lookup, topic_lookup (shared across chunks)        │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 3: Batched Assembly (NEW)                                     │
│                                                                      │
│   3a. Batch embed all facts (1 API call per 128)                    │
│                                                                      │
│   3b. Batch Neo4j writes using UNWIND                               │
│       └── CREATE DocumentNode (1 query)                             │
│       └── CREATE all EpisodicNodes (1 query with UNWIND)            │
│       └── CREATE all EntityNodes (1 query with UNWIND)              │
│       └── CREATE all FactNodes (1 query with UNWIND)                │
│       └── CREATE all relationships (1 query with UNWIND)            │
│                                                                      │
│   Output: Graph populated                                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Tasks

### Task 1: Consolidate Pipeline Files
- [ ] Merge `run_pipeline.py` and `pipeline.py` into single `src/pipeline.py`
- [ ] Remove LangGraph overhead (not needed for this flow)
- [ ] Keep helper functions for Neo4j operations

### Task 2: Add Batch Embedding Support
- [ ] Create `batch_embed(texts: List[str]) -> List[List[float]]` utility
- [ ] Use `embeddings.embed_documents()` instead of `embed_query()`
- [ ] Batch size: 128 (Voyage AI limit)

### Task 3: Parallel Entity Resolution
- [ ] Collect all entities from all extraction results
- [ ] In-memory deduplication by normalized name
- [ ] Batch embed unique entities
- [ ] Parallel vector searches using `asyncio.gather()`
- [ ] Parallel LLM verification with semaphore (limit: 20 concurrent)
- [ ] Build shared `entity_lookup` dict

### Task 4: Parallel Topic Resolution
- [ ] Collect all topics from all facts
- [ ] Batch `batch_define_topics()` across all chunks (already exists)
- [ ] Parallel `resolve_with_definition()` calls
- [ ] Build shared `topic_lookup` dict

### Task 5: Batched Neo4j Writes
- [ ] Create `_batch_create_entities(entities: List[dict])` using UNWIND
- [ ] Create `_batch_create_facts(facts: List[dict])` using UNWIND
- [ ] Create `_batch_create_relationships(rels: List[dict])` using UNWIND
- [ ] Single transaction per batch

### Task 6: Cross-Chunk Entity Deduplication
- [ ] Before LLM verification, group similar entities across chunks
- [ ] Use embedding similarity to cluster potential duplicates
- [ ] Single LLM call per cluster (not per entity)

---

## Estimated Performance Impact

| Operation | Current | Optimized | Speedup |
|-----------|---------|-----------|---------|
| Entity embedding (10 entities) | 1000ms (10×100ms) | 100ms (1 batch) | **10x** |
| Entity LLM verification | Sequential | 5 concurrent | **5x** |
| Topic resolution | Sequential | 10 concurrent | **10x** |
| Neo4j writes (50 ops) | 500ms (50×10ms) | 30ms (1 batch) | **15x** |
| Overall per chunk | 15-30s | 2-5s | **5-10x** |

### Projected Times (10 chunks, 50 entities total, 30 topics)

| Phase | Current | Optimized |
|-------|---------|-----------|
| Phase 1 (extraction) | ~30s | ~30s (unchanged) |
| Phase 2 (resolution) | ~150s | ~15s |
| Phase 3 (assembly) | ~30s | ~5s |
| **Total** | **~210s** | **~50s** |

---

## Risk Mitigation

### Accuracy Concerns
- **Entity Dedup**: Keep LLM verification, just run in parallel
- **Topic Matching**: Keep LLM verification, just run in parallel
- **Testing**: Run both pipelines on same data, compare outputs

### Rate Limiting
- **Gemini**: 100 concurrent requests → use semaphore(20) for safety
- **Voyage AI**: 300 RPM → batch calls reduce total requests
- **Neo4j**: No rate limit, but use transactions for consistency

### Error Handling
- **Partial failures**: Track which entities/topics failed, retry individually
- **Transaction rollback**: Use Neo4j transactions, rollback on error

---

## Files to Modify

| File | Action |
|------|--------|
| `src/run_pipeline.py` | **DELETE** - merge into pipeline.py |
| `src/workflows/pipeline.py` | **REWRITE** - new optimized implementation |
| `src/agents/entity_registry.py` | **ADD** - `resolve_batch_parallel()` method |
| `src/agents/topic_librarian.py` | **ADD** - `resolve_batch_parallel()` method |
| `src/util/services.py` | **ADD** - batch embedding utility |

---

## Success Criteria

1. **Performance**: 5x+ speedup on 10+ chunk documents
2. **Accuracy**: Same entities/facts/relationships as current pipeline
3. **Reliability**: Graceful handling of API failures
4. **Simplicity**: Single entry point, clear flow

---

## Open Questions

1. Should we keep LangGraph or use plain async Python?
   - Recommendation: Plain async (simpler, less overhead)

2. How to handle entity conflicts across chunks?
   - Recommendation: First-seen wins, merge summaries

3. Should we add caching for embeddings?
   - Recommendation: Yes, Redis or in-memory LRU cache

4. Should Phase 2 resolution happen per-file or across all files?
   - Recommendation: Per-file to limit memory, but share entity registry
