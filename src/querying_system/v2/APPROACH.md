# V2 Querying System: GNN-Inspired Knowledge Graph Query Pipeline

## Overview

The V2 querying system is a multi-phase pipeline that combines graph-based retrieval with LLM scoring and synthesis. It draws inspiration from Graph Neural Networks (GNNs) by treating retrieval as "message passing" from resolved nodes.

**Key Design Principles:**
1. **Scoped + Global Retrieval**: High precision (scoped around resolved nodes) + high coverage (global search)
2. **Parallel Execution**: Maximize throughput by parallelizing independent operations
3. **Question-Type Awareness**: Special handling for ENUMERATION, CAUSAL, TEMPORAL questions
4. **Entity Drill-Down**: For enumeration questions, agent can request additional facts for specific entities

## Architecture

```
                                    Question
                                       │
                                       ▼
                          ┌────────────────────────┐
                          │  Phase 1: Decompose    │
                          │      (gpt-5.1)         │
                          └────────────────────────┘
                                       │
                          entity_hints, topic_hints, question_type
                                       │
                                       ▼
                          ┌────────────────────────┐
                          │  Phase 2: Resolve      │
                          │    (gpt-5-mini)        │
                          └────────────────────────┘
                                       │
                          resolved entity/topic nodes
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                     │
                    ▼                                     ▼
       ┌─────────────────────┐               ┌─────────────────────┐
       │ Phase 3: Scoped     │               │ Phase 6: Global     │
       │ Search (parallel)   │               │ Search (parallel)   │
       │                     │               │                     │
       │ Per-entity/topic    │               │ Vector + Keyword    │
       │ fact retrieval      │               │ on all facts        │
       └─────────────────────┘               └─────────────────────┘
                    │                                     │
                    │              PARALLEL               │
                    ▼                                     ▼
       ┌─────────────────────┐               ┌─────────────────────┐
       │ Phase 4: Scoped     │               │ Phase 7: Global     │
       │ Scoring (gpt-5-mini)│               │ Scoring (gpt-5-mini)│
       └─────────────────────┘               └─────────────────────┘
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       │
                                       ▼
                          ┌────────────────────────┐
                          │  Phase 5: Expansion    │
                          │  (graph traversal)     │
                          └────────────────────────┘
                                       │
                                       ▼
                          ┌────────────────────────┐
                          │  Phase 6: Drill-Down   │  ◄── ENUMERATION only
                          │     (gpt-5.1)          │
                          └────────────────────────┘
                                       │
                                       ▼
                          ┌────────────────────────┐
                          │  Phase 8: Combine      │
                          │  Dedupe + Boost        │
                          └────────────────────────┘
                                       │
                                       ▼
                          ┌────────────────────────┐
                          │  Phase 9: Synthesize   │
                          │     (gpt-5.1)          │
                          └────────────────────────┘
                                       │
                                       ▼
                                    Answer
```

## Phase Details

### Phase 1: Decomposition
**LLM**: gpt-5.1
**Purpose**: Extract structured information from the question

**Outputs**:
- `entity_hints`: Entities mentioned (e.g., ["Apple", "Microsoft"])
- `topic_hints`: Topics/themes (e.g., ["revenue", "market share"])
- `question_type`: FACTUAL | COMPARISON | CAUSAL | TEMPORAL | ENUMERATION
- `required_info`: What the answer must contain
- `sub_queries`: Decomposed sub-questions

### Phase 2: Resolution
**LLM**: gpt-5-mini
**Purpose**: Map extracted hints to actual graph nodes

**Process**:
1. Embed each hint using Voyage AI
2. Vector search in Neo4j for candidate nodes
3. LLM verification to select best matches

**Example**:
```
"economic growth" → ["GDP", "Economic Activity"]
"districts" → ["First District", "Second District", ...]
```

### Phase 3 & 6: Parallel Retrieval

Both searches run concurrently via `asyncio.gather()`.

#### Scoped Search (Phase 3)
- For each resolved node, search facts connected to that node
- Uses Qdrant fact vector store with question embedding
- Extracts **unique connected entities** (first 30) for enumeration support

#### Global Search (Phase 6)
- Vector search on all facts using question embedding
- Keyword search using extracted terms
- Merges results with cross-query boosting (+0.2 for facts found by both)

### Phase 4 & 7: Parallel Scoring
**LLM**: gpt-5-mini
**Purpose**: Rank facts by relevance, mark facts for expansion

Both scoring operations run in parallel.

**Scorer sees**:
```
[0] Federal Reserve Bank of Boston -[REPORTED]-> Economic Activity
    FACT: Economic activity expanded slightly with modest growth
    DATE: 2026-01-09

UNIQUE ENTITIES FOUND (for enumeration):
  Connected to 'Economic Activity': Boston, Philadelphia, Richmond, ...
```

**Outputs per fact**:
- `relevance`: 0.0 - 1.0 score
- `should_expand`: Boolean for CAUSAL/COMPARISON questions

### Phase 5: Expansion
**Purpose**: Follow graph edges from high-relevance facts

For facts marked `should_expand=true`:
1. Extract subject/object entities
2. Query 1-hop neighbors in Neo4j
3. Add new facts to evidence pool

### Phase 6: Entity Drill-Down (ENUMERATION only)
**LLM**: gpt-5.1
**Purpose**: Ensure complete coverage for enumeration questions

**Process**:
1. Show agent the list of unique entities found across all scoped searches
2. Agent selects entities that need more facts (no limit on count)
3. Fetch additional facts for selected entities
4. Add to evidence pool

**Example Agent Reasoning**:
> "The question asks which districts reported growth. I see Boston, Richmond, Philadelphia in the unique entities. I should pull more facts for these district-related entities to ensure complete enumeration."

### Phase 8: Combine
**Purpose**: Merge all fact sources with deduplication and boosting

**Priority order**:
1. Scoped facts (highest priority)
2. Expanded facts
3. Drill-down facts
4. Global facts (fills gaps)

**Cross-query boost**: +0.15 for facts found by multiple searches

### Phase 9: Synthesis
**LLM**: gpt-5.1
**Purpose**: Generate final answer from evidence

**Special handling for ENUMERATION**:
- Uses up to 40 facts (vs 20 for other types)
- Includes unique entities list in prompt
- Format instructions emphasize exhaustive listing

## Parallelization Strategy

### Within Pipeline (per question)
```python
# Retrieval phase
scoped_results, global_facts = await asyncio.gather(
    retrieve_scoped(...),
    retrieve_global(...)
)

# Scoring phase
(scored_scoped, _), (scored_global, _) = await asyncio.gather(
    score_scoped(),
    score_global()
)
```

### Within Scoped Search
```python
# All entity/topic searches run in parallel
tasks = [search_node(entity) for entity in resolved_entities]
tasks += [search_node(topic) for topic in resolved_topics]
all_facts = await asyncio.gather(*tasks)
```

### Across Questions
```python
# Multiple questions with semaphore control
semaphore = asyncio.Semaphore(concurrency=3)
results = await asyncio.gather(*[
    process_question(q, semaphore) for q in questions
])
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_k_per_node` | 20 | Facts retrieved per resolved node |
| `top_k_global` | 20 | Global search result limit |
| `top_n_unique_entities` | 30 | Unique entities extracted per node |
| `max_facts_to_score` | 50 | LLM scoring batch size |
| `top_k_evidence` | 20 | Facts sent to synthesis (40 for ENUMERATION) |
| `max_entities_to_expand` | 10 | Expansion entity limit |

## LLM Usage Summary

| Phase | Model | Purpose | Cost |
|-------|-------|---------|------|
| Decomposition | gpt-5.1 | Quality decomposition | $$ |
| Resolution | gpt-5-mini | Candidate verification | $ |
| Scoped Scoring | gpt-5-mini | Batch relevance scoring | $ |
| Global Scoring | gpt-5-mini | Batch relevance scoring | $ |
| Entity Drill-Down | gpt-5.1 | Entity selection (ENUM only) | $$ |
| Synthesis | gpt-5.1 | Answer generation | $$ |

**Total LLM calls**: 5-6 per question (6 for ENUMERATION)

## Question Type Handling

### FACTUAL
- Standard scoped + global retrieval
- No expansion
- 20 facts to synthesis

### COMPARISON
- Expansion enabled for compared entities
- Side-by-side formatting in synthesis

### CAUSAL
- Expansion enabled for cause/effect chains
- Causal chain formatting in synthesis

### TEMPORAL
- Time-aware retrieval
- Chronological formatting in synthesis

### ENUMERATION
- Entity drill-down phase activated
- Up to 40 facts to synthesis
- Unique entities list provided to scorer and synthesizer
- Exhaustive list formatting

## Performance

Based on 10-question evaluation:

| Metric | Value |
|--------|-------|
| Avg query time | ~100s |
| Parallel speedup | 2.7x |
| Avg confidence | 0.99 |
| Avg evidence | 65-97 facts |

## Files

```
src/querying_system/v2/
├── pipeline.py          # Main orchestrator
├── retriever.py         # Scoped + global retrieval
├── resolver.py          # Entity/topic resolution
├── expander.py          # Graph expansion
└── APPROACH.md          # This file

src/querying_system/shared/
├── schemas.py           # Data structures
├── decomposer.py        # Query decomposition
├── scorer.py            # Fact scoring
├── synthesizer.py       # Answer synthesis
├── entity_drilldown.py  # ENUMERATION drill-down
└── prompts.py           # LLM prompts
```
