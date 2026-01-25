# V7 Query Pipeline Design

## Overview

V7 is a simplified query pipeline that follows Microsoft GraphRAG's local search pattern. The core philosophy: **KG curates context, Gemini-3-pro reasons.**

### Key Changes from V6

**Removed**:
- Gap detection loop
- Refinement loop
- LLM fact filtering + critique
- Drilldown LLM selection
- Complex confidence thresholds
- Per-sub-query multi-step processing

**Added (GraphRAG-aligned)**:
- 1-hop entity expansion (neighbor entities + their chunks)
- Explicit FactNode retrieval (relationship descriptions)
- Definition-based entity search (not just names)
- Structured context ordering (guides LLM attention)

**Result**: ~980 LOC vs v6's 2,100 LOC (53% reduction)

---

## Architecture

```
Question → Decompose (gpt-5.1) → [Sub-queries]
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
              [PARALLEL: Per sub-query]
              ├─ Resolve entities (0.3 threshold + gpt-5.1 picks)
              ├─ Resolve topics (0.3 threshold + gpt-5.1 picks)
              ├─ Retrieve: chunks, 1-hop neighbors, FactNodes, topics
              ├─ Global vector search (toggleable)
              ├─ Assemble structured context (ordered by type)
              └─ Synthesize sub-answer (gemini-3-pro)
                    │                 │                 │
                    ▼                 ▼                 ▼
              Sub-answer 1      Sub-answer 2      Sub-answer 3
                    │                 │                 │
                    └─────────────────┼─────────────────┘
                                      ▼
                    Final Merge (gemini-3-pro)
                    Input: [(sub-query, answer), ...]
                                      │
                                      ▼
                              Final Answer
```

---

## Pipeline Flow

### Phase 1: Decomposition

Always decompose the question into sub-queries using gpt-5.1.

**Input**: User question
**Output**: List of sub-queries
**Model**: gpt-5.1

### Phase 2: Per Sub-query Research (Parallel)

Each sub-query runs through an independent Researcher:

#### 2a. Entity Resolution
1. Extract entity mentions from sub-query
2. Vector search against entity names + definitions (threshold: 0.3)
3. LLM (gpt-5.1) picks which candidates match (allows one-to-many)

#### 2b. Topic Resolution
1. Extract topic mentions from sub-query
2. Vector search against topic definitions (threshold: 0.3)
3. LLM (gpt-5.1) picks which candidates match

#### 2c. Retrieval (Parallel)
For each resolved entity/topic:
- **Entity-scoped chunks**: EpisodicNodes where entity appears
- **1-hop neighbors**: Entities connected to resolved entities + their chunks
- **FactNodes**: Relationship descriptions between entities
- **Topic-scoped chunks**: Chunks linked to resolved topics
- **Global vector search** (toggleable): Vector search all chunks with sub-query

#### 2d. Context Assembly
Structured ordering (no token limit):
1. High-relevance text chunks (highest vector scores)
2. Entities + Facts (summaries, relationship descriptions)
3. Topic context (definitions, linked chunks)
4. Lower-relevance chunks (remaining, ordered by score)

#### 2e. Sub-query Synthesis
Single LLM call with full structured context.

**Model**: gemini-3-pro

### Phase 3: Final Merge

Combine sub-answers into final response.

**Input**: List of (sub-query, sub-answer) pairs
**Output**: Final answer
**Model**: gemini-3-pro

---

## LLM Usage

| Step | Model | Count |
|------|-------|-------|
| Decomposition | gpt-5.1 | 1 |
| Entity resolution | gpt-5.1 | 1 per sub-query |
| Topic resolution | gpt-5.1 | 1 per sub-query |
| Sub-query synthesis | gemini-3-pro | 1 per sub-query |
| Final merge | gemini-3-pro | 1 |

**For 3 sub-queries**: 1 + 3 + 3 + 3 + 1 = 11 LLM calls

---

## Configuration

```python
@dataclass
class V7Config:
    # Resolution
    entity_threshold: float = 0.3          # Low threshold, wide net
    topic_threshold: float = 0.3
    search_definitions: bool = True        # Search entity definitions, not just names

    # Retrieval
    enable_1hop_expansion: bool = True     # Grab neighbor entities + their chunks
    enable_global_search: bool = True      # Toggleable global vector search
    global_search_top_k: int = 50          # How many chunks from global search

    # Models
    decomposition_model: str = "gpt-5.1"
    resolution_model: str = "gpt-5.1"
    synthesis_model: str = "gemini-3-pro"

    # Group isolation
    group_id: str = "default"
```

---

## File Structure

```
src/querying_system/v7/
├── __init__.py           # Exports V7Pipeline, query_v7
├── pipeline.py           # Main orchestrator (~150 LOC)
├── researcher.py         # Per-sub-query: resolve → retrieve → synthesize (~300 LOC)
├── graph_store.py        # Neo4j operations: resolution, retrieval, 1-hop (~250 LOC)
├── context_builder.py    # Structured context assembly (~100 LOC)
├── schemas.py            # V7Config, SubAnswer, PipelineResult (~80 LOC)
└── prompts.py            # Decomposition, synthesis, merge prompts (~100 LOC)
```

---

## GraphRAG Alignment

Following Microsoft GraphRAG's local search pattern:

| GraphRAG Concept | V7 Implementation |
|------------------|-------------------|
| Vector search entity descriptions | Vector search names + definitions (0.3 threshold) |
| LLM selects top-k entities | gpt-5.1 picks matches (one-to-many) |
| Connected entities (1-hop) | 1-hop neighbor retrieval |
| Relationships (edge descriptions) | FactNode retrieval |
| Text units (original chunks) | Entity-scoped + topic-scoped chunks |
| Community reports | Topics (similar concept) |
| Token allocation | Structured ordering (no limit) |

**Key difference**: V7 uses structured ordering instead of hard token allocation, trusting Gemini-3-pro's long context capabilities.

---

## Usage

```python
from src.querying_system.v7 import V7Pipeline, query_v7

# Option 1: Async function
result = await query_v7("What economic conditions did Boston report?")

# Option 2: Pipeline class with config
config = V7Config(
    enable_global_search=False,  # Disable global search
    entity_threshold=0.25,       # Even wider net
)
pipeline = V7Pipeline(config=config)
result = await pipeline.query("Compare inflation in Boston vs New York")
```
