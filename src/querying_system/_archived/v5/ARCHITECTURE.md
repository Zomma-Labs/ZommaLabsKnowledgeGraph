# V5 Pipeline Architecture

Entity-Anchored Deep Research pipeline for knowledge graph QA.

## High-Level Flow

```
Question
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: DECOMPOSE (Sequential)                                        │
│  ─────────────────────────────────                                      │
│  Input: Question                                                        │
│  Output: SubQuery[], QuestionType, RequiredInfo[]                       │
│  LLM: gpt-5.1 (1 call)                                                  │
└─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: RESEARCH (PARALLEL - asyncio.gather with semaphore)           │
│  ────────────────────────────────────────────────────────────           │
│                                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                      │
│  │ Researcher 1│  │ Researcher 2│  │ Researcher N│  ← Run in PARALLEL   │
│  │             │  │             │  │             │    (max_concurrent=5)│
│  │ (sub-query) │  │ (sub-query) │  │ (sub-query) │                      │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                      │
│         │                │                │                             │
│         ▼                ▼                ▼                             │
│    SubAnswer_1      SubAnswer_2      SubAnswer_N                        │
└─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: FINAL SYNTHESIS (Sequential)                                  │
│  ─────────────────────────────────────                                  │
│  Input: SubAnswer[], QuestionType                                       │
│  Output: Final answer with citations                                    │
│  LLM: gpt-5.1 (1 call)                                                  │
└─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
PipelineResult
```

## Researcher Internal Flow (Per Sub-Query)

Each researcher runs **sequentially** through these steps:

```
SubQuery
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 1: RESOLVE (Sequential)                                           │
│  ─────────────────────────────                                          │
│  - Entity hints → Resolved EntityNodes (vector search + LLM verify)     │
│  - Topic hints → Resolved TopicNodes                                    │
│  LLM: gpt-5-mini (verification calls)                                   │
└─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 2: RETRIEVE - DUAL PATH (PARALLEL)                                │
│  ───────────────────────────────────────                                │
│  All searches run via asyncio.gather:                                   │
│                                                                         │
│  SCOPED (per resolved entity/topic):    GLOBAL (always runs):           │
│  ├─ Entity 1 facts search               ├─ Vector search (top_k=30)     │
│  ├─ Entity 2 facts search               └─ Keyword search (top_k=30)    │
│  ├─ Entity N facts search                                               │
│  ├─ Topic 1 facts search                                                │
│  └─ Topic M facts search                                                │
│                                                                         │
│  → All results merged + deduped by fact_id                              │
└─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 3: SCORE + GAP DETECTION (Sequential)                             │
│  ──────────────────────────────────────────                             │
│  - LLM scores each fact for relevance (0-1)                             │
│  - LLM identifies information gaps                                      │
│  LLM: gpt-5-mini (batch scoring call)                                   │
└─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 4: GAP EXPANSION (Sequential, if gaps exist)                      │
│  ─────────────────────────────────────────────────                      │
│  For each gap:                                                          │
│    - Expand from suggested entity → get related facts                   │
│  → Merge expanded facts into scored set                                 │
└─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 5: ENTITY DRILL-DOWN (Sequential, ENUMERATION only)               │
│  ────────────────────────────────────────────────────────               │
│  - LLM selects entities needing more facts                              │
│  - Expand from each selected entity                                     │
│  → Ensures comprehensive coverage for list questions                    │
│  LLM: gpt-5-mini (entity selection)                                     │
└─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 6: SYNTHESIZE SUB-ANSWER (Sequential)                             │
│  ──────────────────────────────────────────                             │
│  - Combine top scored facts into coherent answer                        │
│  - Add citations [Source: doc, date]                                    │
│  LLM: gpt-5.1 (synthesis call)                                          │
└─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 7: REFINEMENT LOOP (Sequential, max 2 iterations)                 │
│  ──────────────────────────────────────────────────────                 │
│  For each iteration:                                                    │
│    1. Detect vagueness (compare answer vs evidence)                     │
│    2. If vague: run targeted searches for specifics                     │
│    3. Re-synthesize with new evidence                                   │
│  LLM: gpt-5-mini (vagueness detection) + gpt-5.1 (re-synthesis)         │
└─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
SubAnswer
```

## Parallelization Summary

| Level | Component | Parallel? | Details |
|-------|-----------|-----------|---------|
| Pipeline | Researchers | **YES** | Up to `max_concurrent` (default 5) sub-queries processed simultaneously |
| Researcher | Steps 1-7 | NO | Steps run sequentially within each researcher |
| Retrieval | Searches | **YES** | All scoped + global searches run via asyncio.gather |
| Expansion | Per gap | NO | Gaps processed sequentially (could be parallelized) |
| Refinement | Per iteration | NO | Loops run sequentially (by design - each builds on previous) |

## File Structure

```
src/querying_system/v5/
├── __init__.py           # Exports
├── pipeline.py           # Main orchestrator (Phases 1-3)
├── researcher.py         # Per-subquery agent (Steps 1-7)
├── graph_store.py        # Unified Neo4j/Qdrant access layer
├── schemas.py            # Data models (SubAnswer, ScoredFact, etc.)
├── prompts.py            # All LLM prompts
└── ARCHITECTURE.md       # This file
```

## LLM Usage

| Step | Model | Purpose | Cost |
|------|-------|---------|------|
| Decomposition | gpt-5.1 | Break question into sub-queries | $$ |
| Resolution | gpt-5-mini | Verify entity matches | $ |
| Scoring | gpt-5-mini | Batch score facts for relevance | $ |
| Gap Detection | gpt-5-mini | Identify missing information | $ |
| Drill-down | gpt-5-mini | Select entities for ENUMERATION | $ |
| Sub-synthesis | gpt-5.1 | Generate sub-answer | $$ |
| Vagueness | gpt-5-mini | Detect vague references | $ |
| Refinement | gpt-5.1 | Re-synthesize with specifics | $$ |
| Final Synthesis | gpt-5.1 | Combine sub-answers | $$ |

## Configuration Flags

```python
@dataclass
class ResearcherConfig:
    # Retrieval
    enable_global_search: bool = True      # Always run global search
    global_top_k: int = 30                 # Facts from global search
    scoped_threshold: float = 0.3          # Min similarity for scoped

    # Expansion
    enable_gap_expansion: bool = True      # LLM-guided expansion
    enable_entity_drilldown: bool = True   # ENUMERATION drill-down
    drilldown_max_entities: int = 10       # Max entities to drill

    # Refinement
    enable_refinement_loop: bool = True    # Refine vague answers
    max_refinement_loops: int = 2          # Max iterations
    refinement_search_top_k: int = 20      # Facts per refinement search

    # Scoring
    max_facts_to_score: int = 50           # Max facts sent to LLM
    top_k_evidence: int = 15               # Facts kept for synthesis
```

## Key Design Decisions

1. **Dual-path retrieval**: Always runs both scoped (entity/topic) and global (vector/keyword) searches in parallel. Catches edge cases where resolution fails.

2. **Per-subquery synthesis**: Each researcher produces a complete sub-answer, not just facts. Final synthesis combines answers, not raw evidence.

3. **LLM-guided expansion**: Instead of hardcoding multi-hop rules, the LLM identifies gaps and suggests which entities to expand from.

4. **Refinement loop**: Compares the synthesized answer against its evidence to detect count mismatches or vague references, then searches for specifics.

5. **Configurable features**: All expansion/refinement features can be toggled for A/B testing.
