# OpenAI Embeddings Migration Plan

## Decision

**Fully retire Voyage embeddings and migrate to OpenAI `text-embedding-3-large`.**

Re-run the ingestion pipeline to generate fresh embeddings.

---

## Evidence: Why OpenAI?

### Test 1: "Which districts reported slight to modest economic growth?"

| Metric | Voyage | OpenAI |
|--------|--------|--------|
| Top score | 0.56 | 0.78 |
| Score spread | 0.10 | 0.17 |
| Gap #1 to #2 | 0.06 | 0.13 |

### Test 2: "What happened to manufacturing in the Chicago District?"

| Metric | Voyage | OpenAI |
|--------|--------|--------|
| Top score | 0.45 | 0.77 |
| Score spread | 0.04 | 0.12 |
| Gap #1 to #2 | 0.003 | 0.09 |

**Key Finding:** OpenAI embeddings have ~3x better score separation, enabling threshold-only retrieval without LLM scoring.

---

## Implementation Order

1. Update `llm_client.py` - Replace Voyage with OpenAI
2. Update `pipeline.py` - Change index dimensions to 3072
3. Update `setup_graph_index.py` - Change fact index dimensions to 3072
4. Create V6 querying system - Threshold-only retrieval
5. Clear Neo4j and re-run ingestion pipeline
6. Evaluate V6 vs V5
7. Cleanup - Remove Voyage dependencies

---

## File Changes

### 1. Embedding Layer

**File:** `src/util/llm_client.py`

```python
# REMOVE (line 37):
from langchain_voyageai import VoyageAIEmbeddings

# ADD:
from langchain_openai import OpenAIEmbeddings

# REPLACE get_embeddings static method (lines 32-38):
@staticmethod
def get_embeddings(model="text-embedding-3-large"):
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=model)

# REPLACE get_embeddings() helper (lines 88-93):
def get_embeddings():
    """OpenAI text-embedding-3-large (3072 dimensions)."""
    return LLMClient.get_embeddings(model="text-embedding-3-large")

# REPLACE get_dedup_embeddings() helper (lines 95-100):
def get_dedup_embeddings():
    """OpenAI text-embedding-3-large for deduplication."""
    return LLMClient.get_embeddings(model="text-embedding-3-large")
```

---

### 2. Ingestion Pipeline

**File:** `src/pipeline.py`

**Function:** `_ensure_vector_indexes()` (lines 95-156)

Change all `vector.dimensions` from `1024` to `3072`:

| Index | Line | Change |
|-------|------|--------|
| `entity_name_embeddings` | 107 | 1024 → 3072 |
| `entity_name_only_embeddings` | 129 | 1024 → 3072 |
| `topic_embeddings` | 148 | 1024 → 3072 |

---

**File:** `src/scripts/setup_graph_index.py`

Change `fact_embeddings` index dimensions: 1024 → 3072

---

### 3. V6 Querying System

**Create folder:** `src/querying_system/v6/`

| File | Action |
|------|--------|
| `__init__.py` | Copy from V5 |
| `pipeline.py` | Copy from V5 |
| `graph_store.py` | Copy from V5 (no changes needed) |
| `schemas.py` | Copy from V5, add threshold config |
| `researcher.py` | Copy from V5, replace LLM scoring with threshold |
| `prompts.py` | Copy from V5, remove scoring prompts |

---

**File:** `v6/schemas.py`

Add to `ResearcherConfig`:

```python
class ResearcherConfig(BaseModel):
    # ... existing fields ...

    # V6: Threshold-only retrieval (no LLM scoring)
    use_llm_scoring: bool = False
    relevance_threshold: float = 0.65
```

---

**File:** `v6/researcher.py`

**Remove:**
- `self.scoring_llm` initialization (line 114)
- `self.scoring_and_gap` structured output (line 118)
- `_score_and_identify_gaps()` method (lines 303-376)

**Add:**
```python
async def _filter_by_threshold(
    self,
    facts: list[RawFact],
    target_info: str,
) -> tuple[list[ScoredFact], list[Gap]]:
    """
    V6: Filter facts by embedding score threshold.

    OpenAI embeddings have good discrimination - scores >= 0.65 are relevant.
    No LLM scoring needed.
    """
    if not facts:
        return [], []

    # Filter by threshold
    filtered = [
        ScoredFact.from_raw(f, final_score=f.vector_score)
        for f in facts
        if f.vector_score >= self.config.relevance_threshold
    ]

    # Sort by score descending
    filtered.sort(key=lambda f: f.final_score, reverse=True)

    # Limit to max_facts_to_score
    filtered = filtered[:self.config.max_facts_to_score]

    # Simple gap detection: if too few facts, suggest expansion
    gaps = []
    if self.config.enable_gap_expansion and len(filtered) < 5:
        for fact in filtered[:3]:
            if fact.subject:
                gaps.append(Gap(missing="more context", expand_from=fact.subject))

    return filtered, gaps
```

**Update `research()` method:**
- Replace call to `_score_and_identify_gaps()` with `_filter_by_threshold()`

---

**File:** `v6/prompts.py`

**Remove:**
- `SCORING_AND_GAP_SYSTEM_PROMPT`
- `SCORING_AND_GAP_USER_PROMPT`
- `format_facts_for_scoring()`

---

### 4. Cleanup

**File:** `pyproject.toml`

Remove dependency:
```toml
langchain-voyageai = "..."
```

---

**File:** `CLAUDE.md`

Update references:
- `voyage-finance-2` → `text-embedding-3-large`
- `voyage-3-large` → `text-embedding-3-large`
- `1024 dimensions` → `3072 dimensions`
- Remove `VOYAGE_API_KEY` from environment variables

---

## Summary Table

| Component | File(s) | Change |
|-----------|---------|--------|
| Embeddings | `llm_client.py` | Voyage → OpenAI |
| Ingestion | `pipeline.py` | Index dimensions 1024 → 3072 |
| Ingestion | `setup_graph_index.py` | Fact index 1024 → 3072 |
| V6 Config | `v6/schemas.py` | Add `relevance_threshold=0.65` |
| V6 Retrieval | `v6/researcher.py` | LLM scoring → threshold filter |
| V6 Prompts | `v6/prompts.py` | Remove scoring prompts |
| Cleanup | `pyproject.toml` | Remove `langchain-voyageai` |
| Docs | `CLAUDE.md` | Update embedding docs |

---

## Expected Outcomes

| Metric | Before (V5 + Voyage) | After (V6 + OpenAI) |
|--------|----------------------|---------------------|
| Time per question | ~10 min | ~1-2 min |
| LLM calls per researcher | 5-7 | 2-3 |
| Embedding dimensions | 1024 | 3072 |
| Retrieval method | LLM scoring | Threshold filter |
