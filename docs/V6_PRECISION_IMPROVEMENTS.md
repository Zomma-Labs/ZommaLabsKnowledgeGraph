# V6 Precision Improvements Plan

## Problem Statement

V6 retrieves 58 chunks (32% of corpus) for a question needing ~3-5 chunks.

**Root causes identified:**
1. Topic resolution expands to loosely-related topics (Employment, Labor Market, Consumer Spending)
2. Topic search returns ALL facts for those topics without entity scoping
3. Vector embeddings don't distinguish direction (growth vs decline)

**Example:**
```
"economic growth" → [Economic Activity, Employment, Labor Market, Consumer Spending, ...]
                                          ↓
                    Topic search returns 200+ facts each, unfiltered
                                          ↓
                    58 chunks retrieved, 26% precision
```

---

## Solution 1: Raise Thresholds (Quick Win)

**Current:** `relevance_threshold = 0.65`

**Test:** Try 0.70, 0.75, 0.80 and measure precision/recall trade-off.

| Threshold | Expected Chunks | Precision | Recall Risk |
|-----------|-----------------|-----------|-------------|
| 0.65 | 58 | 26% | Full |
| 0.70 | ~40? | ~40%? | Low |
| 0.75 | ~25? | ~60%? | Medium |
| 0.80 | ~15? | ~80%? | High (may miss relevant) |

**Pros:** Zero code changes, instant improvement
**Cons:** May hurt recall, doesn't fix root cause

---

## Solution 2: Entity-Scoped Topic Search

**Core Idea:** When searching for facts about a topic, scope results to facts that ALSO involve resolved entities.

```python
# Before (returns 200+ facts)
search_topic_facts("Economic Activity")

# After (returns ~20 facts)
search_topic_facts_scoped("Economic Activity", ["Boston", "Richmond", ...])
```

### Implementation

**New method in `graph_store.py`:**
```python
async def search_topic_facts_scoped(
    self,
    topic_name: str,
    entity_names: list[str],
    query_embedding: list[float],
    threshold: float = 0.3,
) -> list[RawFact]:
    """Search facts connected to topic AND involving specific entities."""
```

**Cypher change:**
```cypher
-- Add to WHERE clause:
AND (subj.name IN $entity_names OR obj.name IN $entity_names)
```

---

## Solution 3: Dynamic Entity Filtering for Vague Terms

**Problem:** "districts" resolves to generic entities, not specific ones like "Boston", "Richmond".

**Solution:** Discover relevant entities from the data itself using embeddings/LLM.

### Flow

```
Query: "Which districts reported growth?"
Vague term: "districts"

Step 1: Vector search → 100 facts mentioning various entities:
  - "Boston Federal Reserve District"
  - "Consumer Spending"
  - "Wage Growth"
  - "Richmond"
  - "Fifth District"
  ...

Step 2: Filter entities by similarity to "districts":

  Embedding approach:
    embed("districts") vs embed("Boston Federal Reserve District") → 0.82 ✓
    embed("districts") vs embed("Consumer Spending") → 0.31 ✗
    embed("districts") vs embed("Fifth District") → 0.85 ✓

  OR LLM approach:
    "Which of these are Federal Reserve Districts?"
    → ["Boston Federal Reserve District", "Richmond", "Fifth District", ...]

Step 3: Keep only facts where subject OR object is in filtered list
  - 100 facts → ~15 facts (all about districts)
```

### Implementation

```python
async def filter_entities_by_vague_term(
    self,
    vague_term: str,
    candidate_entities: list[str],
    threshold: float = 0.5,
) -> list[str]:
    """Filter candidates to those matching the vague term."""

    # Embed vague term
    vague_embedding = await self.embed_text(vague_term)

    # Embed all candidates (batch)
    candidate_embeddings = await self.batch_embed(candidate_entities)

    # Score and categorize
    high_confidence = []  # score > 0.7
    borderline = []       # 0.4 < score <= 0.7

    for entity, emb in zip(candidate_entities, candidate_embeddings):
        score = cosine_similarity(vague_embedding, emb)
        if score > 0.7:
            high_confidence.append(entity)
        elif score > 0.4:
            borderline.append(entity)

    # LLM for borderline cases only
    if borderline:
        llm_filtered = await self._llm_filter_entities(vague_term, borderline)
        high_confidence.extend(llm_filtered)

    return high_confidence
```

### Detecting Vague Entities

```python
def is_vague_entity(entity_name: str, resolution_confidence: float) -> bool:
    # Short, generic names
    vague_terms = ['districts', 'banks', 'regions', 'sectors', 'companies']
    if entity_name.lower() in vague_terms:
        return True
    # Low resolution confidence
    if resolution_confidence < 0.7:
        return True
    return False
```

---

## Solution 4: Post-Retrieval Direction Filter

**Problem:** Vector embeddings treat "grew modestly" and "declined modestly" as similar.

**Solution:** After retrieval, filter by directional keywords.

```python
def filter_by_direction(facts: list, question: str) -> list:
    q = question.lower()

    if any(w in q for w in ['growth', 'grew', 'increase', 'expand']):
        required = ['grew', 'growth', 'expand', 'increase', 'rose', 'uptick']
        excluded = ['declined', 'fell', 'contracted', 'softened', 'down']
    elif any(w in q for w in ['decline', 'decrease', 'fell', 'contract']):
        required = ['declined', 'fell', 'contracted', 'softened', 'down']
        excluded = ['grew', 'growth', 'expand', 'increase', 'rose']
    elif any(w in q for w in ['no change', 'unchanged', 'flat']):
        required = ['unchanged', 'flat', 'stable', 'little changed']
        excluded = []
    else:
        return facts

    return [f for f in facts
            if any(r in f.content.lower() for r in required)
            and not any(e in f.content.lower() for e in excluded)]
```

---

## Implementation Priority

| Priority | Solution | Effort | Impact |
|----------|----------|--------|--------|
| 1 | **Raise thresholds** | None | Medium |
| 2 | **Direction filter** | Low | High for enumeration |
| 3 | **Entity-scoped topic search** | Medium | High |
| 4 | **Dynamic vague entity filtering** | Medium | High for vague queries |

---

## Configuration Flags

```python
@dataclass
class ResearcherConfig:
    # Existing
    relevance_threshold: float = 0.65  # Try 0.70-0.75

    # NEW
    enable_scoped_topic_search: bool = True
    enable_direction_filter: bool = True
    enable_vague_entity_expansion: bool = True
    vague_entity_similarity_threshold: float = 0.5
```

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Chunks retrieved | 58 | 15-25 |
| Precision | 26% | 70%+ |
| Recall (growth districts) | 1/3 | 3/3 |
| Query time | ~3 min | ~2 min |

---

## IMPLEMENTED (2024-01-13)

### Changes Made

1. **Top-match-only topic resolution** (`graph_store.py`)
   - Added `TOPIC_RESOLUTION_SYSTEM_PROMPT` - instructs LLM to return only single best match
   - Added `_verify_topic_candidate()` method - returns `Optional[str]` instead of `list[str]`
   - Modified `_resolve_topics()` to use new method

2. **LLM fact filter with critique** (`researcher.py`)
   - Added `FactFilterResult` and `FactFilterCritique` schemas
   - Added `FACT_FILTER_*` prompts
   - Added `_filter_facts_with_llm()` method with critique loop
   - Integrated after Step 5 (drill-down) and before Step 6 (synthesis)
   - Config flags: `enable_llm_fact_filter`, `enable_fact_filter_critique`

### Results (Q5: "Which districts reported slight to modest growth?")

| Metric | Before | After |
|--------|--------|-------|
| Topics resolved | 9 | **1** |
| Facts after threshold | 50 | 50 |
| Facts after LLM filter | N/A | **5** |
| Final evidence | 58 | **5** |
| Districts found | 2/3 | **3/3** ✅ |
| Query time | ~10 min | ~1 min |

### How It Works

```
Topic Resolution:
  "economic growth" → [Economic Activity, Employment, Labor Market, ...]
                                    ↓ (new: top match only)
  "economic growth" → [Economic Activity]

Fact Filtering:
  50 facts (threshold) → LLM filter → 4 facts → Critique → 5 facts
                                                   ↓
                               Added 2 missed, removed 1 wrong
```

---

## Test Cases

**Q5:** "Which districts reported slight to modest economic growth?"
- Expected: Boston, Philadelphia, Richmond
- Current: Missing Richmond, 26% precision
- Target: All 3, 70%+ precision

**Q6:** "Which districts reported no change in economic activity?"
- Expected: Cleveland, Atlanta, Chicago, St. Louis, Dallas
- Current: Missing Atlanta
- Target: All 5
