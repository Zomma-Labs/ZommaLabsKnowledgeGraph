# Temporal-Aware Scoring and Expansion

## Problem

When we have multiple documents from different time periods (e.g., multiple Beige Books from Oct 2025, July 2025, April 2025), the current pipeline can incorrectly correlate information across time periods.

**Example failure case:**
```
Query: "How did employment change in the October 2025 Beige Book?"

Retrieved facts:
- Fact A (Oct 2025): "Employment increased modestly in manufacturing"
- Fact B (July 2025): "Employment declined sharply due to layoffs"
- Fact C (Oct 2025): "Labor markets remained tight"

Current behavior: Expander might follow graph paths from Fact B's entities,
pulling in July 2025 context that gets mixed into the October answer.
```

## Proposed Solution

### 1. Add `temporal_scope_match` to Scoring

Modify the scorer to output a temporal match flag for each fact:

```python
class FactScore(BaseModel):
    fact_index: int
    relevance: float  # 0-1
    should_expand: bool
    temporal_scope_match: bool  # NEW: Does this fact's date match query's temporal focus?
```

The scorer prompt would include:
```
The query focuses on: {temporal_scope}  (e.g., "October 2025", "Q3 2025", "2025")

For each fact, set temporal_scope_match=true ONLY if the fact's date falls within
or is relevant to the query's temporal scope. Set false for facts from different
time periods that should not be correlated with the query's focus period.
```

### 2. Temporal-Aware Expansion

Modify the expander to only expand from temporally-matched facts:

```python
for fact in evidence_pool.scored_facts:
    if fact.should_expand and fact.temporal_scope_match and fact.final_score > 0.3:
        entities_to_expand.add(fact.subject)
        entities_to_expand.add(fact.object)
```

### 3. Temporal-Aware Graph Traversal

When expanding, filter results by date proximity:

```cypher
MATCH (e {name: $entity_name})-[r1]->(c:EpisodicNode)-[r2]->(target)
OPTIONAL MATCH (d:DocumentNode)-[:CONTAINS_CHUNK]->(c)
WHERE d.document_date = $target_date  // Only same-period facts
   OR d.document_date IS NULL         // Allow undated facts
RETURN ...
```

### 4. Decomposer Enhancement

Ensure the decomposer extracts temporal scope from queries:

```python
class QueryDecomposition(BaseModel):
    # ... existing fields ...
    temporal_scope: str | None  # "October 2025", "Q3 2025", "2024-2025", etc.
    temporal_type: Literal["point", "range", "comparison", "none"]
```

Temporal types:
- `point`: Single time period ("October 2025 Beige Book")
- `range`: Time range ("from Q1 to Q3 2025")
- `comparison`: Comparing periods ("How did Q3 differ from Q2?")
- `none`: No temporal focus

### 5. Cross-Temporal Queries (Special Case)

For comparison queries across time periods:

```
Query: "How did inflation trends change between the July and October 2025 Beige Books?"
```

The decomposer would set `temporal_type="comparison"` and the scorer would:
- Mark July 2025 facts with `temporal_group="july_2025"`
- Mark October 2025 facts with `temporal_group="october_2025"`
- Expansion stays within each temporal group (no cross-contamination)
- Synthesizer structures answer by time period

## Implementation Order

1. **Phase 1**: Add `temporal_scope` extraction to decomposer
2. **Phase 2**: Add `temporal_scope_match` to scorer output
3. **Phase 3**: Filter expansion by temporal match
4. **Phase 4**: Add date filtering to Cypher queries
5. **Phase 5**: Handle cross-temporal comparison queries

## Testing Strategy

Create test cases with multiple Beige Books:
- Load Oct 2025 + July 2025 Beige Books
- Query about October specifically
- Verify answer doesn't include July data
- Query comparing both periods
- Verify answer correctly separates the two

## Metrics

- **Temporal precision**: % of cited facts from correct time period
- **Cross-contamination rate**: % of answers mixing unrelated time periods
- **Comparison accuracy**: Correct attribution in cross-temporal queries
