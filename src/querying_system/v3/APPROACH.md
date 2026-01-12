# V3 Querying System: Threshold-Based Retrieval

## Overview

V3 simplifies the query pipeline by replacing top-k limits with similarity thresholds. Instead of "get top 20 facts per entity", we get "ALL facts with similarity > 0.7".

## Key Insight

The V2 approach had a fundamental tension:
- **top_k too low**: Miss relevant facts (especially for ENUMERATION)
- **top_k too high**: Include noise, waste LLM tokens

Threshold-based retrieval resolves this:
- Sparse topics naturally get fewer facts
- Dense topics get more (because they have more relevant content)
- No arbitrary cutoff

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
    ▼
┌────────────────────────┐
│  Phase 2: Resolve      │
│    (gpt-5-mini)        │
└────────────────────────┘
    │
    ▼
┌────────────────────────┐
│  Phase 3: Threshold    │
│  Retrieve (sim > 0.7)  │
│  No k-limit!           │
└────────────────────────┘
    │
    ▼
┌────────────────────────┐
│  Phase 4: Synthesize   │
│     (gpt-5.1)          │
└────────────────────────┘
    │
    ▼
Answer
```

## What V3 Removes

1. **Scoped vs Global split** - Just one threshold search
2. **LLM scoring phase** - Vector similarity is the score
3. **Expansion phase** - Threshold captures connected facts naturally
4. **Drill-down agent** - Deterministic retrieval, no agent decisions

## LLM Calls

| Phase | Model | Purpose |
|-------|-------|---------|
| Decomposition | gpt-5.1 | Extract entities/topics |
| Resolution | gpt-5-mini | Verify graph matches |
| Synthesis | gpt-5.1 | Generate answer |

**Total: 3 LLM calls** (vs 5-6 in V2)

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `similarity_threshold` | 0.7 | Minimum cosine similarity to include |
| `top_k_evidence` | 40 | Max facts sent to synthesis |

## Threshold Selection

- **0.8+**: Very strict, only highly relevant facts
- **0.7**: Balanced (recommended starting point)
- **0.6**: More permissive, good for exploratory questions
- **0.5**: Very permissive, risk of noise

## Usage

```bash
# CLI
uv run -m src.querying_system.v3.pipeline "Which districts reported growth?" -v

# With custom threshold
uv run -m src.querying_system.v3.pipeline "Which districts reported growth?" -t 0.6 -v

# Python
from src.querying_system.v3 import query_v3
result = query_v3("Which districts reported growth?", similarity_threshold=0.7)
```

## Files

```
src/querying_system/v3/
├── __init__.py
├── pipeline.py      # Main orchestrator
├── retriever.py     # Threshold-based retrieval
└── APPROACH.md      # This file
```

## Comparison with V2

| Aspect | V2 | V3 |
|--------|----|----|
| Retrieval | top_k per node | similarity > threshold |
| LLM calls | 5-6 | 3 |
| Scoring | Separate LLM phase | Vector similarity only |
| Expansion | Graph traversal | None (threshold captures) |
| Drill-down | Agent selects entities | None (deterministic) |
| Complexity | High | Low |
