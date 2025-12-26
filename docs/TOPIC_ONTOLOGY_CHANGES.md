# Topic Ontology System - Implementation Notes

## Problem Statement

The entity extraction pipeline was producing garbage topics like "Transparency And Oversight" that polluted the knowledge graph. Topics should be constrained to a controlled vocabulary while entities (companies, people, places) remain unconstrained since they're infinite.

## Solution Architecture

### Key Insight: Entities vs Topics

| Aspect | Entities | Topics |
|--------|----------|--------|
| Cardinality | Infinite (any company, person, place) | Finite (economic concepts, indicators) |
| Validation | Graph deduplication only | Ontology-constrained |
| Source | Extracted from text | Must match controlled vocabulary |
| Examples | Apple Inc, Eric Schmidt, Minneapolis | Inflation, GDP, Labor Market |

### Decision: Remove FIBO for Entities, Keep Ontology for Topics

- **FIBO was removed for entity resolution** - entities are deduplicated against the graph only
- **Topic ontology created** - combines FIBO IND module + custom topics
- **Rejected topics logged** - for manual review and potential ontology expansion

## Files Created

### 1. Custom Topics Definition
**File:** `src/config/topics/financial_topics.json`

69 hand-curated financial topics covering:
- Macroeconomic indicators (Inflation, GDP, Unemployment)
- Market concepts (Volatility, Liquidity, Credit Spreads)
- Business themes (Revenue, Profitability, Innovation)
- Policy topics (Monetary Policy, Fiscal Policy, Regulation)
- Technology topics (AI, Cloud Computing, Cybersecurity)

Each topic has:
```json
{
  "uri": "https://kg.zommalabs.com/topic/Inflation",
  "label": "Inflation",
  "definition": "A general increase in prices...",
  "synonyms": ["Price Increases", "CPI", "Inflationary Pressure"]
}
```

### 2. Topic Loader
**File:** `src/tools/topic_loader.py`

Combines two sources:
1. **FIBO IND Module** (~600 concepts) - Economic indicators, interest rates, market indices
2. **Custom Topics** (~69 concepts) - Business/tech topics not in FIBO

**Usage:**
```bash
uv run python src/tools/topic_loader.py
```

**Output:**
- Creates `qdrant_topics/` directory with `topic_ontology` collection
- Final count: 669 unique topics (after deduplication)

### 3. Topic Librarian
**File:** `src/agents/topic_librarian.py`

Hybrid resolution using:
1. **Vector Search** - Semantic matching via Voyage AI embeddings
2. **Fuzzy Search** - Lexical matching via RapidFuzz

**Key methods:**
- `resolve(text, threshold=0.70)` - Returns canonical match or None
- `resolve_topics(topics_list)` - Batch validation, returns only valid topics

**Threshold:** 0.70 (lower than entity FIBO was at 0.90, since topics are more abstract)

## Pipeline Changes

### File: `src/workflows/main_pipeline.py`

#### Removed
- `FIBOLibrarian` import and initialization
- FIBO resolution in `resolve_single_item()` function
- FIBO resolution in `initialize_episode()` for header dimensions

#### Added
- `TopicLibrarian` import and initialization
- Topic validation in `parallel_resolution_node()`:
  ```python
  for raw_topic in fact.topics:
      match = topic_librarian.resolve(raw_topic)
      if match:
          validated_topics.append(match['label'])
      else:
          # Log rejected topic for manual review
          with open("rejected_topics.log", "a") as f:
              f.write(f"{raw_topic}\t|\t{fact.fact[:100]}...\n")
  ```
- Topic ontology resolution in `initialize_episode()` for header topics

## Resolution Flow (Updated)

### Topics
```
Extracted Topic -> TopicLibrarian.resolve() -> Match Found?
                                                  |
                                    Yes           |           No
                                     |            |            |
                          Use canonical name   Log to rejected_topics.log
                          Add to graph         Don't add to graph
```

### Entities
```
Extracted Entity -> Graph Deduplication -> Match Found?
                                              |
                                Yes           |           No
                                 |            |            |
                         Merge with existing  Create new EntityNode
```

## Rejected Topics Log

**File:** `rejected_topics.log` (created at runtime)

Format:
```
Transparency And Oversight	|	Former executive Eric Schmidt revealed in the conference...
Random Garbage Topic	|	Some fact text here...
```

Review this file periodically to:
1. Identify common rejected topics that should be added to ontology
2. Spot patterns in extraction that produce garbage
3. Expand `financial_topics.json` with valid missing topics

## Running the System

### 1. Index Topic Ontology (one-time or after changes)
```bash
uv run python src/tools/topic_loader.py
```

### 2. Run Pipeline
```bash
uv run scripts/run_pipeline.py
```

### 3. Review Rejected Topics
```bash
cat rejected_topics.log | sort | uniq -c | sort -rn | head -20
```

## Test Results

Topic resolution examples:
| Input | Output | Score |
|-------|--------|-------|
| "Inflation" | Inflation | 0.97 |
| "Price Increases" | Inflation | 0.87 |
| "AI" | Artificial Intelligence | 0.96 |
| "CPI" | Inflation | 1.00 |
| "Random Garbage Topic" | REJECTED | 0.55 |
| "Transparency And Oversight" | Transparency | 0.79 |

## Future Improvements

1. **Expand ontology** - Review rejected_topics.log and add valid topics
2. **Threshold tuning** - May need to adjust 0.70 threshold based on precision/recall
3. **Topic hierarchy** - Could add parent/child relationships between topics
4. **Synonym expansion** - Add more synonyms to improve matching
