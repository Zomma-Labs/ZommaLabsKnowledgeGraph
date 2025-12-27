# Evaluation Issues & Fixes

Based on analysis of `eval_results_20251226_132009.json` (58% strict accuracy, 86% lenient).

## Issues Overview

| # | Issue | Impact | Priority | Status |
|---|-------|--------|----------|--------|
| 1 | Agent ignores retrieved chunks | 6 incorrect answers | HIGH | **DONE** |
| 2 | Entity summaries capped at 1-2 sentences | Loss of information | MEDIUM | **DONE** |
| 3 | Duplicate entities (Alphabet vs Alphabet Inc.) | Inconsistent query results | HIGH | **DONE** |
| 4 | Missing entities (Nest Labs, Chronicle) | Incorrect list answers | MEDIUM | TODO |
| 5 | Free-form topics (no schema) | Inconsistent topic nodes | LOW | **DONE** |
| 6 | MCP resolve lacks entity type rules | Poor entity resolution | LOW | TODO |

---

## Issue 1: Agent Ignores Retrieved Chunks (HIGH) - **FIXED**

### Evidence
Q18: "When was Google reorganized as LLC?"
- Agent retrieved chunk: `"As of September 1, 2o17, their equity is held by XXVI Holdings..."`
- Agent answered: "August 10, 2015" (WRONG)
- The correct date was IN the chunk but ignored

### Root Cause
The QA agent retrieves evidence but doesn't systematically reason about it before answering. It falls back to hallucination when it should be quoting from retrieved data.

### Fix Applied (Option A: Think Tool)
Based on Anthropic's "think tool" pattern (https://www.anthropic.com/engineering/claude-think-tool):

1. **Added `think` tool to MCP server** (`mcp_server.py`):
   - Simple tool that takes a `thought` string and returns acknowledgment
   - Forces the agent to explicitly reason about retrieved evidence before answering

2. **Updated system prompt** (`kg_agent.py`):
   - Added think tool to the workflow as REQUIRED step
   - Added detailed instructions with example usage
   - Agent must list specific facts, quote relevant text, identify citations

3. **Updated MCP server instructions**:
   - Added think tool to recommended workflow pattern
   - Included example of proper think tool usage

### Files Modified
- `src/agents/mcp_server.py` - Added `think` tool
- `src/agents/kg_agent.py` - Updated SYSTEM_PROMPT with think tool instructions

### Acceptance Criteria
- [x] Think tool forces agent to analyze evidence before answering
- [x] System prompt requires explicit fact extraction from chunks
- [ ] Q18, Q30, Q32 should pass after fix (needs evaluation re-run)

---

## Issue 2: Entity Summaries Capped (MEDIUM) - **FIXED**

### Evidence
`graph_enhancer.py:86-89`:
```python
f"provide a brief, 1-sentence summary/definition of what '{entity_name}' IS.\n"
```

Ruth Porat's summary mentions "concluded her tenure" but doesn't specify the new CFO or dates.

### Root Cause
Arbitrary 1-2 sentence limit discards useful information like:
- Temporal context (when did role change?)
- Related entities (who replaced them?)
- Key events (what happened?)

### Fix Applied
1. **Updated `extract_entity_summary()`** to request comprehensive summaries (2-4 sentences)
2. **Updated `batch_extract_summaries()`** with detailed guidelines:
   - For people: role/title, organization, key dates (tenure start/end), notable actions
   - For organizations: type, parent/subsidiary relationships, key events, founding dates
   - For events/documents: what it is, when it occurred, key details

### Files Modified
- `src/agents/graph_enhancer.py` lines 81-95, 117-150

### Acceptance Criteria
- [x] Summaries contain all relevant facts from source text
- [ ] Q9 (CFO question) should have enough info to distinguish current vs former (needs re-ingestion)

---

## Issue 3: Duplicate Entities (HIGH) - **FIXED**

### Evidence
Graph has both:
- `Alphabet` → LED_BY[2]: Ruth Porat, Anat Ashkenazi
- `Alphabet Inc.` → ESTABLISHED_SUBSIDIARY[2]: Google, ...

These are the SAME company but with different relationships attached.

### Root Cause
The `SimilarityLockManager` in `similarity_lock.py` used a **0.90 threshold** for detecting conflicts during parallel processing. This was too high:
- "Alphabet" vs "Alphabet Inc." embeddings have ~0.87 similarity
- They weren't flagged as conflicts
- Both processed in parallel → both CREATE_NEW

The LLM resolution logic was correct - when it sees candidates, it makes good MERGE decisions. The issue was entities being processed in parallel before the graph had data.

### Fix Applied
1. **Lowered SimilarityLockManager threshold** from 0.90 to **0.75**
   - Now entities with 75%+ similarity are processed serially
   - "Alphabet" and "Alphabet Inc." will be detected as conflicts
   - Second entity will wait and see the first in the graph → MERGE

### Files Modified
- `src/util/similarity_lock.py` line 56: `if sim > 0.75:`

### Acceptance Criteria
- [x] Similar entity names processed serially (not in parallel)
- [ ] New ingestions will deduplicate "Alphabet" / "Alphabet Inc." correctly
- [ ] Existing duplicates need manual merge or re-ingestion

---

## Issue 4: Missing Entities from Extraction (MEDIUM)

### Evidence
Q19: "What subsidiaries were merged back into Google?"
- Expected: Nest Labs, Chronicle Security, Sidewalk Labs
- Found: Only Sidewalk Labs has `INTEGRATED_SUBSIDIARY` relationship

### Investigation Results (2024-12-26)

**Source chunk EXISTS with all three subsidiaries:**
```
"Former subsidiaries include Nest Labs, which was merged into Google in February 2018,
and Chronicle Security, which was merged with Google Cloud in June 2019.
Sidewalk Labs was absorbed into Google in 2021..."
```

**Extraction status:**
| Entity | In Graph? | Relationship | Issue |
|--------|-----------|--------------|-------|
| Nest | ✅ Yes | ACQUIRED/DIVESTED | Wrong type (should be INTEGRATED_SUBSIDIARY) |
| Chronicle | ❌ No | - | Entity not extracted at all |
| Sidewalk Labs | ✅ Yes | INTEGRATED_SUBSIDIARY ✅ | Correct |

### Root Cause
This is an **extraction bug**, not a data gap:
1. **Chronicle Security** - Entity extractor completely missed this entity
2. **Nest Labs** - Extracted but with wrong relationship type (ACQUIRED instead of merged/integrated)

The atomizer or entity extractor is inconsistently handling "merged into" vs "absorbed into" language.

### Proposed Fix
Debug the atomizer/entity_extractor on this specific chunk to understand why:
1. Chronicle wasn't extracted as an entity
2. Nest got ACQUIRED instead of INTEGRATED_SUBSIDIARY

May need to add explicit handling for "merged into", "folded into" language patterns.

### Files to Check
- `src/chunker/SAVED/*.jsonl` - search for "Nest Labs", "Chronicle"
- Run extraction debug on relevant chunks

### Acceptance Criteria
- [ ] Determine if this is data gap or extraction bug
- [ ] If extraction bug, fix and re-run
- [ ] Q19 should list all three subsidiaries

---

## Issue 5: Free-form Topics (LOW) - **FIXED**

### Evidence
Graph contains topics like:
- `$1 Trillion Market Value`
- `$3 Trillion In Market Cap`
- `$7.5 Million`

These are really facts/events, not topics.

### Root Cause
1. `subject_type="Topic"` and `object_type="Topic"` bypassed TopicLibrarian validation
2. Fuzzy matching caused false positives (e.g., "CEO" → "CEO Confidence" → "Business Confidence")

### Fix Applied
1. **Added topic validation in pipeline** (`main_pipeline.py`):
   - Subject/object with type "Topic" now validated through TopicLibrarian
   - Invalid topics converted to "Entity" type to avoid garbage TopicNodes
   - Context (source fact) passed for better matching

2. **Rewrote TopicLibrarian** (`topic_librarian.py`):
   - Removed fuzzy search (caused false positives)
   - Vector search only (embeddings capture semantic meaning)
   - Added LLM verification with structured output
   - Context-aware matching using source fact

3. **Added "Market Valuation" topic** (`financial_topics.json`):
   - Captures valuation-related topics like "$1 Trillion Market Value"

### Files Modified
- `src/workflows/main_pipeline.py` - validate subject/object Topic types
- `src/agents/topic_librarian.py` - vector search + LLM verification
- `src/config/topics/financial_topics.json` - added Market Valuation

### Test Results
| Input | Context | Result |
|-------|---------|--------|
| `$1 Trillion Market Value` | Apple reached... | → `Market Valuation` |
| `$7.5 Million` | Equipment purchase... | → REJECTED |
| `CEO` | CEO announced... | → REJECTED |
| `Inflation` | Prices rising... | → `Inflation` |
| `M&A` | M&A activity... | → `Mergers And Acquisitions` |

### Acceptance Criteria
- [x] Topics are abstract concepts (Inflation, M&A, Corporate Restructuring)
- [x] Specific values ($1T, dates) stored as facts, not topics

---

## Issue 6: MCP Resolve Lacks Entity Rules (LOW)

### Evidence
`mcp_server.py:145`:
```python
if score < 0.7:
    continue
```

Just cosine similarity threshold. No semantic rules for Entity vs Topic distinction.

### Root Cause
`resolve_entity_or_topic` is purely vector-based. Doesn't understand entity types or apply the same rules as entity_extractor.

### Proposed Fix
- Add entity type filtering (if searching for Person, filter to person entities)
- Add name pattern matching for common entity types
- Consider separate indices for different entity types

### Files to Modify
- `src/agents/mcp_server.py` - `_resolve_entity_or_topic_logic()`

### Acceptance Criteria
- [ ] Searching for "CEO" prioritizes Person entities
- [ ] Searching for "subsidiary" prioritizes Organization entities

---

## Work Log

### Session 1: 2024-12-26
- [x] Analyzed eval results
- [x] Identified 6 core issues
- [x] Created this tracking document

### Session 2: 2024-12-26
- [x] **Fixed Issue 5: Free-form Topics**
  - Added topic validation for subject_type/object_type="Topic" in pipeline
  - Rewrote TopicLibrarian: removed fuzzy search, added LLM verification
  - Added "Market Valuation" topic to ontology
  - Re-indexed topic ontology (670 topics)

### Session 3: 2024-12-26
- [x] **Investigated Issue 3: Duplicate Entities**
  - Found 4 different "Alphabet" variants in graph (Alphabet, Alphabet Inc., Alphabet Inc, etc.)
  - Tested similarity search - candidates ARE being retrieved correctly
  - Tested LLM resolution - it makes correct MERGE decisions when shown candidates
  - **Root cause**: SimilarityLockManager threshold (0.90) too high for entity name variants
- [x] **Fixed Issue 3: Duplicate Entities**
  - Lowered SimilarityLockManager threshold from 0.90 to 0.75
  - File: `src/util/similarity_lock.py`
- [x] **Fixed Issue 2: Entity Summaries Capped**
  - Removed 1-sentence limit from entity summary prompts
  - Updated to request comprehensive 2-4 sentence summaries with key facts, dates, relationships
  - File: `src/agents/graph_enhancer.py`
- [x] **Fixed Issue 1: Agent Ignores Retrieved Chunks**
  - Added `think` tool to MCP server (based on Anthropic's think tool pattern)
  - Updated kg_agent.py system prompt to require think tool usage before answering
  - Agent must now explicitly extract facts from chunks and cite them
  - Files: `src/agents/mcp_server.py`, `src/agents/kg_agent.py`
- [ ] Issues 2 & 3 fixes require re-ingestion to see full effect on existing data
- [ ] Issue 1 fix needs evaluation re-run to verify improvement
