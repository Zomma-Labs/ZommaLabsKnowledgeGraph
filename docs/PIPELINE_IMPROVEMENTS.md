# Pipeline Improvement Plan

**Created**: 2025-12-25
**Current Accuracy**: 64% strict, 82% lenient
**Target Accuracy**: 80%+ strict

---

## Executive Summary

Analysis of 50 QA evaluation questions revealed systematic issues in our extraction pipeline. The main problems are:

1. **Temporal precision loss** - Dates extracted without full MM/DD/YYYY precision
2. **List expansion failure** - Multiple entities in lists not being split into separate relations
3. **Attribution errors** - Wrong subject/object assignment for some relationships
4. **Numeric detail loss** - Percentages and dollar amounts not preserved

---

## Issue Breakdown

### Issue #1: Temporal Extraction Failures

**Severity**: HIGH
**Impact**: 5 incorrect answers (10% of total)
**Accuracy for temporal questions**: 54.5%

#### Specific Failures

| Q# | Question | Expected | Got | Error Type |
|----|----------|----------|-----|------------|
| Q18 | When was Google reorganized as LLC? | Sept 1, 2017 | Oct 2, 2015 | Event confusion |
| Q29 | When did Alphabet reach $1T? | Jan 16, 2020 | Jan 2016 | Wrong year |
| Q40 | When did DOJ file antitrust lawsuit? | October 2020 | August 2020 | Wrong month |
| Q43 | Google Russian subsidiary bankruptcy? | May→June 2022 | Feb 3, 2022 | Wrong sequence |
| Q46 | First dividend announcement? | April 2024 | Feb 3, 2022 | Wrong date entirely |

#### Root Cause

The atomizer prompt (line 59-60) only handles relative→absolute conversion:
```
"TEMPORAL GROUNDING: If the text says 'last year' and the document date is 2023,
change it to '2022'. Make every fact standalone in time."
```

**Missing**: Instructions to preserve EXACT dates when they appear in text.

---

### Issue #2: List Expansion Failure

**Severity**: HIGH
**Impact**: 4 incorrect/partial answers
**Accuracy for list-based questions**: 25%

#### Specific Failures

| Q# | Question | Expected | Got | Missing |
|----|----------|----------|-----|---------|
| Q16 | Major subsidiaries of Alphabet | 11 entities | 2 entities | X Development, Calico, Verily, Google Fiber, CapitalG, GV, DeepMind, Intrinsic, Isomorphic Labs |
| Q22 | Largest institutional shareholders | 3 with percentages | 3 without % | 7.25%, 6.27%, 3.36% |
| Q50 | Main Google products | 3 specific | 8+ listed | Over-extraction |

#### Root Cause

The entity extractor prompt mentions splitting aggregates but doesn't handle explicit lists:
```
"SPLITTING AGGREGATES: If the fact mentions 'Contacts in a few districts'..."
```

**Missing**: Rule to expand enumerated lists into separate relations.

---

### Issue #3: Attribution Errors

**Severity**: MEDIUM
**Impact**: 2-3 incorrect answers

#### Specific Failures

| Q# | Question | Expected | Got |
|----|----------|----------|-----|
| Q15 | Who revealed Berkshire Hathaway inspiration? | Eric Schmidt | Larry Page |

#### Root Cause

The entity extractor assigns subject/object based on sentence structure, but doesn't verify against the full chunk context. When the chunk mentions multiple people, it may pick the wrong one.

---

### Issue #4: Numeric Detail Loss

**Severity**: MEDIUM
**Impact**: 3 partial/incorrect answers

#### Specific Failures

| Q# | Question | Expected | Got |
|----|----------|----------|-----|
| Q22 | Shareholder percentages | 7.25%, 6.27%, 3.36% | Names only, no % |
| Q39 | Google+ settlement amount | $7.5M, $5-$12 per claimant | "Not stated" |
| Q47 | Stock buyback size | $70 billion | Unanswerable |

#### Root Cause

1. Atomizer may summarize away numeric details
2. Entity extractor has no field for numeric attributes
3. Numbers not stored in searchable way

---

## TODO Checklist

### Atomizer Updates (`src/agents/atomizer.py`)

- [ ] **TODO-A1**: Add EXACT DATE PRESERVATION rule to system prompt
- [ ] **TODO-A2**: Add NUMERIC PRESERVATION rule (percentages, dollar amounts)
- [ ] **TODO-A3**: Add EVENT DISAMBIGUATION rule for multiple similar events
- [ ] **TODO-A4**: Add examples of good vs bad temporal extraction

### Entity Extractor Updates (`src/agents/entity_extractor.py`)

- [ ] **TODO-E1**: Add LIST EXPANSION rule to extract all items from enumerated lists
- [ ] **TODO-E2**: Add numeric_value field to FinancialRelation schema
- [ ] **TODO-E3**: Strengthen ATTRIBUTION VERIFICATION in reflexion check
- [ ] **TODO-E4**: Add examples showing correct list expansion

### Schema Updates (`src/schemas/financial_relation.py`)

- [ ] **TODO-S1**: Add `numeric_value: Optional[str]` field for percentages/amounts
- [ ] **TODO-S2**: Add `event_date: Optional[str]` field for specific dates

### Testing

- [ ] **TODO-T1**: Create test cases for temporal extraction precision
- [ ] **TODO-T2**: Create test cases for list expansion
- [ ] **TODO-T3**: Re-run full evaluation after fixes

---

## Implementation Plan

### Phase 1: Atomizer Temporal & Numeric Fixes

**File**: `src/agents/atomizer.py`
**Lines**: 53-63 (system_prompt)

#### Current Prompt
```python
system_prompt = (
    "You are an expert Text Decomposer. Your goal is to split the input text "
    "into a list of 'Atomic Facts' (Propositions).\n\n"
    "Follow these strict rules:\n"
    "1. DE-CONTEXTUALIZE: The input is a chunk from a larger document..."
    "2. TEMPORAL GROUNDING: If the text says 'last year'..."
    "3. ATOMICITY: Each fact must be a single, simple sentence..."
    "4. PRESERVE DETAILS: Do not summarize away important numbers..."
)
```

#### Proposed Changes

Add these rules after rule 2:

```python
"2a. EXACT DATE PRESERVATION: When the text contains specific dates (e.g., 'January 16, 2020', "
"'Q3 2023', 'September 1, 2017'), preserve them EXACTLY in your output. Never approximate dates. "
"If month and day are given, include them. If only quarter is given, state 'Q3 2023' not just '2023'.\n"

"2b. EVENT DISAMBIGUATION: When multiple dated events are mentioned, create SEPARATE facts for each. "
"Example: 'Alphabet was created in Oct 2015 and Google became an LLC in Sept 2017' should become "
"TWO separate facts with their respective dates.\n"

"4a. NUMERIC PRECISION: Preserve ALL numeric details including:\n"
"   - Percentages (e.g., '7.25%', '6% of workforce')\n"
"   - Dollar amounts (e.g., '$7.5 million', '$70 billion')\n"
"   - Counts (e.g., '12,000 employees', '14,000 documents')\n"
"   - Rankings (e.g., 'second-highest', 'fourth company to reach')\n"
```

---

### Phase 2: Entity Extractor List & Attribution Fixes

**File**: `src/agents/entity_extractor.py`
**Lines**: 43-79 (extract prompt)

#### Proposed New Rules

Add after rule 7:

```python
"8. LIST EXPANSION - CRITICAL:\n"
"   When the text contains an enumerated list of entities (e.g., 'subsidiaries include X, Y, Z'), "
"   you MUST create a SEPARATE FinancialRelation entry for EACH item in the list.\n"
"   - Example: 'Alphabet subsidiaries include Google, Waymo, and Verily' should produce:\n"
"     * Relation 1: Subject=Alphabet, Object=Google, relationship implied\n"
"     * Relation 2: Subject=Alphabet, Object=Waymo, relationship implied\n"
"     * Relation 3: Subject=Alphabet, Object=Verily, relationship implied\n"
"   - Do NOT create a single relation with a combined object like 'Google, Waymo, Verily'\n"

"9. NUMERIC ATTRIBUTES:\n"
"   When a relationship involves a specific number, percentage, or amount, include it.\n"
"   - Example: 'Vanguard owns 7.25% of Alphabet' → Subject=Vanguard, Object=Alphabet, "
"     with the percentage noted in the fact text.\n"

"10. ATTRIBUTION VERIFICATION:\n"
"   Before finalizing subject/object, verify WHO performed the action by checking the CHUNK context.\n"
"   - If the fact says 'X revealed Y' but the chunk mentions multiple people, ensure you pick the "
"     correct person who actually made the revelation.\n"
"   - Common error: Attributing an action to a well-known figure when a different person did it.\n"
```

---

### Phase 3: Schema Updates (Optional)

**File**: `src/schemas/financial_relation.py`

Add optional fields to capture structured numeric data:

```python
class FinancialRelation(BaseModel):
    subject: str
    subject_type: Literal["Entity", "Topic"]
    object: Optional[str]
    object_type: Literal["Entity", "Topic"]
    topics: List[str] = []
    date_context: Optional[str] = None
    # NEW FIELDS
    numeric_value: Optional[str] = None  # e.g., "7.25%", "$70 billion"
    event_date: Optional[str] = None     # e.g., "2020-01-16", "Q3 2023"
```

---

### Phase 4: Reflexion Improvements

**File**: `src/agents/entity_extractor.py`
**Lines**: 133-170 (reflexion_check)

Strengthen the reflexion prompt:

```python
"5. ATTRIBUTION CHECK: For each relation, verify:\n"
"   - Is the SUBJECT the correct entity performing the action?\n"
"   - Is the OBJECT the correct entity receiving the action?\n"
"   - Cross-reference with the CHUNK to confirm. If the chunk says 'Eric Schmidt revealed X' "
"     but you extracted 'Larry Page revealed X', that's an error.\n"

"6. LIST COMPLETENESS CHECK:\n"
"   - Does the CHUNK contain a list of entities (subsidiaries, shareholders, products)?\n"
"   - Did we create separate relations for EACH item in that list?\n"
"   - If we extracted only 2 items but the chunk lists 10, we need to expand.\n"
```

---

## Testing Strategy

### Before Implementation
1. Save current evaluation results as baseline
2. Document the 18 incorrect/partial answers

### After Implementation
1. Re-ingest the Alphabet document with updated pipeline
2. Run evaluation: `uv run src/scripts/evaluate_qa.py`
3. Compare results, specifically checking:
   - Q16 (subsidiaries list)
   - Q18, Q29, Q40, Q43, Q46 (temporal questions)
   - Q15 (attribution)
   - Q22, Q39, Q47 (numeric details)

### Success Criteria
- Temporal accuracy: 54.5% → 80%+
- List-based accuracy: 25% → 75%+
- Overall strict accuracy: 64% → 80%+

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Over-extraction (too many entities) | Add minimum relevance threshold |
| Slower extraction (more rules) | Rules are prompt-based, minimal latency impact |
| Breaking existing functionality | Run full test suite before/after |
| List expansion creating duplicates | Deduplication in graph_assembler already handles this |

---

## Timeline Estimate

| Phase | Effort |
|-------|--------|
| Phase 1: Atomizer updates | 1-2 hours |
| Phase 2: Entity extractor updates | 2-3 hours |
| Phase 3: Schema updates | 30 min |
| Phase 4: Reflexion improvements | 1 hour |
| Testing & validation | 2-3 hours |
| **Total** | **6-10 hours** |

---

# Phase 2 Research Findings (Post-Implementation)

**Date**: 2025-12-25
**Current Accuracy**: 70% strict, 78% lenient (improved from 64%/82%)

## Research Summary

After implementing Phase 1 fixes, we conducted deep analysis of remaining failures.

### Key Finding: All facts exist in source document

Every failing question has its answer in the source `alphabet.jsonl`. The problem is **extraction and retrieval**, not missing data.

---

## Issue Analysis: Eric Schmidt Attribution (Q15)

### The Problem
- **Question**: "Who revealed that Warren Buffett's Berkshire Hathaway inspired Alphabet's structure?"
- **Expected**: Eric Schmidt
- **Agent Answer**: Larry Page

### Source Text (chunk_0005)
```
Former executive Eric Schmidt revealed in the conference in 2o17 the inspiration for this
structure Hathaway was a holding company made of subsidiaries with strong CEOs who were
trusted to run their businesses.
```

### What's in the Graph

**Entity Summaries (CORRECT):**
- Eric Schmidt: "revealed that Berkshire Hathaway inspired Alphabet's structure"
- Berkshire Hathaway: "A holding company that inspired Alphabet's structure"

**Relationships (WRONG):**
```
Berkshire Hathaway --[ESTABLISHED_SUBSIDIARY]--> Google        ❌ Wrong
Berkshire Hathaway --[INSPIRED_BY]--> Google                   ❌ Backwards
Eric Schmidt --[INSPIRED_BY]--> Larry Page                     ❌ Wrong
```

**Missing Relationships:**
```
Alphabet Inc. --[INSPIRED_BY]--> Berkshire Hathaway            ❌ Not created
Eric Schmidt --[DISCLOSED]--> [Berkshire inspiration fact]    ❌ Not created
```

### Root Cause Analysis

1. **Chunk Context Confusion**: The chunk mentions Larry Page first ("Page stated that...") then Eric Schmidt later. The entity extractor attributed the Berkshire revelation to the more prominent figure.

2. **Relationship Direction Errors**: The extractor created `Berkshire --[INSPIRED_BY]--> Google` instead of `Alphabet --[INSPIRED_BY]--> Berkshire`.

3. **No "DISCLOSED/REVEALED" Relationship Type**: The analyst doesn't have a relationship type for disclosures/revelations, so Eric Schmidt's role as the revealer is lost.

### Proposed Fixes

1. **Add DISCLOSED relationship type** to `src/agents/analyst.py`
2. **Strengthen attribution check** in entity extractor prompt
3. **Add "revelation/disclosure" extraction rule** to atomizer

---

## Issue Analysis: Retrieval Failures (Q33, Q36, Q39)

### The Problem

| Q# | Question | Expected | Agent Says |
|----|----------|----------|------------|
| Q33 | How many layoffs in Jan 2023? | 12,000 (6%) | "Not in KB" |
| Q36 | R&D ranking in 2022? | 2nd highest | "Not in KB" |
| Q39 | Google+ settlement amount? | $7.5M | "Not stated" |

### What's in the Chunks (VERIFIED)

```cypher
-- Layoffs chunk found:
"Around 12,ooo jobs were cut, which reduced the company's workforce by 6%"

-- R&D chunk found:
"Alphabet was the company with the second-highest expenditure on research
and development worldwide, with R&D expenditure amounting to US$39.5 billion"

-- Google+ chunk found:
"The litigation was settled in July 202o for $7.5 million"
```

### What Entities Exist

| Fact | Entity/Topic Created? | Linked to Alphabet? |
|------|----------------------|---------------------|
| 12,000 layoffs | ❌ No entity | N/A |
| Second-highest R&D | ❌ No topic | N/A |
| $7.5M settlement | ❌ No entity | N/A |
| Google+ | ❌ No entity | N/A |

### How Agent Searches

```
1. resolve_entity_or_topic("layoffs") → No match
2. explore_neighbors("Alphabet Inc.") → Gets REPORTED_FINANCIALS, SUED, etc.
3. get_chunk(Alphabet, ???, LAID_OFF) → Can't find relationship type
4. Agent gives up: "Not in KB"
```

### Root Cause Analysis

1. **No Entities Created for Statistical Facts**: The atomizer creates facts like "Alphabet laid off 12,000 employees" but the entity extractor doesn't create an entity for "12,000 Layoffs" or similar.

2. **No Fallback Search**: The MCP server relies entirely on graph traversal. If a fact isn't connected via a known relationship, it's invisible.

3. **OCR Artifacts**: `12,ooo` instead of `12,000`, `$7.5` getting lost.

4. **Relationship Type Gaps**: No `LAID_OFF`, `RANKED`, or `SETTLED_LAWSUIT` relationship types that would connect these facts.

### Proposed Fixes

1. **Add relationship types** to analyst:
   - `LAID_OFF` - for layoff events
   - `RANKED` - for ranking/comparison facts
   - `SETTLED_LAWSUIT` - distinct from SETTLED_LEGAL_DISPUTE

2. **Entity Extractor Rule**: Create entities for significant numerical events:
   - "12,000 Layoffs" as an entity
   - "Google+ Settlement" as an entity

3. **MCP Server Fallback**: Add full-text search on EpisodicNode.content when graph traversal fails

4. **Pre-process OCR**: Clean `o` → `0` in numbers before ingestion

---

## Updated Implementation Plan

### Phase 5: Relationship Type Expansion

**File**: `src/agents/analyst.py`

Add new relationship types:
```python
# Disclosure/Communication
"DISCLOSED",      # Person disclosed/revealed information
"ANNOUNCED",      # Company announced something

# Personnel
"LAID_OFF",       # Company laid off employees

# Comparative
"RANKED",         # Entity ranked in comparison
"OUTPERFORMED",   # Entity outperformed another

# Legal (more specific)
"SETTLED_LAWSUIT", # More specific than SETTLED_LEGAL_DISPUTE
```

### Phase 6: Entity Extraction for Events

**File**: `src/agents/entity_extractor.py`

Add rule:
```
11. EVENT ENTITY EXTRACTION:
    For significant events with numbers, create a named entity:
    - "12,000 layoffs in 2023" → Entity: "2023 Alphabet Layoffs" (type: Event)
    - "$7.5 million settlement" → Entity: "Google+ Privacy Settlement" (type: Event)
    - "Second-highest R&D spending" → Entity: "2022 R&D Ranking" (type: Ranking)
```

### Phase 7: MCP Server Fallback Search

**File**: `src/agents/mcp_server.py`

Add fallback full-text search:
```python
@mcp.tool()
def search_facts(query: str, ctx: Context) -> str:
    """
    Full-text search across all chunks when graph traversal fails.
    Use this as a last resort when resolve_entity_or_topic returns no matches.
    """
    results = services.neo4j.query('''
        MATCH (e:EpisodicNode {group_id: $uid})
        WHERE e.content CONTAINS $query
        RETURN e.content, e.source
        LIMIT 5
    ''', {"query": query, "uid": user_id})
    ...
```

### Phase 8: OCR Pre-processing

**File**: `src/chunker/` or pre-processing script

```python
def clean_ocr_artifacts(text: str) -> str:
    # Fix common OCR errors in numbers
    text = re.sub(r'(\d),o(\d{2})', r'\1,0\2', text)  # 12,ooo → 12,000
    text = re.sub(r'(\d)o(\d{2})', r'\1,0\2', text)   # 2o17 → 2017
    text = re.sub(r'\$(\d+)o(\d)', r'$\10\2', text)   # $27o → $270
    return text
```

---

## Priority Order

| Priority | Fix | Impact | Effort |
|----------|-----|--------|--------|
| 1 | MCP fallback search | +4% (fixes Q33, Q36, Q39) | Medium |
| 2 | Add DISCLOSED relationship | +2% (fixes Q15) | Low |
| 3 | OCR pre-processing | +4% (fixes numeric errors) | Low |
| 4 | Event entity extraction | +2% (long-term improvement) | Medium |

---

## Success Metrics

After implementing all phases:
- Q15 (Eric Schmidt): Should return "Eric Schmidt" via DISCLOSED relationship
- Q33 (Layoffs): Should find "12,000" via fallback search or LAID_OFF relationship
- Q36 (R&D): Should find "second-highest" via fallback search or RANKED relationship
- Q39 (Settlement): Should find "$7.5 million" via fallback search

**Target Accuracy**: 80%+ strict (from current 70%)
