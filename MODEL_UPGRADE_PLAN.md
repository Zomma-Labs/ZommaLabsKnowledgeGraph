# Model Upgrade Plan

## Problem Summary

Evaluation revealed 22% incorrect answers (11/50), with many failures caused by **inverted relationship directions** in entity extraction. Example:

- **Input**: "Nest Labs was merged into Google"
- **Expected**: `Google -> merged -> Nest Labs` (Google is the acquirer)
- **Actual**: `Nest Labs -> merged -> Google` (Wrong - implies Nest acquired Google)

## Root Cause Analysis

| Model | Semantic Role Understanding | Score |
|-------|----------------------------|-------|
| gemini-2.5-flash-lite | Cannot reason about passive voice | 0/3 |
| gemini-2.5-flash | Understands with conceptual prompt | 3/3 |

The lite model fundamentally cannot understand:
- Passive voice constructions ("X was done by Y")
- Semantic roles (agent vs patient)
- That word order ≠ semantic direction

## Changes Required

### 1. Entity Extractor (`src/agents/entity_extractor.py`)

**Current**: `gemini-2.5-flash-lite`
**Change to**: `gemini-2.5-flash`

**Prompt Addition** - Add semantic roles section:
```
=== CRITICAL: SEMANTIC ROLE ASSIGNMENT ===

Subject and Object must be assigned based on SEMANTIC ROLES, not word order:

- SUBJECT = the AGENT (who performs, initiates, or causes the action)
- OBJECT = the PATIENT (who receives, undergoes, or is affected by the action)

PASSIVE VOICE WARNING:
In passive constructions, the grammatical subject is often the semantic PATIENT.
You must identify the true AGENT regardless of word order.

Ask yourself: 'Who is DOING the action to whom?'
The doer = Subject. The receiver = Object.
```

### 2. Atomizer (`src/agents/atomizer.py`)

**Status**: ✅ NO CHANGE NEEDED

The atomizer works well with `gemini-2.5-flash-lite`. In fact, upgrading it actually makes results worse (see below).

## A/B Test Results

### Full Pipeline Test (Atomizer → Entity Extractor)

| Config | Atomizer | Extractor | Correct | Wrong | Verdict |
|--------|----------|-----------|---------|-------|---------|
| A (Current) | lite | lite | 0 | 3 | ❌ Broken |
| **B (Recommended)** | **lite** | **flash** | **3** | **0** | ✅ **Best** |
| C | flash | lite | 0 | 5 | ❌ Worse |
| D | flash | flash | 2 | 1 | ⚠️ Degraded |

### Key Findings

1. **Entity Extractor is the bottleneck** - Only upgrading extractor (Config B) achieves 100% accuracy
2. **Atomizer upgrade is counterproductive** - Flash atomizer over-splits facts (8 vs 4), creating more opportunities for extraction errors
3. **Lite atomizer produces better-formed facts** - Keeps related info together (e.g., "which was merged" stays with entity)

### Facts Comparison

**Lite Atomizer (4 facts - cleaner):**
```
1. Nest Labs was a former subsidiary which was merged into Google in February 2018.
2. Chronicle Security was a former subsidiary which was merged with Google Cloud in June 2019.
3. Sidewalk Labs was a former subsidiary which was absorbed into Google in 2021.
4. Daniel L. Doctoroff departed from Sidewalk Labs in 2021 due to a suspected ALS diagnosis.
```

**Flash Atomizer (8 facts - over-split):**
```
1. Nest Labs was a former subsidiary.
2. Nest Labs was merged into Google in February 2018.
3. Chronicle Security was a former subsidiary.
4. Chronicle Security was merged with Google Cloud in June 2019.
... (etc)
```

The lite atomizer's combined facts ("subsidiary which was merged") give the extractor better context.

## Cost Impact

| Component | Current Model | New Model | Cost Multiplier |
|-----------|--------------|-----------|-----------------|
| Atomizer | flash-lite | **flash-lite** (no change) | 1x |
| Entity Extractor | flash-lite | **flash** | ~2-3x |

**Overall pipeline cost increase**: ~1.5-2x (only extractor upgraded)

**ROI Analysis**:
- Current: 0/3 correct directions on merger facts = **0% accuracy**
- After upgrade: 3/3 correct directions = **100% accuracy**
- Even at 2x cost, getting correct data vs garbage is worth it

## Implementation Checklist

- [x] Test atomizer with flash vs flash-lite → **Result: Keep lite**
- [x] Test entity extractor with flash vs flash-lite → **Result: Upgrade to flash**
- [x] Update `entity_extractor.py`:
  - [x] Change model from `gemini-2.5-flash-lite` to `gemini-2.5-flash`
  - [x] Add semantic roles prompt section to existing prompt
- [ ] Re-run QA evaluation to measure improvement
- [ ] Re-ingest problematic chunks to fix existing bad data

## Verification Test (Post-Implementation)

```
FACT: Nest Labs was merged into Google in February 2018.
  Google -> merged with -> Nest Labs  ✓ CORRECT

FACT: Chronicle Security was merged with Google Cloud in June 2019.
  Google Cloud -> merged with -> Chronicle Security  ✓ CORRECT

FACT: Sidewalk Labs was absorbed into Google in 2021.
  Google -> absorbed -> Sidewalk Labs  ✓ CORRECT
```

**Result: 3/3 correct (was 0/3 before)**

## Files to Modify

### `src/agents/entity_extractor.py`

Line 34 - Change model:
```python
# Before
self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

# After
self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
```

Add to prompt (after line 105, before Section A):
```python
f"=== CRITICAL: SEMANTIC ROLE ASSIGNMENT ===\n\n"
f"Subject and Object must be assigned based on SEMANTIC ROLES, not word order:\n\n"
f"- SUBJECT = the AGENT (who performs, initiates, or causes the action)\n"
f"- OBJECT = the PATIENT (who receives, undergoes, or is affected by the action)\n\n"
f"PASSIVE VOICE WARNING:\n"
f"In passive constructions, the grammatical subject is often the semantic PATIENT.\n"
f"You must identify the true AGENT regardless of word order.\n\n"
f"Ask yourself: 'Who is DOING the action to whom?'\n"
f"The doer = Subject. The receiver = Object.\n\n"
```
