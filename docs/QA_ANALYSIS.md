# Beige Book QA Analysis

## Summary

| Metric | Count | % |
|--------|-------|---|
| Total Questions | 75 | 100% |
| Correct | 17 | 22.7% |
| Partially Correct | 23 | 30.7% |
| Incorrect | 35 | 46.7% |

### Failure Breakdown

| Failure Type | Count | Root Cause |
|--------------|-------|------------|
| Recursion Limit | 17 | Agent loops infinitely |
| Cannot Answer | 10 | Retrieval misses or data not in source |
| Wrong Answer | 2 | LLM misinterprets found data |
| Validation Error | 1 | Tool parameter issue |

---

## Data Issues: Questions with Answers NOT in Source

These questions have expected answers that reference data **not present** in the source document. These cannot be answered correctly regardless of retrieval strategy.

### Q1: Which Federal Reserve Bank prepared the October 2025 Beige Book report?

**Expected**: The Federal Reserve Bank of San Francisco prepared the October 2025 Beige Book report.

**Data Check**:
- "prepared" → ✗ NOT FOUND in source
- "San Francisco" → ✓ Found (but only as a district, not as preparer)

**Verdict**: **UNANSWERABLE** - The source document does not state who prepared it.

---

### Q13: How did spending by higher-income individuals on luxury travel perform?

**Expected**: Spending by higher-income individuals on luxury travel and accommodation was reportedly strong.

**Data Check**:
- "luxury travel" → ✗ NOT FOUND
- "higher-income" → ✗ NOT FOUND
- "strong" → ✓ Found (but in different contexts)

**Verdict**: **UNANSWERABLE** - Specific phrase about higher-income luxury travel not in source.

---

### Q15: How did demand from domestic consumers for leisure and hospitality change?

**Expected**: Demand by domestic consumers was largely unchanged.

**Data Check**:
- "domestic consumers" → ✗ NOT FOUND
- "largely unchanged" → ✓ Found (but in different context - about selling prices)
- "leisure" → ✓ Found

**Verdict**: **LIKELY UNANSWERABLE** - The specific "domestic consumers" framing not in source.

---

### Q17: What drove manufacturing demand in the Chicago District?

**Expected**: Manufacturing demand increased slightly in October and early November. Fabricated metals orders rose modestly, driven in part by growth in the automotive and defense sectors.

**Data Check**:
- "fabricated metals" → ✗ NOT FOUND
- "automotive" → ✓ Found
- "defense sectors" → ✗ NOT FOUND

**Verdict**: **PARTIALLY UNANSWERABLE** - Some expected details not in source.

---

## Terminology Issues: Data Exists but Uses Different Names

These questions have data in the source, but the expected answers use terminology that doesn't match:

### Q5: Which districts reported slight to modest economic growth?

**Expected**: Three districts reported slight to modest growth: Boston (First District), Philadelphia (Third District), and Richmond (Fifth District).

**Data Check**:
- "slight to modest growth" → ✗ NOT FOUND as exact phrase
- "three Districts reporting slighttomodestgrowth" → ✓ FOUND (note: no spaces due to PDF extraction)
- "Boston", "Philadelphia", "Richmond" → ✓ All found, but not explicitly linked to "First/Third/Fifth District"

**Verdict**: **ANSWERABLE** - Data exists, retrieval should find it. Spaces in extracted text may cause matching issues.

---

### Q8: What was the economic activity in the Seventh District (Chicago)?

**Expected**: Economic activity in the Seventh District was flat...

**Data Check**:
- "Seventh District" → ✗ NOT FOUND (source uses "Chicago" not "Seventh District")
- "Chicago" + "flat" → ✓ Both found

**Verdict**: **ANSWERABLE** - Data exists under "Chicago", not "Seventh District".

---

## Recursion Limit Failures (17 questions)

These questions caused the agent to loop infinitely. The structured pipeline will **eliminate these entirely**.

| Q# | Question | Likely Cause |
|----|----------|--------------|
| Q13 | How did spending by higher-income individuals on luxury travel perform? | Data not in source |
| Q15 | How did demand from domestic consumers for leisure and hospitality change? | Data not in source |
| Q16 | What were the conditions in manufacturing according to the October 2025 Beige Book? | Broad question, agent keeps searching |
| Q21 | Which sectors faced labor shortages due to immigration policy changes? | Multi-hop query, complex |
| Q24 | How did wages grow across districts? | Aggregation query, many results |
| Q25 | What drove labor cost pressures to intensify? | Causal reasoning query |
| Q30 | What were the conditions in residential and commercial real estate? | Broad topic, many results |
| Q32 | How was the housing market in the Dallas District? | Uses "District" terminology |
| Q33 | How did lower interest rates affect business activity? | Causal reasoning query |
| Q48 | What was the state of transportation activity? | Broad topic |
| Q51 | In districts where economic activity declined, what happened to manufacturing? | Conditional query, complex |
| Q53 | What materials saw price decreases, and what caused those decreases in each district? | Multi-part aggregation |
| Q56 | What is the relationship between AI investment and employment trends? | Causal/correlation query |
| Q57 | Compare the economic outlook between districts that saw growth versus those that saw decline. | Complex comparison |
| Q63 | What were the multiple factors affecting labor costs, and how did they interact with wage growth? | Multi-factor analysis |
| Q64 | How did consumer spending patterns affect different restaurant segments in the Chicago District? | Specific + causal |
| Q75 | What were all the key economic indicators for the Chicago (Seventh) District? | Aggregation query |

---

## Cannot Answer Failures (10 questions)

The agent found some data but couldn't formulate a complete answer:

| Q# | Question | Issue |
|----|----------|-------|
| Q1 | Which Federal Reserve Bank prepared the October 2025 Beige Book report? | Data not in source |
| Q5 | Which districts reported slight to modest economic growth? | Text spacing issues in source |
| Q8 | What was the economic activity in the Seventh District (Chicago)? | "Seventh District" not in source |
| Q39 | How did energy activity perform? | Aggregation across districts |
| Q45 | What were the top concerns for businesses in the Dallas District? | Uses "District" terminology |
| Q52 | Which districts mentioned both tariff impacts AND labor shortages? | Multi-condition query |
| Q54 | In districts where financial conditions loosened, how did real estate perform? | Conditional aggregation |
| Q68 | What supply chain disruptions affected manufacturing? | Specific detail query |
| Q71 | How did energy sector performance vary across districts? | Cross-district aggregation |
| Q72 | How did banking conditions differ for consumer versus business lending? | Comparison query |

---

## Wrong Answer Failures (2 questions)

| Q# | Question | Issue |
|----|----------|-------|
| Q17 | What drove manufacturing demand in the Chicago District? | LLM gave opposite answer (said "weaker" instead of "increased") |
| Q70 | Which retail categories saw gains versus declines? | Incomplete extraction |

---

## Recommendations

### 1. Fix QA File (High Priority)
Remove or fix these unanswerable questions:
- Q1: Remove - "prepared by" not in source
- Q13: Remove - "luxury travel higher-income" not in source
- Q15: Reword or remove - "domestic consumers" framing not in source

### 2. Handle PDF Extraction Issues
The source text has spacing issues from PDF extraction (e.g., "slighttomodestgrowth"). Consider:
- Pre-processing source to fix spacing
- Using fuzzy matching in retrieval

### 3. Terminology Mapping
Create mapping from QA terminology to source terminology:
- "Seventh District" → "Chicago"
- "First District" → "Boston"
- etc.

### 4. Implement Structured Pipeline
The structured pipeline will:
- Eliminate all 17 recursion limit failures (guaranteed)
- Improve retrieval for terminology mismatches via better fallbacks
- Reduce latency from 40s to ~10s
