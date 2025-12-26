# Problem 3: Attribution Not Captured

**Date**: 2025-12-25
**Status**: Investigating

---

## Problem Statement

When a person reveals, discloses, or announces information, their role as the source is not captured in the knowledge graph. The information itself gets extracted, but the attribution is lost.

---

## Example Case

### Source Text
```
Former executive Eric Schmidt revealed in the conference in 2017 the inspiration
for Alphabet's structure came from Berkshire Hathaway, a holding company made of
subsidiaries with strong CEOs.
```

### What Gets Extracted
```
Relation 1:
  Subject: Alphabet
  Object: Berkshire Hathaway
  Relationship: MODELED_AFTER
  Description: "structure modeled after"
```

### What's Missing
```
Relation 2 (NOT CREATED):
  Subject: Eric Schmidt
  Object: [the inspiration fact]
  Relationship: REVEALED / DISCLOSED
  Description: "revealed inspiration for"
```

### Impact
- Question: "Who revealed that Berkshire Hathaway inspired Alphabet's structure?"
- Expected: Eric Schmidt
- Agent Answer: Larry Page (wrong) or "Unknown"
- The graph has no edge connecting Eric Schmidt to this revelation

---

## Evidence from Graph

### Eric Schmidt Entity
```
Summary: "revealed that Berkshire Hathaway inspired Alphabet's structure"
         ✅ Summary is correct!
```

### Eric Schmidt Relationships
```
Eric Schmidt --[PARTNERED]--> Google (from different chunk)
Eric Schmidt --[ISSUED_GUIDANCE]--> ... (from different chunk)
Eric Schmidt --[INSPIRED_BY]--> Larry Page ❌ (wrong)
```

Eric Schmidt has NO relationship connecting him to the Berkshire/Alphabet inspiration fact.

---

## Root Cause Hypotheses

### Hypothesis A: Atomizer Splits Facts Incorrectly
The atomizer might be splitting the compound sentence into separate facts:
- Fact 1: "Alphabet's structure was inspired by Berkshire Hathaway"
- Fact 2: "Eric Schmidt revealed something in 2017"

If split this way, the connection between Eric Schmidt and WHAT he revealed is lost.

### Hypothesis B: Entity Extractor Creates One Relation Per Fact
The entity extractor might only create the "primary" relationship:
- Sees: Subject doing action to Object
- Extracts: Alphabet → Berkshire Hathaway
- Ignores: Eric Schmidt's role as the revealer

### Hypothesis C: No Pattern for "X revealed Y" Extraction
The extractor prompt doesn't have guidance for disclosure patterns:
- "X revealed Y" should create TWO relations:
  1. The content being revealed (Alphabet → Berkshire)
  2. The disclosure itself (Eric Schmidt → revealed → content)

---

## Questions to Investigate

1. **Atomizer Output**: What propositions does the atomizer create from this text?
   - Is Eric Schmidt preserved in the same proposition as Berkshire?
   - Or is it split into separate facts?

2. **Entity Extractor Behavior**: Given a fact with "X revealed Y":
   - Does it create multiple relations?
   - Or just the "Y" content relation?

3. **Relationship Types**: Do we have appropriate types?
   - REVEALED exists ✅
   - DISCLOSED exists ✅
   - But are they being used?

---

## Test Plan

### Step 1: Check Atomizer Output
Run the atomizer on the Eric Schmidt chunk and see what propositions it generates.

### Step 2: Check Entity Extractor Output
For each proposition, see what relations are extracted.

### Step 3: Trace the Full Pipeline
Follow the Eric Schmidt fact through the entire pipeline to see where attribution is lost.

---

## Potential Solutions (To Explore After Investigation)

1. **Atomizer Rule**: Keep revelations as single facts
   - "Eric Schmidt revealed X" should stay together, not split

2. **Entity Extractor Rule**: Create disclosure relations
   - When "X revealed/disclosed/announced Y", create relation for X's role

3. **Multi-Relation Extraction**: One fact → multiple relations
   - Primary: Content relation (Alphabet → Berkshire)
   - Secondary: Disclosure relation (Eric Schmidt → revealed)

---

## Success Criteria

After fixing, the test should produce:
```
Relation 1:
  Subject: Alphabet
  Object: Berkshire Hathaway
  Relationship: MODELED_AFTER

Relation 2:
  Subject: Eric Schmidt
  Object: Alphabet (or the fact itself)
  Relationship: REVEALED / DISCLOSED
```

Both Eric Schmidt AND the Alphabet/Berkshire relationship should be captured.
