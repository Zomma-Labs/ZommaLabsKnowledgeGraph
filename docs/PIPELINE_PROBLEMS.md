# Pipeline Problems - Investigation Log

**Date**: 2025-12-25
**Current Accuracy**: 70% strict, 78% lenient
**Goal**: Identify root causes before implementing fixes

---

## Problem 1: Wrong Relationship Type Assignment

### Observation
The analyst/extractor assigns nonsensical relationship types that don't match the text.

### Example
**Source Text**:
```
Former executive Eric Schmidt revealed in the conference in 2o17 the inspiration
for this structure Hathaway was a holding company made of subsidiaries with
strong CEOs who were trusted to run their businesses.
```

**What was created**:
```
Berkshire Hathaway --[ESTABLISHED_SUBSIDIARY]--> Google
Berkshire Hathaway --[ESTABLISHED_SUBSIDIARY]--> Larry Page
Berkshire Hathaway --[ESTABLISHED_SUBSIDIARY]--> Sundar Pichai
```

**What should have been created**:
```
Alphabet Inc. --[INSPIRED_BY]--> Berkshire Hathaway
```

### Questions to Investigate
- [ ] Is this happening in entity_extractor.py or analyst.py?
- [ ] What relationship types are available in the taxonomy?
- [ ] Is `INSPIRED_BY` or `INSPIRED` even an option?
- [ ] Why did it pick `ESTABLISHED_SUBSIDIARY` for an inspiration relationship?
- [ ] Is the LLM seeing the word "subsidiaries" and matching on that?

---

## Problem 2: Relationship Direction Errors

### Observation
Subject and object are swapped, creating backwards relationships.

### Example
**Created**: `Berkshire Hathaway --[INSPIRED_BY]--> Google`
**Should be**: `Alphabet Inc. --[INSPIRED_BY]--> Berkshire Hathaway`

Also seen:
```
Eric Schmidt --[INSPIRED_BY]--> Larry Page  (makes no sense)
```

### Questions to Investigate
- [ ] Is there guidance in the prompt about relationship direction?
- [ ] Are there examples showing correct directionality?
- [ ] Is this a consistent pattern or random?

---

## Problem 3: Attribution Not Captured

### Observation
When someone reveals/discloses information, they're not linked to that fact.

### Example
**Source Text**: "Eric Schmidt revealed... the inspiration for this structure"

**What happened**:
- Eric Schmidt has relationships from chunk `2d29f1cc` (about 26 subsidiaries)
- Berkshire Hathaway has relationships from chunk `ea593cbe` (the inspiration chunk)
- Both are in the SAME chunk, but Eric Schmidt isn't linked to the Berkshire fact
- Fact_ids are different: Eric's facts vs Berkshire's facts

**Result**: When asked "who revealed the Berkshire inspiration?", agent can't connect Eric Schmidt to that disclosure.

### Questions to Investigate
- [ ] Why did Eric Schmidt get linked to a different chunk?
- [ ] Is there a relationship type for DISCLOSED/REVEALED?
- [ ] How does the atomizer split facts involving revelations?

---

## Problem 4: Missing Topics in Ontology

### Observation
Valid business/financial topics are being rejected because they're not in the ontology.

### Evidence (from rejected_topics.log)
| Topic | Times Rejected |
|-------|----------------|
| Layoffs | 25 |
| Robotics | 23 |
| COVID-19 | 19 |
| Quantum Computing | 16 |
| Autonomous Driving | 13 |
| Stock Split | 9 |
| Subsidiaries | 37 |

### Impact
- Facts about layoffs can't be found via topic search
- Quantum computing achievements (Willow chip) not linked to topic
- Agent says "not in KB" for valid questions

### Questions to Investigate
- [ ] What topics ARE in the current ontology?
- [ ] What's the threshold for topic matching?
- [ ] Should we auto-add frequently rejected topics?

---

## Problem 5: Retrieval Gaps

### Observation
Agent says "not in KB" for facts that exist in the chunks.

### Examples
| Question | Expected Answer | Agent Response | Fact in Chunk? |
|----------|-----------------|----------------|----------------|
| How many layoffs in 2023? | 12,000 (6%) | "Not in KB" | YES |
| R&D ranking in 2022? | 2nd highest | "Not in KB" | YES |
| Google+ settlement? | $7.5M | "Not stated" | YES |

### Current Retrieval Flow
```
1. resolve_entity_or_topic("layoffs") → No match (not a topic)
2. explore_neighbors("Alphabet Inc.") → Gets relationship types
3. No LAID_OFF relationship exists → Can't find chunk
4. Agent gives up
```

### Questions to Investigate
- [ ] What relationship types exist for these facts?
- [ ] Were entities/topics created for these facts during ingestion?
- [ ] Is there a fallback search mechanism?

---

## Problem 6: Duplicate/Nonsensical Edges from Single Fact

### Observation
One fact_id creates many edges, including nonsensical ones.

### Example
Fact_id `70c8015e` creates ALL of these:
```
Eric Schmidt --[PARTNERED]--> Google
Eric Schmidt --[PARTNERED]--> 26              ← number as entity?
Eric Schmidt --[PARTNERED]--> Astro Teller
Eric Schmidt --[PARTNERED]--> X Development
Eric Schmidt --[PARTNERED]--> Android
Eric Schmidt --[PARTNERED]--> Youtube
Eric Schmidt --[PARTNERED]--> Google Search
Eric Schmidt --[PARTNERED]--> Ceos Of The Current And Proposed Alphabet Subsidiaries
Eric Schmidt --[PARTNERED]--> Alphabet Subsidiaries
```

### Problems
1. `26` extracted as an entity (should be filtered)
2. `PARTNERED` makes no sense for most of these
3. One fact shouldn't create 15+ edges
4. "Ceos Of The Current And Proposed Alphabet Subsidiaries" is a phrase, not an entity

### Questions to Investigate
- [ ] Why is the number `26` being extracted as an entity?
- [ ] What's the source fact text that generated this?
- [ ] Is there validation to prevent phrases as entities?
- [ ] Why PARTNERED for all of these?

---

## Problem 7: Null Subjects in Relationships

### Observation
Some relationships have `None` as the subject.

### Example
```
None --[MENTIONED_IN]--> Google
None --[MENTIONED_IN]--> Google
None --[MENTIONED_IN]--> Google
... (repeated many times)
```

### Questions to Investigate
- [ ] Where in the pipeline do these get created?
- [ ] Is this a deduplication issue?
- [ ] Are these from failed entity resolution?

---

## Summary: Root Cause Hypotheses

| Problem | Likely Location | Hypothesis |
|---------|-----------------|------------|
| Wrong relationship types | analyst.py | Limited taxonomy, LLM keyword matching |
| Direction errors | entity_extractor.py | Prompt lacks direction examples |
| Attribution lost | atomizer.py | Facts split incorrectly |
| Missing topics | topic ontology | Ontology too narrow |
| Retrieval gaps | mcp_server.py | No fallback search |
| Duplicate edges | entity_extractor.py | Over-extraction, no validation |
| Null subjects | graph_assembler.py | Failed resolution not handled |

---

## Investigation Priority

1. **Problem 1 + 2**: Wrong relationships and direction (analyst + extractor)
2. **Problem 3**: Attribution (atomizer fact splitting)
3. **Problem 6**: Duplicate/nonsensical edges (extractor validation)
4. **Problem 4**: Missing topics (ontology expansion)
5. **Problem 5**: Retrieval gaps (MCP server)
6. **Problem 7**: Null subjects (assembler)

---

## Next Steps

- [ ] Read analyst.py to understand relationship taxonomy
- [ ] Read entity_extractor.py prompts for direction guidance
- [ ] Trace a single fact through the pipeline to see where it goes wrong
- [ ] Check topic ontology coverage
