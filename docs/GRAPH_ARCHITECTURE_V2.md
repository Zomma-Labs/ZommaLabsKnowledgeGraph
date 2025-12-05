# Graph Architecture V2: Chunk-Centric Design

## Overview

This document describes a redesign of the ZommaLabsKG graph schema, moving from a **Fact-as-Hub** model to a **Chunk-as-Hub** model. The core idea is that the **EpisodicNode (chunk)** becomes the central connection point for entity relationships, with typed edges encoding semantic relationships.

---

## Design Goals

1. **Provenance-First**: Every relationship goes through the source chunk—you can never have a relationship without evidence.
2. **Simpler Entity Queries**: Direct typed edges (`:SUED`, `:ACQUIRED`) replace generic `:PERFORMED`/`:TARGET` patterns.
3. **Entity Deduplication**: EntityNodes are resolved and merged globally. One entity, many edge connections to different chunks.
4. **Reduced Node Types**: Remove `SectionNode`, move its `header_path` property to `EpisodicNode`.

---

## Node Schema

### Core Nodes

| Node Type | Purpose | Key Properties |
|-----------|---------|----------------|
| `DocumentNode` | Parent container for all chunks from a source file | `name`, `file_type`, `document_date`, `metadata`, `group_id` |
| `EpisodicNode` | **The Hub** — A chunk of text from a document | `content`, `header_path`, `source`, `valid_at`, `group_id` |
| `EntityNode` | A deduplicated real-world entity (Person, Org, Place, Concept) | `name`, `summary`, `embedding`, `group_id` |
| `FactNode` | An atomic fact extracted from text (for causation tracking) | `content`, `fact_type`, `embedding`, `group_id` |
| `TopicNode` | A global theme/topic | `name`, `fibo_id`, `embedding`, `group_id` |

### Removed Nodes

| Node Type | Reason |
|-----------|--------|
| `SectionNode` | Merged into `EpisodicNode.header_path` |

---

## Edge Schema

### Document Structure

```
(DocumentNode) ─[:HAS_CHUNK]─▶ (EpisodicNode)
```

### Topic Classification

```
(TopicNode) ─[:ABOUT]─▶ (EpisodicNode)
```

The TopicNode connects to chunks that discuss that topic.

### Entity-Chunk Relationships (The Core Change)

Instead of:
```
(Entity) ─[:PERFORMED]─▶ (FactNode) ─[:TARGET]─▶ (Entity)
                              │
                        [:MENTIONED_IN]
                              ▼
                        (EpisodicNode)
```

We now have:
```
(EntityNode) ─[:RELATIONSHIP_TYPE]─▶ (EpisodicNode) ─[:RELATIONSHIP_TYPE_PASSIVE]─▶ (EntityNode)
   Subject           Active Edge           Chunk             Passive Edge             Object
```

#### Example: "Apple sued Intel"

```
(EntityNode:Apple) ─[:SUED]─▶ (EpisodicNode:chunk_123) ─[:GOT_SUED]─▶ (EntityNode:Intel)
```

#### Example: "Federal Reserve raised policy rates affecting inflation"

```
(EntityNode:Federal Reserve) ─[:RAISED_POLICY_RATE]─▶ (EpisodicNode:chunk_456) ─[:AFFECTED_BY_RATE_RAISE]─▶ (EntityNode:Inflation)
```

### Fact Provenance & Causation

```
(FactNode) ─[:MENTIONED_IN]─▶ (EpisodicNode)
(FactNode) ─[:CAUSES]─▶ (FactNode)
```

FactNodes are still created for atomic fact text storage and causal chaining.

---

## Relationship Types

### Active Edges (Subject → Chunk)

These are the existing `RelationshipType` enum values, used when the subject PERFORMS the action:

```python
# Corporate Actions
ACQUIRED, SUED, PARTNERED, INVESTED, DIVESTED, HIRED, LAUNCHED_PRODUCT, EXPANDED, CLOSED

# Financial
REPORTED_FINANCIALS, ISSUED_GUIDANCE, DECLARED_DIVIDEND, AUTHORIZED_BUYBACK, ISSUED_DEBT, DEFAULTED, FILED_BANKRUPTCY

# Regulatory / Legal
REGULATED, SETTLED_LEGAL_DISPUTE, GRANTED_PATENT, RECALLED_PRODUCT, EXPERIENCED_DATA_BREACH

# Executive / Analyst
EXECUTIVE_RESIGNATION, ANALYST_RATING_CHANGE

# Macro / Economic
RAISED_POLICY_RATE, LOWERED_POLICY_RATE, REPORTED_ECONOMIC_DATA, IMPOSED_SANCTIONS

# Subsidiary / Structure
ESTABLISHED_SUBSIDIARY, INTEGRATED_SUBSIDIARY, SPUN_OFF

# Beige Book Signals
REPORTED_WAGE_PRESSURE_EASING, REPORTED_WAGE_PRESSURE_RISING, REPORTED_LABOR_MARKET_SOFTENING, ...

# Causation (between FactNodes, not through chunks)
CAUSED, EFFECTED_BY, CONTRIBUTED_TO, PREVENTED
```

### Passive Edges (Chunk → Object)

For each active edge, we define a passive/inverse edge:

| Active (Subject → Chunk) | Passive (Chunk → Object) |
|--------------------------|--------------------------|
| `ACQUIRED` | `GOT_ACQUIRED` |
| `SUED` | `GOT_SUED` |
| `INVESTED` | `RECEIVED_INVESTMENT` |
| `REGULATED` | `GOT_REGULATED` |
| `RAISED_POLICY_RATE` | `AFFECTED_BY_RATE_RAISE` |
| ... | ... |

> **Note**: The full passive edge mapping needs to be defined for all relationship types.

---

## Visual Architecture

```
                          ┌─────────────────────┐
                          │    DocumentNode     │
                          │  "beigebook_2024"   │
                          └──────────┬──────────┘
                                     │ :HAS_CHUNK
                                     ▼
┌─────────────┐                ┌─────────────────────┐                ┌─────────────┐
│  TopicNode  │───[:ABOUT]────▶│    EpisodicNode     │◀───[:ABOUT]───│  TopicNode  │
│ "Inflation" │                │   header_path:      │                │  "Labor"    │
└─────────────┘                │   "NY > Economy"    │                └─────────────┘
                               │   content: "..."    │
                               └─────────────────────┘
                                   ▲           │
                                   │           │
                         :SUED     │           │   :GOT_SUED
                                   │           ▼
                            ┌──────┴─────┐  ┌─────────────┐
                            │ EntityNode │  │ EntityNode  │
                            │  (Apple)   │  │  (Intel)    │
                            │  SUBJECT   │  │   OBJECT    │
                            └────────────┘  └─────────────┘

                               ┌─────────────┐
                               │  FactNode   │
                               │ "Apple sued │───[:MENTIONED_IN]───▶ EpisodicNode
                               │   Intel"    │
                               └──────┬──────┘
                                      │ :CAUSES
                                      ▼
                               ┌─────────────┐
                               │  FactNode   │
                               │ "Intel stock│
                               │  dropped"   │
                               └─────────────┘
```

---

## Entity Resolution Rules

1. **Global Deduplication**: EntityNodes are unique per `(name, group_id)`. If "Apple" appears in 10 chunks, there is ONE EntityNode with 10+ edges.

2. **Resolution Process**:
   - Extract entity from atomic fact
   - Search for existing EntityNode by name (exact match)
   - If found: reuse existing node, add new edge to chunk
   - If not found: create new EntityNode, add edge to chunk

3. **Disambiguation**: Use entity descriptions and LLM verification to distinguish "Apple (company)" from "Apple (fruit)".

---

## Query Patterns

### "What did Apple do?"
```cypher
MATCH (e:EntityNode {name: "Apple", group_id: $gid})-[r]->(chunk:EpisodicNode)
RETURN type(r) as action, chunk.content as evidence
```

### "Who got sued?"
```cypher
MATCH (chunk:EpisodicNode)-[:GOT_SUED]->(e:EntityNode {group_id: $gid})
RETURN e.name, chunk.content
```

### "What topics does this chunk cover?"
```cypher
MATCH (t:TopicNode)-[:ABOUT]->(chunk:EpisodicNode {uuid: $chunk_id})
RETURN t.name
```

### "What caused Apple to sue Intel?" (via FactNode causation)
```cypher
MATCH (cause:FactNode)-[:CAUSES]->(effect:FactNode {content: ~".*Apple.*sued.*Intel.*"})
RETURN cause.content
```

### "Show me everything about this chunk"
```cypher
MATCH (chunk:EpisodicNode {uuid: $chunk_id})
OPTIONAL MATCH (subj:EntityNode)-[r1]->(chunk)
OPTIONAL MATCH (chunk)-[r2]->(obj:EntityNode)
OPTIONAL MATCH (t:TopicNode)-[:ABOUT]->(chunk)
RETURN chunk, collect(DISTINCT {entity: subj.name, role: "subject", rel: type(r1)}) as subjects,
       collect(DISTINCT {entity: obj.name, role: "object", rel: type(r2)}) as objects,
       collect(DISTINCT t.name) as topics
```

---

## Migration Considerations

### Schema Changes Required

1. **`EpisodicNode`**: Add `header_path` property
2. **`SectionNode`**: Remove entirely (or deprecate)
3. **`RelationshipType`**: Add passive edge variants
4. **`GraphAssembler`**: Rewrite to create Entity→Chunk→Entity pattern

### Data Migration

If existing data needs migration:
1. For each existing `Entity -[:PERFORMED]-> FactNode -[:TARGET]-> Entity`:
   - Get the EpisodicNode via `FactNode -[:MENTIONED_IN]-> EpisodicNode`
   - Create `Subject -[:REL_TYPE]-> EpisodicNode -[:REL_TYPE_PASSIVE]-> Object`

---

## Open Questions

1. **Passive Edge Naming Convention**: Should we use `GOT_X`, `X_BY`, `RECEIVED_X`, or something else?

2. **Edge Properties**: Should edges carry properties like `confidence`, `timestamp`, `fact_uuid` (link back to FactNode)?

3. **Multiple Facts Same Chunk**: If one chunk contains "Apple sued Intel" AND "Apple acquired Intel", the EntityNode→Chunk edges would be `:SUED` and `:ACQUIRED`. Is this correct?

---

## Summary

| Aspect | Before (V1) | After (V2) |
|--------|-------------|------------|
| Relationship hub | FactNode | EpisodicNode (Chunk) |
| Entity→Entity path | 2 hops through FactNode | 2 hops through Chunk |
| Edge types | Generic (PERFORMED, TARGET) | Semantic (SUED, ACQUIRED) |
| Provenance | Via MENTIONED_IN | Structural (edges go through chunk) |
| SectionNode | Separate node | Merged into EpisodicNode.header_path |
| FactNode | Hub for relationships | Storage for atomic text + causation only |
