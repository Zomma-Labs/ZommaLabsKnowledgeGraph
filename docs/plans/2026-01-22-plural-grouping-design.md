# Plural Grouping Feature Design

## Overview

Add automatic detection and grouping of plural/collective entity references during entity extraction. When the pipeline encounters entities like "districts" alongside specific entities like "Boston District", "New York District", etc., it creates an `INCLUDES` relationship linking the plural to its members.

This enables query-time expansion: when V6 finds a fact about "districts", it can traverse INCLUDES edges to retrieve facts about specific districts.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Grouping depth | Flat only | Avoid over-engineering; no nested hierarchies |
| Member type constraint | Homogeneous only | Can't mix PERSON and ORG in one grouping |
| Node type | EntityNode with `is_plural: true` | Reuse existing schema, simpler queries |
| Scope | Per-document, then global merge | Preserves document-level extraction, global dedup unifies |
| Merge behavior | Union INCLUDES edges | Simple; combined node includes all members |
| LLM input | Entities + definitions + sample facts | Context helps identify collective references |
| Orphan plurals | Create empty plural node | Preserves signal; global merge can populate later |
| LLM model | gpt-5.1 | Balanced cost/capability for classification task |

## Schema Changes

### EntityNode (modified)

```python
class EntityNode(Node):
    name: str
    name_embedding: list[float] | None = None
    name_only_embedding: list[float] | None = None
    summary: str = ""
    attributes: dict[str, Any] = {}
    is_plural: bool = False  # NEW: marks collective/plural entities
```

### New Relationship: INCLUDES

```cypher
(EntityNode {is_plural: true}) -[:INCLUDES]-> (EntityNode)
```

Properties: None required (simple membership edge)

## Pipeline Changes

### New Phase: 2f - Plural Grouping

**Location**: After Phase 2c (in-document entity dedup), before Phase 2d (graph resolution)

**Input**:
- Deduped entity list from Phase 2c
- Facts from Phase 1 extraction (for context)

**Process**:
1. Collect all deduped entities with their definitions
2. For each entity, gather sample facts it appears in (subject or object)
3. Call gpt-5.1 with structured output to identify plural groupings
4. Create plural EntityNodes with `is_plural=true`
5. Create INCLUDES edges to member entities

**Output**:
- Updated entity list (some marked as plural)
- INCLUDES relationships to buffer

### Phase 2d Enhancement: Plural Resolution

When resolving entities against Neo4j graph:

1. Regular entities: existing logic (unchanged)
2. Plural entities (`is_plural=true`):
   - Vector search for matching plural in graph
   - LLM verification (same as regular entities)
   - On match: **union INCLUDES edges** from both nodes

```python
def merge_plural_includes(existing_uuid: str, new_includes: List[str], neo4j):
    """Add new INCLUDES edges to existing plural node."""
    neo4j.query("""
        UNWIND $members AS member_uuid
        MATCH (p:EntityNode {uuid: $plural_uuid})
        MATCH (m:EntityNode {uuid: member_uuid})
        MERGE (p)-[:INCLUDES]->(m)
    """, {"plural_uuid": existing_uuid, "members": new_includes})
```

## LLM Prompt Design

### System Prompt

```
You are analyzing entities extracted from financial documents to identify plural/collective groupings.

TASK: Find entities that represent collections/groups of other entities in the list.

WHAT TO LOOK FOR:
- Plural nouns that refer to multiple specific entities: "districts" → [Boston District, NY District, ...]
- Collective terms: "subsidiaries" → [AWS, Whole Foods, ...], "segments" → [Cloud, Enterprise, ...]
- Group references: "banks" → [JPMorgan, Goldman Sachs, ...]

RULES:
1. Only group entities of the SAME TYPE (all ORG, all PLACE, etc.)
2. Use the sample facts to understand how entities are used
3. A plural with no clear members should still be identified (mark with empty members list)
4. Prefer the extracted plural name, but suggest a better name if clearer
5. Provide a definition for each plural grouping

OUTPUT: List of plural groupings with their members
```

### User Prompt Template

```
ENTITIES (with definitions and sample usage):

{for each entity}
{i}. "{entity.name}" ({entity.type})
   Definition: {entity.summary or "No definition"}
   Used in facts:
   - {fact1}
   - {fact2}
   ...
{end for}

Identify any plural/collective groupings among these entities.
```

### Structured Output Schema

```python
class PluralMember(BaseModel):
    """A member of a plural grouping."""
    entity_index: int  # Index from input list
    entity_name: str   # For verification

class PluralGrouping(BaseModel):
    """A plural/collective entity and its members."""
    plural_name: str
    definition: str
    entity_type: str  # Must match member types
    member_indices: List[int]  # Indices of member entities (can be empty)
    source_entity_index: Optional[int]  # If plural was in input list, its index

class PluralGroupingResult(BaseModel):
    """LLM output for plural grouping detection."""
    groupings: List[PluralGrouping]
```

## Implementation Plan

### File Changes

| File | Changes |
|------|---------|
| `src/schemas/nodes.py` | Add `is_plural: bool = False` to EntityNode |
| `src/agents/plural_grouper.py` | NEW: LLM agent for plural detection |
| `src/pipeline.py` | Add Phase 2f, enhance Phase 2d for plural merge |
| `src/agents/entity_registry.py` | Handle INCLUDES merge on plural resolution |
| `src/util/entity_dedup.py` | Pass facts context to plural grouper |

### New File: `src/agents/plural_grouper.py`

```python
"""
MODULE: Plural Grouper
DESCRIPTION: Identifies plural/collective entities and their members.

Uses gpt-5.1 to analyze deduped entity list + sample facts and detect
groupings like "districts" → [Boston District, NY District, ...].
"""

class PluralGrouper:
    def __init__(self):
        self.llm = get_critique_llm()  # gpt-5.1
        self.structured_llm = self.llm.with_structured_output(PluralGroupingResult)

    def detect_groupings(
        self,
        entities: List[Dict],  # {name, type, summary, uuid}
        facts: List[Dict],     # {subject, object, fact_text}
        group_id: str
    ) -> List[PluralGrouping]:
        """Detect plural groupings from entity list."""
        # Build prompt with entities + sample facts
        # Call LLM
        # Return groupings
        ...

    def create_plural_nodes(
        self,
        groupings: List[PluralGrouping],
        entity_lookup: Dict[str, EntityResolution],
        buffer: BulkWriteBuffer
    ) -> None:
        """Add plural EntityNodes and INCLUDES edges to buffer."""
        ...
```

### Pipeline Integration

```python
# In process_file(), after Phase 2c:

# ===== PHASE 2f: PLURAL GROUPING =====
print("  Phase 2f: Detecting plural groupings...")
plural_grouper = PluralGrouper()

# Collect entities with sample facts
entities_with_facts = []
for name, entity_data in entities_by_name.items():
    sample_facts = _get_sample_facts_for_entity(name, extractions, limit=3)
    entities_with_facts.append({
        "name": name,
        "type": entity_data[0]["entity"].entity_type,
        "summary": entity_data[0]["entity"].summary,
        "uuid": uuid_by_name[name],
        "sample_facts": sample_facts
    })

groupings = plural_grouper.detect_groupings(
    entities=entities_with_facts,
    facts=all_facts,
    group_id=group_id
)

print(f"    Found {len(groupings)} plural groupings")

# Mark plural entities and track INCLUDES relationships
plural_includes = []  # [(plural_uuid, member_uuid), ...]
for grouping in groupings:
    # Create or mark plural entity
    plural_uuid = _get_or_create_plural_entity(grouping, uuid_by_name, dedup_manager)

    # Track INCLUDES edges
    for member_idx in grouping.member_indices:
        member_name = entities_with_facts[member_idx]["name"]
        member_uuid = uuid_by_name[member_name]
        plural_includes.append((plural_uuid, member_uuid))
```

## Query-Time Enhancement (V6)

### GraphStore Enhancement

Add method to expand plural entities:

```python
class GraphStore:
    def expand_plural_entities(self, entity_uuids: List[str]) -> Dict[str, List[str]]:
        """
        For plural entities, return their INCLUDES members.

        Returns: {plural_uuid: [member_uuid, ...]}
        """
        result = self.neo4j.query("""
            UNWIND $uuids AS uuid
            MATCH (p:EntityNode {uuid: uuid, is_plural: true})-[:INCLUDES]->(m:EntityNode)
            RETURN p.uuid AS plural_uuid, collect(m.uuid) AS member_uuids
        """, {"uuids": entity_uuids})

        return {r["plural_uuid"]: r["member_uuids"] for r in result}
```

### Researcher Enhancement

When retrieving facts for an entity:

```python
def retrieve_facts_for_entity(self, entity_uuid: str) -> List[Fact]:
    # Check if plural
    expansions = self.graph_store.expand_plural_entities([entity_uuid])

    if entity_uuid in expansions:
        # Include facts from all members
        all_uuids = [entity_uuid] + expansions[entity_uuid]
        return self._get_facts_for_entities(all_uuids)
    else:
        return self._get_facts_for_entities([entity_uuid])
```

## Testing Strategy

### Unit Tests

1. **PluralGrouper detection**:
   - Given entities with clear plural ("districts" + specific districts) → detects grouping
   - Given homogeneous types → groups correctly
   - Given mixed types → does not group
   - Given orphan plural (no members) → creates empty grouping

2. **INCLUDES edge creation**:
   - Plural node gets `is_plural=true`
   - INCLUDES edges created to all members
   - Empty plural has no INCLUDES edges

3. **Global merge**:
   - Two "districts" nodes from different docs → merge into one
   - INCLUDES edges from both → union

### Integration Tests

1. **End-to-end pipeline**:
   - Process Beige Book with national summary + district sections
   - Verify "districts" plural created with correct members
   - Query for "districts" → get facts from all districts

2. **Multi-document merge**:
   - Process two Beige Books
   - Verify single "districts" node with combined members

## Rollout Plan

1. **Phase 1**: Implement PluralGrouper agent (detection only, no writes)
2. **Phase 2**: Add schema changes, integrate into pipeline
3. **Phase 3**: Add V6 query expansion
4. **Phase 4**: Test on sample Beige Book data
5. **Phase 5**: Production deployment

## Open Questions (Resolved)

| Question | Resolution |
|----------|------------|
| Flat vs hierarchical? | Flat only |
| Node type? | EntityNode with is_plural flag |
| Per-doc vs global? | Per-doc, then global merge |
| Merge behavior? | Union INCLUDES edges |
| Orphan plurals? | Create empty plural node |
| LLM model? | gpt-5.1 |
| Input context? | Entities + definitions + sample facts |

## Cost Estimate

- **LLM calls**: 1 call per document (gpt-5.1)
- **Tokens**: ~2000-5000 tokens per call depending on entity count
- **Added latency**: ~2-5 seconds per document

Minimal impact on overall pipeline cost/time.
