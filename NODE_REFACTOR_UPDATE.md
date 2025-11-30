# Project Architecture: Causal Knowledge Graph (Fact-as-Node)

## 1\. Executive Summary

**Current Objective:** Transition from a standard Property Graph to a **Reified Hypergraph** ("Fact-as-Node").
**Why:** We need to model **Causality** (e.g., *Fact A causes Fact B*). Standard graph databases (Neo4j) cannot create edges pointing to other edges. Therefore, we must promote "Events/Facts" to be nodes themselves.
**Key Adoption:** We are adopting the **Graphiti** framework for Entity Extraction, Deduplication (RRF), and Provenance (`EpisodicNode`), but implementing a custom **Assembler** for the Fact logic.

-----

## 2\. Schema Architecture

### A. The Schema Shift

The AI Agent must enforce the **New Model** in all graph write operations.

  * **❌ OLD Model (Forbidden):** Information compressed into edges.

      * `(Apple)-[:RELEASED_PRODUCT]->(iPhone)`
      * *Problem:* We cannot link a "Cause" to this specific release event.

  * **✅ NEW Model (Required):** Information exploded into nodes.

      * `(Apple)-[:PERFORMED]->(FactNode)-[:TARGET]->(iPhone)`
      * `(FactNode)-[:MENTIONED_IN]->(EpisodicNode)`
      * `(FactNode_A)-[:CAUSES]->(FactNode_B)`

### B. Python Node Definitions (`src/schemas/nodes.py`)

*Use this exact code structure for Pydantic models. Note that `FactNode` is new, while others are adapted from Graphiti.*

```python
from datetime import datetime
from enum import Enum
from uuid import uuid4
from typing import Any, List, Optional
from pydantic import BaseModel, Field

def utc_now(): return datetime.utcnow()

# --- Enums ---
class EpisodeType(str, Enum):
    message = 'message'
    json = 'json'
    text = 'text'

# --- Base Node ---
class Node(BaseModel):
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(default="")
    group_id: str = Field(description='partition of the graph')
    labels: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

# --- The Source (Provenance) ---
class EpisodicNode(Node):
    """Represents the document/chat source. All Facts must link here."""
    source: EpisodeType = Field(default=EpisodeType.text)
    source_description: str = Field(default="")
    content: str = Field(description='raw episode data')
    valid_at: datetime = Field(default_factory=utc_now)

# --- The Entity (Noun) ---
class EntityNode(Node):
    """Represents extraction anchors (People, Places, Orgs)."""
    name_embedding: list[float] | None = None
    summary: str = ""
    attributes: dict[str, Any] = {}

# --- The Fact (Event/Verb) ---
class FactNode(Node):
    """
    [NEW] Represents an atomic event. 
    Reified from an edge to allow Causal Linking.
    """
    content: str = Field(description="Natural text, e.g., 'Apple released iPhone'")
    fact_type: str = Field(default="statement")
    confidence: float = 1.0
    embedding: Optional[List[float]] = None # Critical for semantic deduplication
```

-----

## 3\. Implementation Workflow

### Step 1: Provenance Initialization

**Before** any extraction begins, an `EpisodicNode` must be created and saved. This acts as the "Container" for the graph update.

  * **Action:** Instantiate `EpisodicNode` with the raw document text.
  * **DB Action:** Save `EpisodicNode` to Neo4j.

### Step 2: Entity Extraction & Resolution (Graphiti Logic)

Use the existing `graphiti_core` logic for this step. Do not rewrite.

  * **Function:** `extract_nodes()` (Extracts raw entities)
  * **Function:** `resolve_extracted_nodes()` (Deduplicates against DB using Hybrid Search/RRF)
  * **Output:** A list of resolved `EntityNode` objects.

### Step 3: Fact Assembly (Custom Logic)

This is where the new architecture is applied. The assembler must take the resolved entities and the `EpisodicNode`, and create `FactNodes` connecting them.

**Logic flow for a single Fact:**

1.  **Semantic Check:** Embed the `fact_text`. Search Vector Index for existing `FactNodes`.
      * *Match Condition:* High cosine similarity (\>0.95) AND connected to the same Subject/Object Entities.
      * *If Match:* Use existing `FactNode`.
      * *If No Match:* Create new `FactNode`.
2.  **Structural Linking:** Connect Subject -\> Fact -\> Object.
3.  **Provenance Linking:** Connect Fact -\> Episode.

-----

## 4\. Cypher Write Patterns

The agent must use these specific Cypher queries to ensure data integrity.

### Pattern A: Creating a Fact

*Used in the Assembler to save an atomic event.*

```cypher
MATCH (episode:EpisodicNode {uuid: $episode_uuid})
MATCH (subj:EntityNode {uuid: $subject_uuid})
MATCH (obj:EntityNode {uuid: $object_uuid})

// 1. Merge Fact Node (Idempotent based on content hash or vector)
MERGE (f:FactNode {content: $fact_content})
ON CREATE SET 
    f.uuid = randomUUID(),
    f.type = $fact_type,
    f.embedding = $embedding,
    f.created_at = datetime()

// 2. Create Structure (Reification)
MERGE (subj)-[:PERFORMED]->(f)
MERGE (f)-[:TARGET]->(obj)

// 3. Create Provenance (Link to Source)
MERGE (f)-[:MENTIONED_IN]->(episode)
```

### Pattern B: Linking Causality

*Used when the LLM identifies that Fact A led to Fact B.*

```cypher
MATCH (cause:FactNode {uuid: $cause_uuid})
MATCH (effect:FactNode {uuid: $effect_uuid})

MERGE (cause)-[r:CAUSES]->(effect)
SET r.confidence = $confidence, r.created_at = datetime()
```

-----

## 5\. Critical Constraints

1.  **No Direct Entity-to-Entity Edges:** Never create `(Entity)-[:VERB]->(Entity)`. All actions must pass through a `FactNode`.
2.  **Mandatory Provenance:** Every single `FactNode` must have a `[:MENTIONED_IN]` relationship to an `EpisodicNode`. Orphan facts are not allowed.
3.  **Reuse Graphiti Resolution:** Do not attempt to resolve Entities manually. Rely on the `resolve_extracted_nodes` function which handles the complex RRF (Reciprocal Rank Fusion) logic.