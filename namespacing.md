Project Architecture: Causal Knowledge Graph (Fact-as-Node)

1. Executive Summary

Goal: Transition to a Reified Hypergraph ("Fact-as-Node") with strict Multi-tenancy.
Core Requirement 1 (Architecture): We must model Causality (e.g., "Event A caused Event B"). Since edges cannot point to edges, "Facts" must be Nodes.
Core Requirement 2 (Security): We use Soft Multi-tenancy. Every single node must carry a group_id (Tenant/User ID). Queries must strictly filter by this ID to prevent data leakage.

2. Security & Namespacing (Mandatory)

The Rule: Data isolation is enforced at the application level via the group_id property.

A. The Inheritance Chain

Data lineage flows from the Input Source down to the atomic units. The Agent must ensure group_id is passed down at every step.

API Input: Request contains user_id or org_id.

EpisodicNode: Created first. MUST be initialized with group_id = input_id.

EntityNode: Extracted from text. MUST inherit group_id from the EpisodicNode.

FactNode: Created by Assembler. MUST inherit group_id from the EpisodicNode.

B. Cypher Enforcement Patterns

Every database read/write operation must include the group_id constraint.

❌ BAD (Data Leak):

MATCH (n:Entity {name: 'Apple'}) RETURN n


✅ GOOD (Secure):

MATCH (n:Entity {name: 'Apple', group_id: $group_id}) RETURN n


✅ GOOD (Vector Search):
When running vector searches for deduplication, the filter is mandatory:

index.query(vector=..., filter={'group_id': current_group_id})


C. Maintenance

The Node class must implement a bulk deletion method for offboarding users.

@classmethod
async def delete_by_group_id(cls, driver, group_id):
    # Deletes all nodes of this type belonging to the tenant
    query = "MATCH (n {group_id: $group_id}) DETACH DELETE n"


3. The New Schema (Reification)

The "Fact-as-Node" Pattern

Instead of (Subject)-[:VERB]->(Object), we explode the relationship:

graph LR
    Sub[Entity: Apple] -->|PERFORMED| Fact(FactNode: "Apple releases iPhone")
    Fact -->|TARGET| Obj[Entity: iPhone]
    Fact -->|MENTIONED_IN| Ep[EpisodicNode: "TechCrunch Article"]
    
    Fact2(FactNode: "Stock hits record high")
    Fact -->|CAUSES {confidence: 0.9}| Fact2


Python Models (src/schemas/nodes.py)

Copy this exact schema. Note the group_id field in the base class.

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

# --- Base Node (With Security) ---
class Node(BaseModel):
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(default="")
    group_id: str = Field(description='Tenant/User ID - CRITICAL for security')
    labels: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

# --- The Source Wrapper (Provenance) ---
class EpisodicNode(Node):
    """Container for a document. All Facts must link here via [:MENTIONED_IN]."""
    source: EpisodeType = Field(default=EpisodeType.text)
    source_description: str = Field(default="") 
    content: str = Field(description='Raw text content')
    valid_at: datetime = Field(default_factory=utc_now)

# --- The Noun (Entity) ---
class EntityNode(Node):
    """Physical entities (People, Companies). Deduplicated via Graphiti RRF."""
    name_embedding: list[float] | None = None
    summary: str = ""
    attributes: dict[str, Any] = {}

# --- The Verb/Event (Fact) ---
class FactNode(Node):
    """
    [NEW] An atomic event reified as a node.
    Allows (Fact)-[:CAUSES]->(Fact).
    """
    content: str = Field(description="Natural text, e.g. 'Apple released iPhone'")
    fact_type: str = Field(default="statement")
    confidence: float = 1.0
    embedding: Optional[List[float]] = None # Vector for semantic deduplication


4. Implementation Blueprint

What to Reuse (From Graphiti)

Do not rewrite these. Import them or copy the logic exactly.

extract_nodes: The LLM/Reflexion loop to find raw entities.

resolve_extracted_nodes: The Hybrid Search (RRF) logic. Ensure you pass group_id to this function.

retrieve_episodes: The logic to fetch previous context. Ensure it filters by group_id.

What to Replace (Custom Logic)

DO NOT USE Graphiti.add_episode. It enforces the old edge schema.
IMPLEMENT CausalGraphPipeline.add_episode_causal instead.

The New Pipeline Flow (src/workflows/custom_pipeline.py)

Initialize Episode: Create EpisodicNode with group_id. Save to DB.

Context Retrieval: Call retrieve_episodes(group_id=...).

Entity Extraction: Call extract_nodes + resolve_extracted_nodes.

Fact Extraction (LLM): Prompt LLM for atomic facts.

Fact Assembly (The Assembler):

Semantic Deduplication: Vector Search FactNodes.

Constraint: filter={'group_id': group_id}.

Write to Graph: Create FactNode.

Constraint: Set property f.group_id = $group_id.

5. Cypher Write Patterns

Use these exact patterns for the Assembler. Note the ubiquitous usage of group_id.

A. Creating a Reified Fact

MATCH (episode:EpisodicNode {uuid: $ep_uuid, group_id: $group_id})
MATCH (subj:EntityNode {uuid: $sub_uuid, group_id: $group_id})
MATCH (obj:EntityNode {uuid: $obj_uuid, group_id: $group_id})

MERGE (f:FactNode {content: $content, group_id: $group_id})
ON CREATE SET 
    f.uuid = randomUUID(),
    f.embedding = $embedding,
    f.created_at = datetime()

MERGE (subj)-[:PERFORMED]->(f)
MERGE (f)-[:TARGET]->(obj)
MERGE (f)-[:MENTIONED_IN]->(episode)


B. Linking Causality

MATCH (cause:FactNode {uuid: $cause_id, group_id: $group_id})
MATCH (effect:FactNode {uuid: $effect_id, group_id: $group_id})

MERGE (cause)-[r:CAUSES]->(effect)
SET r.confidence = $confidence
