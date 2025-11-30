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
    group_id: str = Field(description='partition of the graph', default="default")
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
