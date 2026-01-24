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
    group_id: str = Field(description='Tenant/User ID - CRITICAL for security')
    labels: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

# --- The Source (Provenance) ---
class EpisodicNode(Node):
    """Represents the document/chat source. All Facts must link here."""
    source: EpisodeType = Field(default=EpisodeType.text)
    source_description: str = Field(default="")
    content: str = Field(description='raw episode data')
    header_path: str = Field(default="", description="Full breadcrumb path of the section")
    valid_at: datetime = Field(default_factory=utc_now)
    document_date: Optional[str] = Field(default=None, description="Date of the source document (YYYY-MM-DD)")

# --- The Document (Source Container) ---
class DocumentNode(Node):
    """
    Represents the parent document for EpisodicNodes.
    Ensures all chunks from the same document are connected.
    """
    file_type: str = Field(default="text")
    metadata: dict[str, Any] = Field(default_factory=dict)
    document_date: datetime = Field(default_factory=utc_now, description="When the document was written/created")

# --- The Entity (Noun) ---
class EntityNode(Node):
    """Represents extraction anchors (People, Places, Orgs)."""
    name_embedding: list[float] | None = None        # Embedding of name + summary (for semantic matching)
    name_only_embedding: list[float] | None = None   # Embedding of just the name (for direct name lookup)
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

# --- The Section (Hierarchy) ---


# --- The Topic (Theme) ---
class TopicNode(Node):
    """
    A Global Theme. Shared across all documents.
    Example: "Inflation", "Labor Markets", "Risk".
    """
    name: str = Field(description="Normalized name of the topic")
    definition: Optional[str] = Field(default=None, description="One-sentence definition from ontology")
    # fibo_id removed - will implement custom ontology later
    embedding: Optional[List[float]] = None
