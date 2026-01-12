"""
Extraction Schemas for the Rearchitected Pipeline

Key difference from original: NO RelationshipType enum.
Relationships are free-form text descriptions, preserving nuance.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ExtractedFact(BaseModel):
    """
    A single extracted fact from financial text.

    Financial analyst perspective: What would someone search for?
    What connections matter for understanding entity interactions?
    """
    fact: str = Field(
        ...,
        description="The atomic proposition - a complete, self-contained statement"
    )

    # Subject entity
    subject: str = Field(
        ...,
        description="The primary entity performing the action (company, person, org)"
    )
    subject_type: str = Field(
        ...,
        description="Entity type: 'Company', 'Person', 'Organization', 'Location', 'Product', 'Topic'"
    )
    subject_summary: str = Field(
        default="",
        description="1-2 sentence description of the subject entity"
    )

    # Object entity
    object: str = Field(
        ...,
        description="The entity being acted upon or related to"
    )
    object_type: str = Field(
        ...,
        description="Entity type: 'Company', 'Person', 'Organization', 'Location', 'Product', 'Topic'"
    )
    object_summary: str = Field(
        default="",
        description="1-2 sentence description of the object entity"
    )

    # Relationship - FREE-FORM (no enum!)
    relationship: str = Field(
        ...,
        description="Free-form description of how subject relates to object (e.g., 'acquired', 'partnered with', 'influenced')"
    )

    # Context - REQUIRED for temporal search
    date_context: str = Field(
        ...,
        description="REQUIRED: Temporal context. Use specific date/period from text (e.g., 'Q3 2024', 'August 5, 2024'), or 'Document date: YYYY-MM-DD' as fallback"
    )
    topics: List[str] = Field(
        default_factory=list,
        description="Related financial concepts (e.g., 'M&A', 'Earnings', 'Labor Market')"
    )


class ExtractionResult(BaseModel):
    """Result of extracting facts from a chunk."""
    facts: List[ExtractedFact] = Field(
        default_factory=list,
        description="List of extracted facts from the chunk"
    )


class CritiqueResult(BaseModel):
    """Result of the reflexion critique step."""
    is_approved: bool = Field(
        ...,
        description="True if extraction is satisfactory, False if issues found"
    )
    critique: Optional[str] = Field(
        default=None,
        description="Specific issues found and corrections needed (if not approved)"
    )
    missed_facts: List[str] = Field(
        default_factory=list,
        description="Facts that should have been extracted but were missed"
    )
    corrections: List[str] = Field(
        default_factory=list,
        description="Specific corrections to entity names, types, or relationships"
    )


class EntityResolution(BaseModel):
    """Result of resolving an entity against the graph."""
    uuid: str = Field(..., description="UUID of the canonical entity")
    canonical_name: str = Field(..., description="Established name to use")
    is_new: bool = Field(..., description="True if this created a new entity")
    updated_summary: str = Field(default="", description="Combined summary with new info")
    source_chunks: List[str] = Field(default_factory=list, description="Chunk UUIDs that contributed")
    aliases: List[str] = Field(default_factory=list, description="Alternative names for this entity")


class TopicResolution(BaseModel):
    """Result of resolving a topic against the ontology."""
    uuid: str = Field(..., description="UUID of the resolved topic node")
    canonical_label: str = Field(..., description="Canonical topic name from ontology")
    is_new: bool = Field(..., description="True if this is a new topic not in ontology")
    definition: str = Field(default="", description="Definition of the topic from ontology")

    @property
    def canonical_name(self) -> str:
        """Alias for compatibility with EntityResolution interface in fact assembly."""
        return self.canonical_label


class EntityMatchDecision(BaseModel):
    """LLM decision on whether a new entity matches an existing one."""
    is_same: bool = Field(..., description="True if new entity is same as an existing candidate")
    match_index: Optional[int] = Field(
        default=None,
        description="Index of the matching candidate (1-based), None if distinct"
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of the decision"
    )


class EntityGroup(BaseModel):
    """A group of entity names that refer to the same real-world entity."""
    reasoning: str = Field(
        ...,
        description="Why these are the same entity - think step by step"
    )
    entity_type: str = Field(
        ...,
        description="PERSON|ORGANIZATION|INDEX|CURRENCY|COMMODITY"
    )
    canonical: str = Field(
        ...,
        description="The most formal/complete name to use as canonical"
    )
    members: list[str] = Field(
        ...,
        description="Other names/aliases (NOT including canonical)"
    )


class EntityDedupeResult(BaseModel):
    """LLM output for entity deduplication with reasoning."""
    groups: list[EntityGroup] = Field(
        default_factory=list,
        description="List of entity groups. Each group contains names referring to the same entity. "
                    "Singleton entities (no aliases) should be omitted."
    )


# ============================================================================
# Chain-of-Thought Extraction Schemas (V2 Extractor)
# ============================================================================

class EnumeratedEntity(BaseModel):
    """An entity discovered during the enumeration step of chain-of-thought extraction."""
    name: str = Field(..., description="Entity name as it appears in the text")
    entity_type: str = Field(
        ...,
        description="Entity type: 'Company', 'Person', 'Organization', 'Location', 'Product', 'Topic'"
    )
    summary: str = Field(
        default="",
        description="1-2 sentence description of the entity based on context"
    )


class ChainOfThoughtResult(BaseModel):
    """
    Chain-of-thought extraction result: enumerate entities first, then generate relationships.

    This forces the LLM to explicitly list all entities before determining relationships,
    which may improve entity coverage compared to single-pass extraction.
    """
    entities: List[EnumeratedEntity] = Field(
        default_factory=list,
        description="Step 1: ALL entities mentioned in the text (companies, people, orgs, etc.)"
    )
    facts: List[ExtractedFact] = Field(
        default_factory=list,
        description="Step 2: Relationships between the enumerated entities"
    )
