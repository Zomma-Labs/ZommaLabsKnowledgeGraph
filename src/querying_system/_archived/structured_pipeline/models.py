"""
Data models for the Structured KG-RAG Pipeline.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class QueryType(str, Enum):
    """Classification of user query types for routing to retrieval patterns."""

    ENTITY_ATTRIBUTE = "entity_attribute"
    # Questions about properties of a single entity
    # Examples: "What is Alphabet?", "Who is Sundar Pichai?"

    ENTITY_RELATIONSHIP = "entity_relationship"
    # Questions about relationships FROM or TO an entity
    # Examples: "What companies did Alphabet acquire?", "Who invested in OpenAI?"

    COMPARISON = "comparison"
    # Questions comparing multiple entities
    # Examples: "Compare Google and Microsoft's AI investments"

    TEMPORAL = "temporal"
    # Questions with explicit time constraints
    # Examples: "What happened to Alphabet in 2024?", "Recent acquisitions"

    GLOBAL_THEME = "global_theme"
    # Questions about trends/themes across the entire graph
    # Examples: "What are the main AI investment trends?", "Economic conditions"

    MULTI_HOP = "multi_hop"
    # Questions requiring traversal through intermediate entities
    # Examples: "What AI companies has Alphabet invested in?"

    UNKNOWN = "unknown"
    # Fallback for unclassifiable queries


class QueryPlan(BaseModel):
    """
    Structured output from the Query Analyzer.
    This is the contract between Step 1 (analysis) and Step 2 (retrieval).
    """

    query_type: QueryType = Field(default=QueryType.UNKNOWN)

    anchor_entities: list[str] = Field(default_factory=list)
    # Entity names mentioned in the question that should be resolved

    target_relationship: Optional[str] = Field(default=None)
    # Specific relationship/edge type to look for
    # Examples: "ACQUIRED", "INVESTED_IN", "HIRED"

    relationship_direction: Optional[str] = Field(default=None)
    # "outgoing" (entity did something), "incoming" (something done to entity), or "both"

    target_entity_type: Optional[str] = Field(default=None)
    # If looking for specific type of connected entity
    # Examples: "Person", "Organization", "Product"

    temporal_filter: Optional[dict] = Field(default=None)
    # Time constraints: {"start": "2024-01-01", "end": "2024-12-31"}

    comparison_entities: list[str] = Field(default_factory=list)
    # For COMPARISON queries: entities to compare

    comparison_aspects: list[str] = Field(default_factory=list)
    # What aspects to compare: ["revenue", "market_cap", "acquisitions"]

    fallback_search_terms: list[str] = Field(default_factory=list)
    # Descriptive phrases for search_relationships fallback

    confidence: float = Field(default=1.0)
    # Analyzer's confidence in this plan (0-1)

    reasoning: str = Field(default="")
    # Brief explanation of why this plan was chosen


@dataclass
class ResolvedEntity:
    """Result of entity resolution."""

    query: str  # Original search term
    resolved_name: Optional[str]  # Canonical name in graph (None if not found)
    match_type: str  # "exact", "semantic", "not_found"
    score: float  # Confidence score
    alternatives: list[str] = field(default_factory=list)  # Other possible matches


@dataclass
class RetrievedChunk:
    """A single piece of evidence from the graph."""

    chunk_id: str
    doc_id: str
    content: str
    header_path: str = ""

    # Provenance
    source_entity: str = ""  # Entity we traversed from
    target_entity: str = ""  # Entity we traversed to
    edge_type: str = ""  # Relationship type
    direction: str = ""  # "outgoing" or "incoming"

    # Temporal
    fact_date: Optional[str] = None  # Date from fact extraction
    doc_date: Optional[str] = None  # Document date

    # Relevance
    relevance_score: float = 1.0  # For ranking when we have many chunks

    def to_formatted_string(self) -> str:
        """Format for LLM context."""
        relationship = ""
        if self.source_entity and self.target_entity and self.edge_type:
            relationship = f"RELATIONSHIP: {self.source_entity} -[{self.edge_type}]-> {self.target_entity}\n"

        date_str = self.fact_date or self.doc_date or "N/A"

        return f'''"""
DOCUMENT: {self.doc_id}
CHUNK_ID: {self.chunk_id}
{relationship}DATE: {date_str}
HEADER: {self.header_path}

{self.content}
"""'''


@dataclass
class RetrievalResult:
    """Complete result from the retrieval step."""

    plan: QueryPlan
    chunks: list[RetrievedChunk] = field(default_factory=list)

    # Resolution results (for debugging)
    resolved_entities: list[ResolvedEntity] = field(default_factory=list)

    # Execution metadata
    retrieval_pattern_used: str = ""
    fallback_used: bool = False
    total_candidates_found: int = 0

    # Errors/warnings
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Whether retrieval found any evidence."""
        return len(self.chunks) > 0

    def get_context_for_llm(self, max_chunks: int = 5) -> str:
        """Format chunks for answer generation."""
        selected = self.chunks[:max_chunks]

        if not selected:
            return "No evidence chunks were retrieved."

        parts = [f"### Retrieved Evidence ({len(selected)} chunks)\n"]
        for i, chunk in enumerate(selected, 1):
            parts.append(f"--- Evidence {i} ---")
            parts.append(chunk.to_formatted_string())
            parts.append("")

        return "\n".join(parts)


@dataclass
class PipelineResult:
    """Final output from the complete pipeline."""

    question: str
    answer: str

    # Intermediate results
    plan: QueryPlan
    retrieval: RetrievalResult

    # Timing (milliseconds)
    analysis_time_ms: int = 0
    retrieval_time_ms: int = 0
    generation_time_ms: int = 0

    @property
    def total_time_ms(self) -> int:
        return self.analysis_time_ms + self.retrieval_time_ms + self.generation_time_ms

    def to_dict(self) -> dict:
        """Serialize for logging/eval."""
        return {
            "question": self.question,
            "answer": self.answer,
            "query_type": self.plan.query_type.value,
            "entities_resolved": [
                e.resolved_name
                for e in self.retrieval.resolved_entities
                if e.resolved_name
            ],
            "chunks_retrieved": len(self.retrieval.chunks),
            "retrieval_pattern": self.retrieval.retrieval_pattern_used,
            "fallback_used": self.retrieval.fallback_used,
            "timing": {
                "analysis_ms": self.analysis_time_ms,
                "retrieval_ms": self.retrieval_time_ms,
                "generation_ms": self.generation_time_ms,
                "total_ms": self.total_time_ms,
            },
            "errors": self.retrieval.errors,
            "warnings": self.retrieval.warnings,
        }
