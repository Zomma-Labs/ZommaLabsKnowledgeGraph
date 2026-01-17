"""
Pydantic schemas for the Hybrid CoT-GNN Query Pipeline.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from dataclasses import dataclass, field


class QuestionType(str, Enum):
    """Classification of question types for routing retrieval and expansion."""

    FACTUAL = "factual"  # Simple fact lookup: "What happened in Boston?"
    COMPARISON = "comparison"  # Compare entities: "How do Boston and NY differ?"
    CAUSAL = "causal"  # Cause/effect: "Why did wages increase?"
    TEMPORAL = "temporal"  # Time-based: "What changed from Oct to Nov?"
    ENUMERATION = "enumeration"  # List items: "Which districts saw growth?"


class EntityHint(BaseModel):
    """An entity or topic hint with contextual definition for better matching."""

    name: str = Field(..., description="The entity/topic name as mentioned in the question")
    definition: str = Field(
        ...,
        description="Brief contextual definition to aid matching (e.g., 'Federal Reserve regional banking district')"
    )


class SubQuery(BaseModel):
    """A focused sub-query for targeted retrieval."""

    query_text: str = Field(..., description="Search query text (e.g., 'inflation Boston')")
    target_info: str = Field(..., description="What this sub-query aims to find")
    entity_hints: list[str] = Field(
        default_factory=list, description="Entity names to resolve"
    )
    topic_hints: list[str] = Field(
        default_factory=list, description="Topic names to resolve"
    )


class QueryDecomposition(BaseModel):
    """
    Structured output from Phase 1: Query Decomposition.
    Uses chain-of-thought: enumerate required info -> generate sub-queries -> classify.
    """

    # Step 1: What information is needed?
    required_info: list[str] = Field(
        ..., description="List of distinct pieces of information needed to answer"
    )

    # Step 2: Sub-queries to find that information
    sub_queries: list[SubQuery] = Field(
        ..., description="Targeted search queries (combinatorial for multi-entity)"
    )

    # Step 3: Hints for direct graph lookup (with definitions for better matching)
    entity_hints: list[EntityHint] = Field(
        default_factory=list,
        description="Entities to resolve, each with a contextual definition"
    )
    topic_hints: list[EntityHint] = Field(
        default_factory=list,
        description="Topics/themes to search for, each with a contextual definition"
    )
    relationship_hints: list[str] = Field(
        default_factory=list,
        description="Relationship phrases with modifiers (e.g., 'reported slight growth')"
    )

    # Step 4: Temporal scope
    temporal_scope: Optional[str] = Field(
        default=None,
        description="Time period if specified (e.g., 'October 2025', 'recent')",
    )

    # Step 5: Question classification
    question_type: QuestionType = Field(
        ..., description="Classification for routing to retrieval/expansion strategy"
    )

    # Confidence and reasoning
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reasoning: str = Field(default="", description="Chain-of-thought explanation")


class ScoredFact(BaseModel):
    """A fact with multi-source scoring."""

    fact_id: str
    content: str
    subject: str
    edge_type: str
    object: str
    chunk_id: Optional[str] = None
    chunk_content: Optional[str] = None
    chunk_header: Optional[str] = None
    doc_id: Optional[str] = None
    document_date: Optional[str] = None

    # Scoring components
    vector_score: float = 0.0  # From vector search
    rrf_score: float = 0.0  # From RRF fusion
    llm_relevance: float = 0.0  # From gpt-5-mini scoring
    cross_query_boost: float = 0.0  # Bonus for multi-query hits
    final_score: float = 0.0  # Combined score

    # Expansion flag
    should_expand: bool = False  # Whether to do 1-hop expansion from entities

    # Provenance
    found_by_queries: list[str] = Field(
        default_factory=list
    )  # Which sub-queries found this


class EvidencePool(BaseModel):
    """Collected and scored evidence for synthesis."""

    scored_facts: list[ScoredFact] = Field(default_factory=list)
    coverage_map: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Maps required_info items to fact_ids that cover them",
    )
    entities_found: list[str] = Field(default_factory=list)
    expansion_performed: bool = False


class StructuredAnswer(BaseModel):
    """Final structured answer with provenance."""

    answer: str = Field(..., description="The synthesized answer")
    evidence_refs: list[str] = Field(
        default_factory=list, description="Chunk IDs supporting the answer"
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    gaps: list[str] = Field(
        default_factory=list,
        description="Required info that wasn't fully covered",
    )

    # Timing breakdown
    decomposition_time_ms: int = 0
    retrieval_time_ms: int = 0
    scoring_time_ms: int = 0
    expansion_time_ms: int = 0
    synthesis_time_ms: int = 0

    @property
    def total_time_ms(self) -> int:
        return (
            self.decomposition_time_ms
            + self.retrieval_time_ms
            + self.scoring_time_ms
            + self.expansion_time_ms
            + self.synthesis_time_ms
        )


@dataclass
class PipelineResult:
    """Complete result from the hybrid pipeline."""

    question: str
    answer: StructuredAnswer
    decomposition: QueryDecomposition
    evidence_pool: EvidencePool

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer.answer,
            "confidence": self.answer.confidence,
            "question_type": self.decomposition.question_type.value,
            "sub_queries": [sq.query_text for sq in self.decomposition.sub_queries],
            "facts_retrieved": len(self.evidence_pool.scored_facts),
            "expansion_performed": self.evidence_pool.expansion_performed,
            "gaps": self.answer.gaps,
            "timing": {
                "decomposition_ms": self.answer.decomposition_time_ms,
                "retrieval_ms": self.answer.retrieval_time_ms,
                "scoring_ms": self.answer.scoring_time_ms,
                "expansion_ms": self.answer.expansion_time_ms,
                "synthesis_ms": self.answer.synthesis_time_ms,
                "total_ms": self.answer.total_time_ms,
            },
        }


# Scoring schema for gpt-5-mini batch scoring
class FactScore(BaseModel):
    """Scoring result for a single fact."""

    fact_index: int = Field(..., description="0-indexed position in the input list")
    relevance: float = Field(
        ..., ge=0.0, le=1.0, description="How relevant to the question"
    )
    should_expand: bool = Field(
        default=False, description="Whether to explore connected entities"
    )


class BatchScoringResult(BaseModel):
    """Batch scoring output."""

    scores: list[FactScore] = Field(default_factory=list)


# =============================================================================
# Sub-Query Parallel Retrieval Schemas
# =============================================================================

@dataclass
class SubQueryResult:
    """Result from a single sub-query retrieval."""

    sub_query_text: str
    target_info: str
    facts: list = field(default_factory=list)  # list[ScoredFact]
    resolved_entities: list = field(default_factory=list)
    resolved_topics: list = field(default_factory=list)
    retrieval_time_ms: int = 0
    resolution_time_ms: int = 0


@dataclass
class ParallelRetrievalResult:
    """Combined result from all parallel sub-query retrievals."""

    sub_query_results: list = field(default_factory=list)  # list[SubQueryResult]
    combined_facts: list = field(default_factory=list)  # list[ScoredFact] - deduped and boosted
    cross_query_boosted_fact_ids: set = field(default_factory=set)  # fact_ids that got boosted
    total_retrieval_time_ms: int = 0
