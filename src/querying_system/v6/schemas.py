"""
V6 Schemas: Data models for the Threshold-Only Deep Research pipeline.

Key difference from V5: threshold-only retrieval (no LLM scoring).
OpenAI text-embedding-3-large provides 3x better score separation,
enabling us to use a simple threshold filter instead of LLM scoring.
"""

from dataclasses import dataclass, field
from typing import Optional
from pydantic import BaseModel, Field


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ResearcherConfig:
    """
    Configurable flags for A/B testing different retrieval strategies.

    V6 CHANGES:
    - use_llm_scoring: False by default (threshold-only)
    - relevance_threshold: 0.65 for OpenAI embeddings
    """
    # Feature toggles
    enable_global_search: bool = True      # Always run global search alongside scoped
    enable_gap_expansion: bool = True      # LLM-guided expansion when gaps detected
    enable_entity_drilldown: bool = True   # Extra retrieval for ENUMERATION questions
    enable_refinement_loop: bool = True    # Refine vague answers with targeted searches
    enable_llm_fact_filter: bool = True    # LLM-based fact filtering for precision
    enable_fact_filter_critique: bool = True  # Critique loop for fact filter

    # V6: Threshold-only retrieval (no LLM scoring)
    use_llm_scoring: bool = False          # Disabled by default in V6
    relevance_threshold: float = 0.65      # OpenAI embeddings score cutoff

    # Retrieval parameters
    global_top_k: int = 30                 # Max facts from global search
    scoped_threshold: float = 0.3          # Min similarity for scoped search
    drilldown_max_entities: int = 10       # Max entities to drill down on
    max_facts_to_score: int = 50           # Max facts to filter/score

    # Refinement parameters
    max_refinement_loops: int = 2          # Max iterations of refinement
    refinement_search_top_k: int = 20      # Facts per refinement search
    refinement_confidence_threshold: float = 0.85  # Skip refinement if confidence >= this

    # Synthesis parameters
    top_k_evidence: int = 15               # Facts to include in synthesis
    top_k_evidence_enumeration: int = 40   # More facts for enumeration questions


# =============================================================================
# Resolution Results
# =============================================================================

@dataclass
class ResolvedEntity:
    """A resolved entity from the knowledge graph."""
    original_hint: str           # The hint from decomposition
    resolved_name: str           # Canonical name in the graph
    confidence: float = 1.0      # Resolution confidence


@dataclass
class ResolvedTopic:
    """A resolved topic from the knowledge graph."""
    original_hint: str
    resolved_name: str
    confidence: float = 1.0


@dataclass
class ResolvedContext:
    """Combined resolution results for a sub-query."""
    entities: list[ResolvedEntity] = field(default_factory=list)
    topics: list[ResolvedTopic] = field(default_factory=list)

    @property
    def entity_names(self) -> list[str]:
        return [e.resolved_name for e in self.entities]

    @property
    def topic_names(self) -> list[str]:
        return [t.resolved_name for t in self.topics]

    @property
    def is_empty(self) -> bool:
        return not self.entities and not self.topics


# =============================================================================
# Facts
# =============================================================================

@dataclass
class RawFact:
    """
    A fact retrieved from the graph before scoring.

    Minimal structure - scoring adds additional fields.
    """
    fact_id: str
    content: str
    subject: str
    edge_type: str
    object: str

    # Chunk context (for evidence)
    chunk_id: str
    chunk_content: Optional[str] = None
    chunk_header: str = ""

    # Document context
    doc_id: str = ""
    document_date: str = ""

    # Retrieval metadata
    vector_score: float = 0.0
    source: str = ""  # "scoped:EntityName", "global_vector", "global_keyword", "expansion"
    found_by_sources: list[str] = field(default_factory=list)


@dataclass
class ScoredFact:
    """
    A fact with full scoring after threshold filtering.

    V6 CHANGE: llm_relevance is always 0 (not used).
    final_score = vector_score (threshold filtering only).
    """
    # Core fact data
    fact_id: str
    content: str
    subject: str
    edge_type: str
    object: str

    # Chunk context
    chunk_id: str
    chunk_content: Optional[str] = None
    chunk_header: str = ""

    # Document context
    doc_id: str = ""
    document_date: str = ""

    # Scoring components
    vector_score: float = 0.0          # From vector search
    llm_relevance: float = 0.0         # Not used in V6 (always 0)
    cross_source_boost: float = 0.0    # Found by multiple sources
    final_score: float = 0.0           # V6: same as vector_score

    # Provenance
    source: str = ""                   # Original source
    found_by_sources: list[str] = field(default_factory=list)

    # Expansion flag
    should_expand: bool = False

    @classmethod
    def from_raw(cls, raw: RawFact, **overrides) -> "ScoredFact":
        """Create ScoredFact from RawFact with additional scoring fields."""
        raw_sources = raw.found_by_sources or ([raw.source] if raw.source else [])
        return cls(
            fact_id=raw.fact_id,
            content=raw.content,
            subject=raw.subject,
            edge_type=raw.edge_type,
            object=raw.object,
            chunk_id=raw.chunk_id,
            chunk_content=raw.chunk_content,
            chunk_header=raw.chunk_header,
            doc_id=raw.doc_id,
            document_date=raw.document_date,
            vector_score=raw.vector_score,
            source=raw.source,
            found_by_sources=raw_sources,
            **overrides
        )


# =============================================================================
# Gap Detection
# =============================================================================

@dataclass
class Gap:
    """A gap in the retrieved information."""
    missing: str              # What information is missing
    expand_from: str          # Entity to expand from (if applicable)


class GapItem(BaseModel):
    """A single gap in retrieved information."""
    missing: str = Field(..., description="What information is missing")
    expand_from: str = Field(default="", description="Entity to expand from")


class GapDetectionResult(BaseModel):
    """LLM output for gap detection."""
    gaps: list[GapItem] = Field(
        default_factory=list,
        description="List of gaps in the information"
    )
    sufficient: bool = Field(
        default=True,
        description="Whether current facts are sufficient to answer"
    )


# =============================================================================
# Sub-Query Results
# =============================================================================

@dataclass
class SubAnswer:
    """
    Result from a single sub-query researcher.

    Contains both the synthesized answer AND the evidence used.
    This is the key innovation from deep research pattern.
    """
    sub_query: str               # The question being answered
    target_info: str             # What this sub-query sought
    answer: str                  # Synthesized answer for this sub-query
    confidence: float            # 0-1 quality signal

    # Evidence trail
    facts_used: list[ScoredFact] = field(default_factory=list)
    entities_found: list[str] = field(default_factory=list)

    # Timing
    resolution_time_ms: int = 0
    retrieval_time_ms: int = 0
    scoring_time_ms: int = 0
    expansion_time_ms: int = 0
    synthesis_time_ms: int = 0

    @property
    def total_time_ms(self) -> int:
        return (
            self.resolution_time_ms +
            self.retrieval_time_ms +
            self.scoring_time_ms +
            self.expansion_time_ms +
            self.synthesis_time_ms
        )


# =============================================================================
# Evidence
# =============================================================================

@dataclass
class Evidence:
    """Evidence item for final output."""
    fact_id: str
    content: str
    subject: str
    edge_type: str
    object: str
    source_chunk: str
    chunk_header: str
    source_doc: str
    document_date: str
    score: float


# =============================================================================
# Pipeline Result
# =============================================================================

@dataclass
class PipelineResult:
    """Complete result from the V6 pipeline."""
    question: str
    answer: str
    confidence: float

    # Sub-query breakdown (the deep research pattern)
    sub_answers: list[SubAnswer] = field(default_factory=list)

    # Evidence (deduplicated across sub-answers)
    evidence: list[Evidence] = field(default_factory=list)

    # Metadata
    question_type: str = ""
    gaps: list[str] = field(default_factory=list)

    # Timing
    decomposition_time_ms: int = 0
    research_time_ms: int = 0       # Total time for all researchers
    synthesis_time_ms: int = 0      # Final synthesis time

    @property
    def total_time_ms(self) -> int:
        return (
            self.decomposition_time_ms +
            self.research_time_ms +
            self.synthesis_time_ms
        )

    def to_dict(self) -> dict:
        """Serialize for logging/evaluation."""
        return {
            "question": self.question,
            "answer": self.answer,
            "confidence": self.confidence,
            "question_type": self.question_type,
            "num_sub_queries": len(self.sub_answers),
            "sub_answers": [
                {
                    "query": sa.sub_query,
                    "answer": sa.answer[:200] + "..." if len(sa.answer) > 200 else sa.answer,
                    "confidence": sa.confidence,
                    "num_facts": len(sa.facts_used),
                }
                for sa in self.sub_answers
            ],
            "num_evidence": len(self.evidence),
            "gaps": self.gaps,
            "timing": {
                "decomposition_ms": self.decomposition_time_ms,
                "research_ms": self.research_time_ms,
                "synthesis_ms": self.synthesis_time_ms,
                "total_ms": self.total_time_ms,
            },
        }


# =============================================================================
# LLM Schemas (V6: Only used for synthesis, not scoring)
# =============================================================================

class SubAnswerSynthesis(BaseModel):
    """LLM output for sub-answer synthesis."""
    answer: str = Field(..., description="Synthesized answer to the sub-query")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Answer confidence")
    entities_mentioned: list[str] = Field(default_factory=list, description="Entities in the answer")


class FinalSynthesis(BaseModel):
    """LLM output for final answer synthesis."""
    answer: str = Field(..., description="Final synthesized answer")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    gaps: list[str] = Field(default_factory=list, description="Information gaps if any")


# =============================================================================
# Drilldown Selection
# =============================================================================

class DrilldownSelection(BaseModel):
    """LLM output for selecting entities to drill down on."""
    entities: list[str] = Field(
        default_factory=list,
        description="Entity names to drill down on"
    )


# =============================================================================
# Vagueness Detection (for refinement loop)
# =============================================================================

class VagueReference(BaseModel):
    """A vague reference that needs resolution."""
    vague_text: str = Field(..., description="The vague phrase (e.g., 'three Districts', 'several companies')")
    what_is_missing: str = Field(..., description="What specific info is missing (e.g., 'district names')")
    search_queries: list[str] = Field(
        default_factory=list,
        description="Targeted search queries to find the specifics (e.g., 'Boston economic activity', 'district growth summary')"
    )


class VaguenessDetectionResult(BaseModel):
    """LLM output for detecting vagueness in an answer."""
    is_vague: bool = Field(
        default=False,
        description="Whether the answer contains vague references that should be resolved"
    )
    vague_references: list[VagueReference] = Field(
        default_factory=list,
        description="List of vague references found"
    )
    reasoning: str = Field(
        default="",
        description="Why the answer is or isn't vague"
    )


# =============================================================================
# LLM Fact Filter (Precision Improvement)
# =============================================================================

class FactRelevance(BaseModel):
    """Assessment of a single fact's relevance."""
    fact_index: int = Field(..., description="Index of the fact (0-based)")
    relevant: bool = Field(..., description="Is this fact relevant to answering the question?")
    reason: str = Field(default="", description="Brief reason for relevance decision")


class FactFilterResult(BaseModel):
    """LLM output for filtering facts by relevance."""
    relevant_indices: list[int] = Field(
        default_factory=list,
        description="Indices of facts that are relevant to the question"
    )
    reasoning: str = Field(
        default="",
        description="Overall reasoning about which facts were selected"
    )


class FactFilterCritique(BaseModel):
    """Critique of the initial fact filtering."""
    missed_facts: list[int] = Field(
        default_factory=list,
        description="Indices of facts that should have been included but weren't"
    )
    wrong_inclusions: list[int] = Field(
        default_factory=list,
        description="Indices of facts that were incorrectly included"
    )
    critique: str = Field(
        default="",
        description="Explanation of what was missed or incorrectly included"
    )
