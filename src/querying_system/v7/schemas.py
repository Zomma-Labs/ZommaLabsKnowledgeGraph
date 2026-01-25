"""
V7 Schemas: Data models for the GraphRAG-aligned query pipeline.

V7 is simpler than V6 - no gap detection, refinement loops, or LLM fact filtering.
Uses KG to curate context for Gemini-3-pro synthesis.
"""

from dataclasses import dataclass, field
from typing import Optional
from pydantic import BaseModel, Field


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class V7Config:
    """
    Configuration for V7 GraphRAG-aligned pipeline.

    Simpler than V6 - threshold-only retrieval with no LLM scoring or refinement.
    """
    # Resolution thresholds (higher = more precise, lower = wider net)
    entity_threshold: float = 0.35      # Min similarity for entity resolution
    topic_threshold: float = 0.35       # Min similarity for topic resolution

    # Feature toggles
    search_definitions: bool = True     # Search topic definitions
    enable_1hop_expansion: bool = True  # 1-hop expansion for related entities
    enable_global_search: bool = True   # Global vector search alongside scoped

    # Retrieval parameters
    global_search_top_k: int = 50       # Max results from global search

    # Context limits (increased for better coverage)
    max_high_relevance_chunks: int = 30  # High relevance chunks for synthesis
    max_facts: int = 40                  # Facts for synthesis
    max_topic_chunks: int = 15           # Topic-related chunks
    max_low_relevance_chunks: int = 20   # Supporting context

    # Threshold for high vs low relevance split
    high_relevance_threshold: float = 0.45  # Lowered to include more chunks

    # Model assignments - use stronger models for synthesis
    decomposition_model: str = "gemini-3-flash-preview"  # Question decomposition (fast)
    resolution_model: str = "gemini-3-flash-preview"     # Entity/topic resolution (fast)
    synthesis_model: str = "gpt-5.2"                     # Final answer synthesis (accurate)

    # Abstention
    abstention_threshold: float = 0.3  # Abstain if confidence below this

    # Multi-tenant
    group_id: str = "default"


# =============================================================================
# Resolution Results
# =============================================================================

@dataclass
class ResolvedEntity:
    """A resolved entity from the knowledge graph."""
    original_hint: str           # The hint from decomposition
    resolved_name: str           # Canonical name in the graph
    summary: str = ""            # Entity summary from the graph
    confidence: float = 1.0      # Resolution confidence


@dataclass
class ResolvedTopic:
    """A resolved topic from the knowledge graph."""
    original_hint: str           # The hint from decomposition
    resolved_name: str           # Canonical topic name
    definition: str = ""         # Topic definition
    confidence: float = 1.0      # Resolution confidence


@dataclass
class ResolvedContext:
    """Combined resolution results for a query or sub-query."""
    entities: list[ResolvedEntity] = field(default_factory=list)
    topics: list[ResolvedTopic] = field(default_factory=list)

    @property
    def entity_names(self) -> list[str]:
        """Get list of resolved entity names."""
        return [e.resolved_name for e in self.entities]

    @property
    def topic_names(self) -> list[str]:
        """Get list of resolved topic names."""
        return [t.resolved_name for t in self.topics]

    @property
    def is_empty(self) -> bool:
        """Check if no entities or topics were resolved."""
        return not self.entities and not self.topics

    @property
    def entity_summaries(self) -> dict[str, str]:
        """Get mapping of entity names to summaries."""
        return {e.resolved_name: e.summary for e in self.entities if e.summary}

    @property
    def topic_definitions(self) -> dict[str, str]:
        """Get mapping of topic names to definitions."""
        return {t.resolved_name: t.definition for t in self.topics if t.definition}


# =============================================================================
# Retrieved Content
# =============================================================================

@dataclass
class RetrievedChunk:
    """A chunk retrieved from the knowledge graph."""
    chunk_id: str
    content: str
    header_path: str             # Hierarchical path for context
    doc_id: str
    document_date: str
    vector_score: float          # Similarity score from vector search
    source: str                  # How it was retrieved (e.g., "entity:Apple", "topic:M&A")


@dataclass
class RetrievedFact:
    """A fact retrieved from the knowledge graph."""
    fact_id: str
    content: str                 # The fact text
    subject: str                 # Subject entity
    edge_type: str               # Relationship type
    object: str                  # Object entity
    chunk_id: str                # Source chunk for provenance
    vector_score: float          # Similarity score


@dataclass
class RetrievedEntity:
    """An entity retrieved with its metadata."""
    name: str
    summary: str
    entity_type: str             # PERSON, ORGANIZATION, PLACE, etc.


# =============================================================================
# Structured Context
# =============================================================================

@dataclass
class StructuredContext:
    """
    Organized context for synthesis.

    Separates high/low relevance content and different content types
    for more effective prompting.
    """
    high_relevance_chunks: list[RetrievedChunk] = field(default_factory=list)
    entities: list[RetrievedEntity] = field(default_factory=list)
    facts: list[RetrievedFact] = field(default_factory=list)
    topic_chunks: list[RetrievedChunk] = field(default_factory=list)
    low_relevance_chunks: list[RetrievedChunk] = field(default_factory=list)

    def _format_chunk(self, chunk: "RetrievedChunk", index: int) -> str:
        """Format a single chunk with full metadata like V6."""
        lines = []
        lines.append(f"[Chunk {index}]")
        lines.append(f"DOCUMENT: {chunk.doc_id or 'unknown'}")
        if chunk.document_date:
            lines.append(f"TIME: {chunk.document_date}")
        if chunk.header_path:
            lines.append(f"HEADER: {chunk.header_path}")
        lines.append("")  # Blank line before content
        lines.append(chunk.content)
        return "\n".join(lines)

    def to_prompt_text(self) -> str:
        """Format context as text for LLM synthesis prompt."""
        sections = []

        # Entity summaries
        if self.entities:
            entity_lines = []
            for e in self.entities:
                if e.summary:
                    entity_lines.append(f"- {e.name} ({e.entity_type}): {e.summary}")
                else:
                    entity_lines.append(f"- {e.name} ({e.entity_type})")
            sections.append("## Entities\n" + "\n".join(entity_lines))

        # High relevance chunks - with full metadata
        if self.high_relevance_chunks:
            chunk_lines = []
            for i, c in enumerate(self.high_relevance_chunks, 1):
                chunk_lines.append(self._format_chunk(c, i))
            sections.append("## High Relevance Context\n" + "\n\n---\n\n".join(chunk_lines))

        # Facts
        if self.facts:
            fact_lines = []
            for f in self.facts:
                fact_lines.append(f"- {f.subject} {f.edge_type} {f.object}: {f.content}")
            sections.append("## Facts\n" + "\n".join(fact_lines))

        # Topic-related chunks - with full metadata
        if self.topic_chunks:
            chunk_lines = []
            start_idx = len(self.high_relevance_chunks) + 1
            for i, c in enumerate(self.topic_chunks, start_idx):
                chunk_lines.append(self._format_chunk(c, i))
            sections.append("## Topic Context\n" + "\n\n---\n\n".join(chunk_lines))

        # Low relevance chunks - with full metadata
        if self.low_relevance_chunks:
            chunk_lines = []
            start_idx = len(self.high_relevance_chunks) + len(self.topic_chunks) + 1
            for i, c in enumerate(self.low_relevance_chunks, start_idx):
                chunk_lines.append(self._format_chunk(c, i))
            sections.append("## Additional Context\n" + "\n\n---\n\n".join(chunk_lines))

        return "\n\n".join(sections) if sections else "No context available."


# =============================================================================
# Sub-Query Results
# =============================================================================

@dataclass
class SubAnswer:
    """
    Result from a single sub-query.

    Contains the synthesized answer and metadata about the retrieval.
    """
    sub_query: str               # The sub-question being answered
    target_info: str             # What this sub-query seeks to find
    answer: str                  # Synthesized answer
    confidence: float            # 0-1 quality signal

    # Context used
    context: StructuredContext = field(default_factory=StructuredContext)
    entities_found: list[str] = field(default_factory=list)

    # Timing breakdown (milliseconds)
    resolution_time_ms: int = 0
    retrieval_time_ms: int = 0
    synthesis_time_ms: int = 0

    @property
    def total_time_ms(self) -> int:
        return (
            self.resolution_time_ms +
            self.retrieval_time_ms +
            self.synthesis_time_ms
        )


# =============================================================================
# Pipeline Result
# =============================================================================

@dataclass
class PipelineResult:
    """Complete result from the V7 pipeline."""
    question: str
    answer: str
    confidence: float

    # Sub-query breakdown
    sub_answers: list[SubAnswer] = field(default_factory=list)

    # Metadata
    question_type: str = ""      # FACTUAL, COMPARISON, ENUMERATION, etc.

    # Timing (milliseconds)
    decomposition_time_ms: int = 0
    resolution_time_ms: int = 0
    retrieval_time_ms: int = 0
    synthesis_time_ms: int = 0

    @property
    def total_time_ms(self) -> int:
        return (
            self.decomposition_time_ms +
            self.resolution_time_ms +
            self.retrieval_time_ms +
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
                    "target_info": sa.target_info,
                    "answer": sa.answer[:200] + "..." if len(sa.answer) > 200 else sa.answer,
                    "confidence": sa.confidence,
                    "entities_found": sa.entities_found,
                    "timing_ms": sa.total_time_ms,
                }
                for sa in self.sub_answers
            ],
            "timing": {
                "decomposition_ms": self.decomposition_time_ms,
                "resolution_ms": self.resolution_time_ms,
                "retrieval_ms": self.retrieval_time_ms,
                "synthesis_ms": self.synthesis_time_ms,
                "total_ms": self.total_time_ms,
            },
        }


# =============================================================================
# LLM Output Schemas (Pydantic for structured output)
# =============================================================================

class SubAnswerSynthesis(BaseModel):
    """LLM output for sub-answer synthesis with chain-of-thought."""
    thinking: str = Field(
        ...,
        description="Your reasoning process: verify the question, check evidence relevance, identify key facts"
    )
    answer: str = Field(..., description="Synthesized answer to the sub-query based on your thinking")
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Answer confidence (0-1) based on evidence quality and relevance"
    )
    entities_mentioned: list[str] = Field(
        default_factory=list,
        description="Entity names mentioned in the answer"
    )


class FinalSynthesis(BaseModel):
    """LLM output for final answer synthesis."""
    answer: str = Field(..., description="Final synthesized answer")
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Overall confidence in the answer"
    )
