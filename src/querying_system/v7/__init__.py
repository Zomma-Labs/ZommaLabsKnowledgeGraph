"""
V7 Query Pipeline: GraphRAG-aligned query system.

V7 is a simpler pipeline than V6, designed to:
1. Use the knowledge graph to curate context (entities, facts, chunks)
2. Pass curated context to Gemini-3-pro for synthesis
3. No gap detection, refinement loops, or LLM fact filtering

The pipeline uses the KG structure for retrieval:
- Entity resolution via vector search
- Topic resolution via vector search
- Scoped fact retrieval through resolved entities
- Optional 1-hop expansion for related entities
- Optional global search for broader context

Usage:
    from src.querying_system.v7 import V7Config, PipelineResult

    config = V7Config(
        entity_threshold=0.3,
        enable_1hop_expansion=True,
        synthesis_model="gemini-3-pro-preview",
    )
"""

from .schemas import (
    # Configuration
    V7Config,
    # Resolution
    ResolvedEntity,
    ResolvedTopic,
    ResolvedContext,
    # Retrieved content
    RetrievedChunk,
    RetrievedFact,
    RetrievedEntity,
    # Context
    StructuredContext,
    # Results
    SubAnswer,
    PipelineResult,
    # LLM schemas
    SubAnswerSynthesis,
    FinalSynthesis,
)

from .prompts import (
    # Entity/Topic Resolution
    ENTITY_RESOLUTION_SYSTEM_PROMPT,
    TOPIC_RESOLUTION_SYSTEM_PROMPT,
    RESOLUTION_USER_PROMPT,
    # Sub-Answer Synthesis
    SUB_ANSWER_SYSTEM_PROMPT,
    SUB_ANSWER_USER_PROMPT,
    # Final Synthesis
    FINAL_SYNTHESIS_SYSTEM_PROMPT,
    FINAL_SYNTHESIS_USER_PROMPT,
    # Helper functions
    format_candidates_for_resolution,
    format_sub_answers_for_final,
    get_question_type_instructions,
    format_context_sections,
)

from .graph_store import GraphStore
from .context_builder import ContextBuilder
from .researcher import Researcher
from .pipeline import V7Pipeline, query_v7

__all__ = [
    # Configuration
    "V7Config",
    # Resolution
    "ResolvedEntity",
    "ResolvedTopic",
    "ResolvedContext",
    # Retrieved content
    "RetrievedChunk",
    "RetrievedFact",
    "RetrievedEntity",
    # Context
    "StructuredContext",
    # Results
    "SubAnswer",
    "PipelineResult",
    # LLM schemas
    "SubAnswerSynthesis",
    "FinalSynthesis",
    # Prompts
    "ENTITY_RESOLUTION_SYSTEM_PROMPT",
    "TOPIC_RESOLUTION_SYSTEM_PROMPT",
    "RESOLUTION_USER_PROMPT",
    "SUB_ANSWER_SYSTEM_PROMPT",
    "SUB_ANSWER_USER_PROMPT",
    "FINAL_SYNTHESIS_SYSTEM_PROMPT",
    "FINAL_SYNTHESIS_USER_PROMPT",
    # Helper functions
    "format_candidates_for_resolution",
    "format_sub_answers_for_final",
    "get_question_type_instructions",
    "format_context_sections",
    # GraphStore
    "GraphStore",
    # ContextBuilder
    "ContextBuilder",
    # Researcher
    "Researcher",
    # Pipeline
    "V7Pipeline",
    "query_v7",
]
