"""
V6 Querying System: Threshold-Only Deep Research

A simplified, deep-research-inspired querying system using OpenAI embeddings
with threshold-only retrieval (no LLM scoring).

Key features:
- Single execution path (no dual modes)
- Per-subquery answers before final synthesis
- Dynamic multi-hop via LLM-guided expansion
- Threshold-only scoring using OpenAI text-embedding-3-large (3072 dims)
- 3x better score separation enables elimination of LLM scoring
"""

from .schemas import (
    ResearcherConfig,
    SubAnswer,
    PipelineResult,
    RawFact,
    ScoredFact,
    ResolvedEntity,
    ResolvedTopic,
    Gap,
    Evidence,
)
from .pipeline import V6Pipeline, query_v6
from .researcher import Researcher
from .graph_store import GraphStore

__all__ = [
    # Main entry point
    "V6Pipeline",
    "query_v6",
    # Components
    "Researcher",
    "GraphStore",
    # Config
    "ResearcherConfig",
    # Data models
    "SubAnswer",
    "PipelineResult",
    "RawFact",
    "ScoredFact",
    "ResolvedEntity",
    "ResolvedTopic",
    "Gap",
    "Evidence",
]
