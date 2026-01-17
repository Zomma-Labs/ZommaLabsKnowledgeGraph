"""
V5 Querying System: Entity-Anchored Deep Research

A simplified, deep-research-inspired querying system for knowledge graph QA.
Follows the pattern: Decompose -> Parallel Sub-Research -> Per-Subquery Synthesis -> Final Synthesis.

Key features:
- Single execution path (no dual modes)
- Per-subquery answers before final synthesis
- Dynamic multi-hop via LLM-guided expansion
- Configurable features for A/B testing
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
from .pipeline import V5Pipeline
from .researcher import Researcher
from .graph_store import GraphStore

__all__ = [
    # Main entry point
    "V5Pipeline",
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
