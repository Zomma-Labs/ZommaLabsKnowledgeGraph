"""
Hybrid CoT-GNN Query Pipeline.

Combines Chain-of-Thought query decomposition with GNN-inspired
multi-hop graph expansion for knowledge graph querying.

LLM Call Budget:
- Decomposition: gpt-5.2 (1 call)
- Scoring: gpt-5-mini (1 call)
- Synthesis: gpt-5.2 (1 call)
Total: 3 LLM calls (2 main + 1 mini)
"""

from .schemas import (
    QuestionType,
    SubQuery,
    QueryDecomposition,
    ScoredFact,
    EvidencePool,
    StructuredAnswer,
    PipelineResult,
)
from .pipeline import HybridCoTGNNPipeline, query_hybrid_cot

__all__ = [
    "QuestionType",
    "SubQuery",
    "QueryDecomposition",
    "ScoredFact",
    "EvidencePool",
    "StructuredAnswer",
    "PipelineResult",
    "HybridCoTGNNPipeline",
    "query_hybrid_cot",
]
