"""
Deep Research KG-RAG Pipeline
=============================

Implements a multi-agent research pattern for complex queries:

1. Supervisor: Plans research and delegates to sub-agents
2. Researchers: Parallel agents investigate specific topics
3. Synthesizer: Combines findings into coherent answer

Inspired by LangChain's Open Deep Research pattern.
"""

from .pipeline import DeepResearchPipeline
from .state import DeepResearchResult, ResearchFinding
from .synthesizer import Synthesizer

__all__ = [
    "DeepResearchPipeline",
    "DeepResearchResult",
    "ResearchFinding",
    "Synthesizer",
]
