"""
Deep Research with Simple RAG backend.
Same supervisor/researcher/synthesizer architecture, but uses vector search over chunks
instead of knowledge graph traversal.
"""

from .pipeline import DeepResearchRAGPipeline

__all__ = ["DeepResearchRAGPipeline"]
