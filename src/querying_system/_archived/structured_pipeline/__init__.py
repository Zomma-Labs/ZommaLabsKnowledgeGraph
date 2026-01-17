"""
Structured KG-RAG Pipeline
==========================

A deterministic retrieval pipeline for Knowledge Graph QA.

Replaces the two-agent system with a 3-step flow:
1. QueryAnalyzer - Classifies query and creates retrieval plan (1 LLM call)
2. Retriever - Executes deterministic retrieval patterns (0 LLM calls)
3. AnswerGenerator - Synthesizes grounded answer from evidence (1 LLM call)

Usage:
    from src.querying_system.structured_pipeline import StructuredKGRAG

    pipeline = StructuredKGRAG(user_id="default")
    answer = await pipeline.query("What companies did Alphabet acquire?")
"""

from .pipeline import StructuredKGRAG
from .models import QueryType, QueryPlan, RetrievalResult, PipelineResult
from .query_analyzer import QueryAnalyzer
from .retriever import Retriever
from .answer_generator import AnswerGenerator

__all__ = [
    "StructuredKGRAG",
    "QueryType",
    "QueryPlan",
    "RetrievalResult",
    "PipelineResult",
    "QueryAnalyzer",
    "Retriever",
    "AnswerGenerator",
]
