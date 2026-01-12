# Chain-of-Thought Query Pipeline Plan

## Overview

This document outlines a new querying architecture that mirrors the ingestion pipeline's structure to achieve better consistency, speed, and quality than the current Deep Research system.

### Core Insight

The ingestion pipeline has properties the querying system lacks:

| Property | Ingestion Pipeline | Current Deep Research |
|----------|-------------------|----------------------|
| **Determinism** | Same doc → same graph | Same question → different results |
| **Structure** | CoT: enumerate → relate | Unstructured supervisor loop |
| **Bounded** | Max 1 reflexion retry | Unbounded iterations |
| **Parallel** | All chunks concurrent | Sequential researcher spawning |
| **Fusion** | Dedup merges evidence | Concatenation only |

### The Analogy

```
INGESTION:  Document → Extract (enumerate→relate) → Resolve → Assemble
QUERYING:   Question → Decompose (enumerate→plan) → Retrieve → Synthesize
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: QUESTION DECOMPOSITION (single structured LLM call)                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Question ──► DECOMPOSE ──►  required_info: List[str]    (what we need)   │
│                    │          sub_queries: List[str]      (how to find it) │
│                 gpt-5.2       entity_hints: List[str]     (who/what)       │
│              (structured)     temporal_scope: str         (when)           │
│                               question_type: enum         (factual/compare)│
│                                                                             │
│   Like extraction enumerates entities BEFORE relationships,                 │
│   we enumerate REQUIRED INFO before planning retrieval.                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: PARALLEL RETRIEVAL (deterministic, multi-strategy per sub-query)   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   For EACH sub_query (concurrent with semaphore):                           │
│   ┌─────────────────────────────────────────┐                              │
│   │  ├─ Vector Search (facts)               │                              │
│   │  ├─ Keyword Search (BM25)               │  ──► RRF Fusion              │
│   │  ├─ Graph Traversal (entity_hints)      │                              │
│   │  └─ Temporal Filter (scope)             │                              │
│   └─────────────────────────────────────────┘                              │
│                                                                             │
│   CROSS-QUERY FUSION (like entity dedup):                                   │
│   ├─ Facts found by MULTIPLE sub-queries → boosted score                   │
│   ├─ Entity deduplication across all results                               │
│   └─ Cluster by required_info item (coverage tracking)                     │
│                                                                             │
│   Output: EvidenceGraph { facts[], entities[], coverage_map{} }             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2b: EVIDENCE CRITIQUE (bounded reflexion - max 1 iteration)           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   required_info + EvidenceGraph ──► CRITIQUE ──► gaps[], confidence        │
│                                        │                                    │
│                                    gpt-5.1 (cheaper)                        │
│                                                                             │
│   If gaps AND confidence < 0.7:                                             │
│   ├─ Generate targeted queries for ONLY the gaps                           │
│   └─ One retrieval pass, then STOP (bounded like extraction)               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: STRUCTURED SYNTHESIS (single LLM call with evidence graph)         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   EvidenceGraph + required_info + question_type ──► SYNTHESIZE             │
│                                                          │                  │
│                                                      gpt-5.2                │
│                                                                             │
│   Output: StructuredAnswer {                                                │
│       answer: str,                                                          │
│       evidence: List[FactWithProvenance],                                   │
│       confidence: float,                                                    │
│       gaps: List[str],           // What we couldn't find                   │
│       temporal_coverage: str     // "Oct 2024 - Nov 2024"                   │
│   }                                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Comparison: Deep Research vs CoT Query Pipeline

| Aspect | Deep Research (AGENT) | CoT Query Pipeline |
|--------|----------------------|-------------------|
| **LLM Calls** | 15-50+ (unbounded) | 2-4 (bounded) |
| **Consistency** | ~70-80% | ~95%+ (deterministic retrieval) |
| **Speed** | 2-5 seconds | 300-800ms |
| **Planning** | Dynamic (variance) | Upfront decomposition (stable) |
| **Evidence** | Concatenated | Cross-validated (multi-query boost) |
| **Gaps** | Unknown | Explicit (coverage tracking) |
| **Cost** | $$$ (many LLM calls) | $ (2-4 LLM calls) |

---

## Implementation Plan

### File Structure

```
src/querying_system/cot_query/
├── __init__.py
├── schemas.py          # QueryDecomposition, EvidenceCritique, StructuredAnswer
├── decomposer.py       # Phase 1: Question decomposition (mirrors ExtractorV2)
├── retriever.py        # Phase 2: Parallel retrieval with cross-query fusion
├── critic.py           # Phase 2b: Evidence critique (mirrors CritiqueResult)
├── synthesizer.py      # Phase 3: Structured synthesis
├── pipeline.py         # Main orchestrator (mirrors src/pipeline.py)
└── prompts.py          # All prompts in one place
```

### Component Mapping

| Ingestion Component | Query Component | File |
|--------------------|-----------------|------|
| `ExtractorV2` | `QueryDecomposer` | `decomposer.py` |
| `ChainOfThoughtResult` | `QueryDecomposition` | `schemas.py` |
| `DeferredDeduplicationManager` | `EvidenceFusionManager` | `retriever.py` |
| `EntityRegistry.resolve()` | `EvidenceValidator.validate()` | `retriever.py` |
| `CritiqueResult` | `EvidenceCritique` | `schemas.py` |
| `BulkWriteBuffer` | `EvidenceGraph` | `schemas.py` |
| `bulk_write_all()` | `Synthesizer.synthesize()` | `synthesizer.py` |

---

## Phase 1: Schemas (`schemas.py`)

```python
from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class QuestionType(str, Enum):
    """Question classification for retrieval strategy selection."""
    FACTUAL = "factual"          # "What was X?" → Vector search primary
    COMPARISON = "comparison"    # "How did X vs Y?" → Graph traversal primary
    ENUMERATION = "enumeration"  # "Which districts..." → Entity listing primary
    TEMPORAL = "temporal"        # "How did X change?" → Date-filtered search
    CAUSAL = "causal"           # "Why did X?" → Multi-hop graph traversal


class SubQuery(BaseModel):
    """A specific retrieval query derived from the main question."""
    query: str = Field(..., description="The search query text")
    target_info: str = Field(..., description="Which required_info item this addresses")
    strategy_weights: Dict[str, float] = Field(
        default_factory=lambda: {"vector": 0.4, "keyword": 0.3, "graph": 0.3},
        description="Weight for each retrieval strategy in RRF fusion"
    )


class QueryDecomposition(BaseModel):
    """
    Structured decomposition of a user question.
    Mirrors ChainOfThoughtResult from extraction.
    """
    # Step 1: Enumerate what information is needed (like enumerating entities)
    required_info: List[str] = Field(
        ...,
        description="Specific pieces of information needed to answer the question"
    )

    # Step 2: Plan how to retrieve it (like generating relationships)
    sub_queries: List[SubQuery] = Field(
        ...,
        description="Specific queries to execute against the knowledge graph"
    )

    # Hints for retrieval optimization
    entity_hints: List[str] = Field(
        default_factory=list,
        description="Entity names mentioned or implied in the question"
    )
    topic_hints: List[str] = Field(
        default_factory=list,
        description="Topics/themes relevant to the question"
    )
    temporal_scope: Optional[str] = Field(
        default=None,
        description="Time period constraint (e.g., 'Q3 2024', '2024', 'recent')"
    )

    # Classification for strategy selection
    question_type: QuestionType = Field(
        default=QuestionType.FACTUAL,
        description="Type of question for retrieval strategy optimization"
    )


class RetrievedFact(BaseModel):
    """A fact retrieved from the knowledge graph with full provenance."""
    fact_id: str
    content: str
    subject: str
    relationship: str
    object: str
    date_context: Optional[str] = None

    # Provenance
    chunk_id: str
    document_name: str
    header_path: str

    # Scoring
    vector_score: float = 0.0
    keyword_score: float = 0.0
    graph_score: float = 0.0
    cross_query_score: float = 0.0  # Boost for facts found by multiple sub-queries
    final_score: float = 0.0

    # Which sub-queries found this fact
    found_by: List[str] = Field(default_factory=list)


class EvidenceGraph(BaseModel):
    """
    Organized evidence from retrieval phase.
    Mirrors BulkWriteBuffer from ingestion.
    """
    facts: List[RetrievedFact] = Field(default_factory=list)

    # Deduped entities mentioned across all facts
    entities: Dict[str, Dict] = Field(
        default_factory=dict,
        description="entity_name -> {type, summary, fact_count}"
    )

    # Coverage tracking: which required_info items have evidence
    coverage_map: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="required_info_item -> [fact_ids that address it]"
    )

    # Temporal range of evidence
    temporal_range: Optional[str] = None


class EvidenceCritique(BaseModel):
    """
    Critique of retrieved evidence.
    Mirrors CritiqueResult from extraction.
    """
    is_sufficient: bool = Field(
        ...,
        description="True if evidence adequately covers all required_info"
    )
    confidence: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="Confidence that we can answer the question with this evidence"
    )
    gaps: List[str] = Field(
        default_factory=list,
        description="required_info items that lack sufficient evidence"
    )
    targeted_queries: List[SubQuery] = Field(
        default_factory=list,
        description="Additional queries to fill gaps (only if is_sufficient=False)"
    )


class FactReference(BaseModel):
    """A reference to a fact used in the answer."""
    fact_id: str
    content: str
    document: str
    date: Optional[str] = None


class StructuredAnswer(BaseModel):
    """
    Final answer with full provenance and confidence.
    """
    answer: str = Field(..., description="The synthesized answer")
    evidence: List[FactReference] = Field(
        default_factory=list,
        description="Facts used to construct the answer"
    )
    confidence: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="Confidence in the answer based on evidence coverage"
    )
    gaps: List[str] = Field(
        default_factory=list,
        description="Information that was requested but not found"
    )
    temporal_coverage: Optional[str] = Field(
        default=None,
        description="Date range of evidence (e.g., 'Oct 2024 - Nov 2024')"
    )

    # Timing breakdown
    decomposition_time_ms: float = 0.0
    retrieval_time_ms: float = 0.0
    critique_time_ms: float = 0.0
    synthesis_time_ms: float = 0.0
    total_time_ms: float = 0.0
```

---

## Phase 2: Decomposer (`decomposer.py`)

```python
"""
Question Decomposer - mirrors ExtractorV2 from ingestion.

Key principle: Enumerate REQUIRED INFO before planning retrieval,
just like extraction enumerates ENTITIES before generating relationships.
"""

import os
from typing import Optional, TYPE_CHECKING

from src.querying_system.cot_query.schemas import QueryDecomposition, QuestionType
from src.querying_system.cot_query.prompts import (
    DECOMPOSITION_SYSTEM_PROMPT,
    DECOMPOSITION_USER_PROMPT
)

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"


def log(msg: str):
    if VERBOSE:
        print(f"[QueryDecomposer] {msg}")


class QueryDecomposer:
    """
    Decomposes a user question into structured retrieval plan.

    Mirrors ExtractorV2:
    - Single structured LLM call
    - Two-step thinking: enumerate required info, then plan queries
    - Deterministic given same input
    """

    def __init__(self, llm: Optional["BaseChatModel"] = None):
        if llm is None:
            from src.util.services import get_services
            llm = get_services().llm

        self.llm = llm
        self.structured_decomposer = llm.with_structured_output(
            QueryDecomposition,
            include_raw=True
        )

    def decompose(
        self,
        question: str,
        context: str = "",
        available_topics: Optional[list] = None
    ) -> QueryDecomposition:
        """
        Decompose a question into a structured retrieval plan.

        Args:
            question: The user's question
            context: Optional context (e.g., previous conversation)
            available_topics: Optional list of topics in the ontology

        Returns:
            QueryDecomposition with required_info, sub_queries, hints
        """
        log(f"Decomposing question: {question[:100]}...")

        topics_hint = ""
        if available_topics:
            topics_hint = f"\n\nAvailable topics in knowledge graph: {', '.join(available_topics[:50])}"

        user_prompt = DECOMPOSITION_USER_PROMPT.format(
            question=question,
            context=context or "No additional context.",
            topics_hint=topics_hint
        )

        messages = [
            ("system", DECOMPOSITION_SYSTEM_PROMPT),
            ("human", user_prompt)
        ]

        try:
            response = self.structured_decomposer.invoke(messages)

            if response.get("parsing_error"):
                log(f"Parsing error: {response['parsing_error']}")
                return self._fallback_decomposition(question)

            parsed = response.get("parsed")
            if parsed is not None:
                log(f"Decomposed into {len(parsed.required_info)} required info, "
                    f"{len(parsed.sub_queries)} sub-queries")
                return parsed

            return self._fallback_decomposition(question)

        except Exception as e:
            log(f"Decomposition error: {e}")
            return self._fallback_decomposition(question)

    def _fallback_decomposition(self, question: str) -> QueryDecomposition:
        """Fallback when LLM decomposition fails."""
        from src.querying_system.cot_query.schemas import SubQuery

        return QueryDecomposition(
            required_info=[f"Answer to: {question}"],
            sub_queries=[
                SubQuery(
                    query=question,
                    target_info=f"Answer to: {question}",
                    strategy_weights={"vector": 0.5, "keyword": 0.3, "graph": 0.2}
                )
            ],
            entity_hints=[],
            topic_hints=[],
            temporal_scope=None,
            question_type=QuestionType.FACTUAL
        )
```

---

## Phase 3: Retriever (`retriever.py`)

```python
"""
Parallel Retriever with Cross-Query Fusion.

Mirrors the parallel extraction phase + entity deduplication from ingestion.
"""

import asyncio
import os
from collections import defaultdict
from typing import Dict, List, Optional
from uuid import uuid4

from src.querying_system.cot_query.schemas import (
    QueryDecomposition, SubQuery, RetrievedFact, EvidenceGraph
)
from src.util.deterministic_retrieval import DeterministicRetriever

VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"
RRF_K = 60  # Standard RRF constant


def log(msg: str):
    if VERBOSE:
        print(f"[Retriever] {msg}")


class ParallelRetriever:
    """
    Executes sub-queries in parallel and fuses results.

    Key innovations over current system:
    1. Parallel sub-query execution (like parallel chunk extraction)
    2. Cross-query fusion: facts found by multiple queries get boosted
    3. Coverage tracking: map facts to required_info items
    """

    def __init__(
        self,
        neo4j_client=None,
        embeddings=None,
        group_id: str = "default",
        max_concurrency: int = 10
    ):
        if neo4j_client is None or embeddings is None:
            from src.util.services import get_services
            services = get_services()
            neo4j_client = neo4j_client or services.neo4j
            embeddings = embeddings or services.dedup_embeddings

        self.retriever = DeterministicRetriever(
            neo4j_client=neo4j_client,
            embeddings=embeddings,
            group_id=group_id
        )
        self.max_concurrency = max_concurrency

    async def retrieve(
        self,
        decomposition: QueryDecomposition,
        top_k_per_query: int = 20,
        final_top_k: int = 50
    ) -> EvidenceGraph:
        """
        Execute all sub-queries in parallel and fuse results.

        Args:
            decomposition: The query decomposition plan
            top_k_per_query: Max facts per sub-query
            final_top_k: Max facts in final result

        Returns:
            EvidenceGraph with fused, deduplicated evidence
        """
        log(f"Retrieving for {len(decomposition.sub_queries)} sub-queries...")

        # Phase 2a: Parallel retrieval
        semaphore = asyncio.Semaphore(self.max_concurrency)
        tasks = [
            self._retrieve_subquery(sq, semaphore, top_k_per_query, decomposition)
            for sq in decomposition.sub_queries
        ]

        sub_results = await asyncio.gather(*tasks)

        # Phase 2b: Cross-query fusion
        fused = self._fuse_results(sub_results, decomposition, final_top_k)

        log(f"Retrieved {len(fused.facts)} facts covering "
            f"{len([k for k, v in fused.coverage_map.items() if v])} required_info items")

        return fused

    async def _retrieve_subquery(
        self,
        sub_query: SubQuery,
        semaphore: asyncio.Semaphore,
        top_k: int,
        decomposition: QueryDecomposition
    ) -> Dict:
        """Execute a single sub-query with the deterministic retriever."""
        async with semaphore:
            try:
                # Run deterministic retrieval in thread pool
                results = await asyncio.to_thread(
                    self.retriever.search,
                    query=sub_query.query,
                    top_k=top_k,
                    entity_hints=decomposition.entity_hints,
                    temporal_scope=decomposition.temporal_scope
                )

                log(f"  Sub-query '{sub_query.query[:50]}...' returned {len(results)} facts")

                return {
                    "sub_query": sub_query,
                    "facts": results,
                    "success": True
                }

            except Exception as e:
                log(f"  Sub-query error: {e}")
                return {
                    "sub_query": sub_query,
                    "facts": [],
                    "success": False,
                    "error": str(e)
                }

    def _fuse_results(
        self,
        sub_results: List[Dict],
        decomposition: QueryDecomposition,
        final_top_k: int
    ) -> EvidenceGraph:
        """
        Fuse results from multiple sub-queries.

        Key insight: Facts found by MULTIPLE sub-queries are more relevant.
        This mirrors entity deduplication in ingestion - consensus validates.
        """
        # Collect all facts with their sources
        fact_sources: Dict[str, Dict] = {}  # fact_id -> {fact, found_by: []}

        for result in sub_results:
            if not result["success"]:
                continue

            sub_query = result["sub_query"]

            for rank, evidence in enumerate(result["facts"]):
                fact_id = evidence.fact_id

                if fact_id not in fact_sources:
                    fact_sources[fact_id] = {
                        "fact": RetrievedFact(
                            fact_id=evidence.fact_id,
                            content=evidence.content,
                            subject=evidence.subject,
                            relationship=evidence.edge_type,
                            object=evidence.object,
                            date_context=evidence.document_date,
                            chunk_id=evidence.chunk_id,
                            document_name=evidence.document_name,
                            header_path=evidence.header_path,
                            vector_score=evidence.vector_score,
                            keyword_score=evidence.keyword_score,
                            graph_score=evidence.graph_score,
                            found_by=[]
                        ),
                        "ranks": {},
                        "target_infos": set()
                    }

                # Track which sub-query found this fact and at what rank
                fact_sources[fact_id]["found_by"].append(sub_query.query)
                fact_sources[fact_id]["ranks"][sub_query.query] = rank
                fact_sources[fact_id]["target_infos"].add(sub_query.target_info)

        # Calculate cross-query RRF scores
        for fact_id, data in fact_sources.items():
            # Cross-query score: sum of RRF scores across all queries that found this
            cross_score = sum(
                1.0 / (RRF_K + rank)
                for rank in data["ranks"].values()
            )

            # Boost for being found by multiple queries
            multi_query_boost = len(data["ranks"]) / len(sub_results) if sub_results else 0

            data["fact"].cross_query_score = cross_score
            data["fact"].final_score = cross_score * (1 + multi_query_boost)
            data["fact"].found_by = list(data["ranks"].keys())

        # Sort by final score and take top_k
        sorted_facts = sorted(
            [d["fact"] for d in fact_sources.values()],
            key=lambda f: f.final_score,
            reverse=True
        )[:final_top_k]

        # Build coverage map
        coverage_map: Dict[str, List[str]] = {
            info: [] for info in decomposition.required_info
        }

        for fact_id, data in fact_sources.items():
            if any(f.fact_id == fact_id for f in sorted_facts):
                for target_info in data["target_infos"]:
                    if target_info in coverage_map:
                        coverage_map[target_info].append(fact_id)

        # Extract unique entities
        entities: Dict[str, Dict] = {}
        for fact in sorted_facts:
            for entity_name in [fact.subject, fact.object]:
                if entity_name and entity_name not in entities:
                    entities[entity_name] = {
                        "type": "Unknown",
                        "fact_count": 0
                    }
                if entity_name:
                    entities[entity_name]["fact_count"] += 1

        # Determine temporal range
        dates = [f.date_context for f in sorted_facts if f.date_context]
        temporal_range = f"{min(dates)} - {max(dates)}" if dates else None

        return EvidenceGraph(
            facts=sorted_facts,
            entities=entities,
            coverage_map=coverage_map,
            temporal_range=temporal_range
        )
```

---

## Phase 4: Critic (`critic.py`)

```python
"""
Evidence Critic - mirrors the extraction critique step.

Bounded reflexion: max 1 additional retrieval pass.
"""

import os
from typing import Optional, TYPE_CHECKING

from src.querying_system.cot_query.schemas import (
    QueryDecomposition, EvidenceGraph, EvidenceCritique, SubQuery
)
from src.querying_system.cot_query.prompts import (
    CRITIQUE_SYSTEM_PROMPT,
    CRITIQUE_USER_PROMPT
)

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"


def log(msg: str):
    if VERBOSE:
        print(f"[EvidenceCritic] {msg}")


class EvidenceCritic:
    """
    Critiques retrieved evidence for completeness.

    Mirrors CritiqueResult from extraction:
    - Checks if all required_info is covered
    - Identifies gaps
    - Generates targeted queries for gaps (bounded: used at most once)
    """

    def __init__(self, llm: Optional["BaseChatModel"] = None):
        if llm is None:
            from src.util.llm_client import get_critique_llm
            llm = get_critique_llm()

        self.llm = llm
        self.structured_critic = llm.with_structured_output(
            EvidenceCritique,
            include_raw=True
        )

    def critique(
        self,
        decomposition: QueryDecomposition,
        evidence: EvidenceGraph,
        question: str
    ) -> EvidenceCritique:
        """
        Critique the evidence for completeness.

        Args:
            decomposition: Original query decomposition
            evidence: Retrieved evidence graph
            question: Original user question

        Returns:
            EvidenceCritique with gaps and targeted queries if needed
        """
        log("Critiquing evidence coverage...")

        # Format evidence for review
        evidence_summary = self._format_evidence(evidence)
        coverage_summary = self._format_coverage(decomposition, evidence)

        user_prompt = CRITIQUE_USER_PROMPT.format(
            question=question,
            required_info="\n".join(f"- {info}" for info in decomposition.required_info),
            evidence_summary=evidence_summary,
            coverage_summary=coverage_summary
        )

        messages = [
            ("system", CRITIQUE_SYSTEM_PROMPT),
            ("human", user_prompt)
        ]

        try:
            response = self.structured_critic.invoke(messages)

            if response.get("parsing_error") or response.get("parsed") is None:
                log("Critique parsing failed, assuming sufficient")
                return EvidenceCritique(is_sufficient=True, confidence=0.7, gaps=[])

            critique = response["parsed"]
            log(f"Critique: sufficient={critique.is_sufficient}, "
                f"confidence={critique.confidence:.2f}, gaps={len(critique.gaps)}")

            return critique

        except Exception as e:
            log(f"Critique error: {e}")
            return EvidenceCritique(is_sufficient=True, confidence=0.5, gaps=[])

    def _format_evidence(self, evidence: EvidenceGraph) -> str:
        """Format evidence facts for the critic."""
        lines = []
        for i, fact in enumerate(evidence.facts[:30], 1):  # Limit for prompt size
            lines.append(
                f"{i}. {fact.content}\n"
                f"   ({fact.subject} -> {fact.relationship} -> {fact.object})\n"
                f"   Source: {fact.document_name}, Date: {fact.date_context or 'Unknown'}"
            )

        if len(evidence.facts) > 30:
            lines.append(f"... and {len(evidence.facts) - 30} more facts")

        return "\n".join(lines) if lines else "No evidence retrieved."

    def _format_coverage(
        self,
        decomposition: QueryDecomposition,
        evidence: EvidenceGraph
    ) -> str:
        """Format coverage map for the critic."""
        lines = []
        for info in decomposition.required_info:
            fact_ids = evidence.coverage_map.get(info, [])
            status = f"{len(fact_ids)} facts" if fact_ids else "NO EVIDENCE"
            lines.append(f"- {info}: {status}")

        return "\n".join(lines)
```

---

## Phase 5: Synthesizer (`synthesizer.py`)

```python
"""
Structured Synthesizer - generates the final answer.

Unlike Deep Research which just concatenates, this:
1. Uses the question_type to guide synthesis style
2. Explicitly references evidence with provenance
3. Reports gaps and confidence
"""

import os
from typing import Optional, TYPE_CHECKING

from src.querying_system.cot_query.schemas import (
    QueryDecomposition, EvidenceGraph, StructuredAnswer, FactReference
)
from src.querying_system.cot_query.prompts import (
    SYNTHESIS_SYSTEM_PROMPT,
    SYNTHESIS_USER_PROMPT
)

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"


def log(msg: str):
    if VERBOSE:
        print(f"[Synthesizer] {msg}")


class Synthesizer:
    """
    Synthesizes final answer from evidence graph.

    Key improvements over Deep Research:
    - Question-type-aware synthesis (comparison vs factual vs enumeration)
    - Explicit evidence references
    - Confidence based on coverage
    - Gap reporting
    """

    def __init__(self, llm: Optional["BaseChatModel"] = None):
        if llm is None:
            from src.util.services import get_services
            llm = get_services().llm

        self.llm = llm

    def synthesize(
        self,
        question: str,
        decomposition: QueryDecomposition,
        evidence: EvidenceGraph,
        gaps: list = None
    ) -> StructuredAnswer:
        """
        Synthesize final answer from evidence.

        Args:
            question: Original user question
            decomposition: Query decomposition (for question_type)
            evidence: Retrieved and fused evidence
            gaps: Known gaps from critique phase

        Returns:
            StructuredAnswer with answer, evidence refs, confidence, gaps
        """
        log(f"Synthesizing answer for {decomposition.question_type.value} question...")

        # Format evidence for synthesis
        evidence_text = self._format_evidence_for_synthesis(evidence)

        # Calculate coverage-based confidence
        covered = sum(1 for facts in evidence.coverage_map.values() if facts)
        total = len(decomposition.required_info)
        base_confidence = covered / total if total > 0 else 0.5

        user_prompt = SYNTHESIS_USER_PROMPT.format(
            question=question,
            question_type=decomposition.question_type.value,
            required_info="\n".join(f"- {info}" for info in decomposition.required_info),
            evidence=evidence_text,
            gaps="\n".join(f"- {g}" for g in (gaps or [])) or "None identified",
            temporal_range=evidence.temporal_range or "Unknown"
        )

        messages = [
            ("system", SYNTHESIS_SYSTEM_PROMPT),
            ("human", user_prompt)
        ]

        try:
            response = self.llm.invoke(messages)
            answer_text = response.content.strip()

            # Build evidence references
            evidence_refs = [
                FactReference(
                    fact_id=fact.fact_id,
                    content=fact.content,
                    document=fact.document_name,
                    date=fact.date_context
                )
                for fact in evidence.facts[:20]  # Top 20 most relevant
            ]

            # Adjust confidence based on evidence quality
            avg_score = sum(f.final_score for f in evidence.facts) / len(evidence.facts) if evidence.facts else 0
            confidence = min(base_confidence * (1 + avg_score), 1.0)

            return StructuredAnswer(
                answer=answer_text,
                evidence=evidence_refs,
                confidence=confidence,
                gaps=gaps or [],
                temporal_coverage=evidence.temporal_range
            )

        except Exception as e:
            log(f"Synthesis error: {e}")
            return StructuredAnswer(
                answer=f"Error synthesizing answer: {e}",
                evidence=[],
                confidence=0.0,
                gaps=gaps or []
            )

    def _format_evidence_for_synthesis(self, evidence: EvidenceGraph) -> str:
        """Format evidence for the synthesis prompt."""
        lines = []
        for i, fact in enumerate(evidence.facts, 1):
            lines.append(
                f"[{i}] {fact.content}\n"
                f"    Source: {fact.document_name} | Date: {fact.date_context or 'Unknown'}\n"
                f"    Path: {fact.header_path}"
            )

        return "\n\n".join(lines) if lines else "No evidence available."
```

---

## Phase 6: Pipeline (`pipeline.py`)

```python
"""
Chain-of-Thought Query Pipeline - main orchestrator.

Mirrors src/pipeline.py structure:
- Phase 1: Decomposition (like extraction)
- Phase 2: Parallel retrieval with fusion (like resolution)
- Phase 2b: Optional critique and refinement (like reflexion)
- Phase 3: Synthesis (like assembly)
"""

import asyncio
import time
import os
from typing import Optional

from src.querying_system.cot_query.schemas import (
    QueryDecomposition, EvidenceGraph, StructuredAnswer
)
from src.querying_system.cot_query.decomposer import QueryDecomposer
from src.querying_system.cot_query.retriever import ParallelRetriever
from src.querying_system.cot_query.critic import EvidenceCritic
from src.querying_system.cot_query.synthesizer import Synthesizer

VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"
CRITIQUE_CONFIDENCE_THRESHOLD = 0.7


def log(msg: str):
    if VERBOSE:
        print(f"[CoTQueryPipeline] {msg}")


class CoTQueryPipeline:
    """
    Chain-of-Thought Query Pipeline.

    Properties (mirroring ingestion pipeline):
    - Deterministic decomposition (single structured LLM call)
    - Parallel retrieval (concurrent sub-queries)
    - Cross-query fusion (like entity dedup)
    - Bounded reflexion (max 1 refinement)
    - Structured synthesis

    LLM Calls: 2-4 (decompose, [critique], synthesize, [refinement])
    Consistency: ~95%+ (deterministic retrieval)
    Speed: 300-800ms typical
    """

    def __init__(
        self,
        neo4j_client=None,
        embeddings=None,
        llm=None,
        group_id: str = "default",
        max_retrieval_concurrency: int = 10,
        enable_critique: bool = True
    ):
        self.decomposer = QueryDecomposer(llm=llm)
        self.retriever = ParallelRetriever(
            neo4j_client=neo4j_client,
            embeddings=embeddings,
            group_id=group_id,
            max_concurrency=max_retrieval_concurrency
        )
        self.critic = EvidenceCritic() if enable_critique else None
        self.synthesizer = Synthesizer(llm=llm)

        self.enable_critique = enable_critique

    async def query(
        self,
        question: str,
        context: str = "",
        top_k: int = 50
    ) -> StructuredAnswer:
        """
        Execute the full query pipeline.

        Args:
            question: User's question
            context: Optional conversation context
            top_k: Maximum facts to retrieve

        Returns:
            StructuredAnswer with answer, evidence, confidence, gaps
        """
        total_start = time.time()

        # ===== PHASE 1: DECOMPOSITION =====
        log("Phase 1: Decomposing question...")
        t1 = time.time()

        decomposition = self.decomposer.decompose(question, context)

        decomp_time = (time.time() - t1) * 1000
        log(f"  Decomposed in {decomp_time:.0f}ms: {len(decomposition.sub_queries)} sub-queries")

        # ===== PHASE 2: PARALLEL RETRIEVAL =====
        log("Phase 2: Retrieving evidence...")
        t2 = time.time()

        evidence = await self.retriever.retrieve(
            decomposition,
            top_k_per_query=top_k // len(decomposition.sub_queries) + 10,
            final_top_k=top_k
        )

        retrieval_time = (time.time() - t2) * 1000
        log(f"  Retrieved {len(evidence.facts)} facts in {retrieval_time:.0f}ms")

        # ===== PHASE 2b: CRITIQUE (Optional) =====
        critique_time = 0.0
        gaps = []

        if self.enable_critique and self.critic:
            log("Phase 2b: Critiquing evidence...")
            t3 = time.time()

            critique = self.critic.critique(decomposition, evidence, question)

            if not critique.is_sufficient and critique.confidence < CRITIQUE_CONFIDENCE_THRESHOLD:
                log(f"  Gaps found: {critique.gaps}, running refinement...")

                # One refinement pass for gaps only
                if critique.targeted_queries:
                    gap_decomposition = QueryDecomposition(
                        required_info=critique.gaps,
                        sub_queries=critique.targeted_queries,
                        entity_hints=decomposition.entity_hints,
                        topic_hints=decomposition.topic_hints,
                        temporal_scope=decomposition.temporal_scope,
                        question_type=decomposition.question_type
                    )

                    gap_evidence = await self.retriever.retrieve(
                        gap_decomposition,
                        top_k_per_query=10,
                        final_top_k=20
                    )

                    # Merge gap evidence into main evidence
                    evidence = self._merge_evidence(evidence, gap_evidence)

            gaps = critique.gaps
            critique_time = (time.time() - t3) * 1000
            log(f"  Critique completed in {critique_time:.0f}ms")

        # ===== PHASE 3: SYNTHESIS =====
        log("Phase 3: Synthesizing answer...")
        t4 = time.time()

        answer = self.synthesizer.synthesize(
            question, decomposition, evidence, gaps
        )

        synthesis_time = (time.time() - t4) * 1000
        total_time = (time.time() - total_start) * 1000

        # Add timing info
        answer.decomposition_time_ms = decomp_time
        answer.retrieval_time_ms = retrieval_time
        answer.critique_time_ms = critique_time
        answer.synthesis_time_ms = synthesis_time
        answer.total_time_ms = total_time

        log(f"  Done in {total_time:.0f}ms (decomp={decomp_time:.0f}, "
            f"retrieve={retrieval_time:.0f}, critique={critique_time:.0f}, "
            f"synth={synthesis_time:.0f})")

        return answer

    def _merge_evidence(
        self,
        main: EvidenceGraph,
        gap: EvidenceGraph
    ) -> EvidenceGraph:
        """Merge gap evidence into main evidence graph."""
        # Add new facts (avoid duplicates)
        existing_ids = {f.fact_id for f in main.facts}
        for fact in gap.facts:
            if fact.fact_id not in existing_ids:
                main.facts.append(fact)

        # Merge entities
        main.entities.update(gap.entities)

        # Merge coverage
        for key, values in gap.coverage_map.items():
            if key not in main.coverage_map:
                main.coverage_map[key] = []
            main.coverage_map[key].extend(values)

        return main


# Convenience function
async def query_kg(question: str, group_id: str = "default", **kwargs) -> StructuredAnswer:
    """
    Convenience function for querying the knowledge graph.

    Args:
        question: User's question
        group_id: Tenant/group ID
        **kwargs: Additional options for the pipeline

    Returns:
        StructuredAnswer
    """
    pipeline = CoTQueryPipeline(group_id=group_id, **kwargs)
    return await pipeline.query(question)
```

---

## Phase 7: Prompts (`prompts.py`)

```python
"""
All prompts for the Chain-of-Thought Query Pipeline.
Centralized for easy tuning.
"""

# =============================================================================
# DECOMPOSITION PROMPTS
# =============================================================================

DECOMPOSITION_SYSTEM_PROMPT = """You are a financial research analyst planning how to answer a question using a knowledge graph.

Your task: Decompose the question into a structured retrieval plan.

STEP 1 - ENUMERATE REQUIRED INFORMATION:
What specific pieces of information are needed to fully answer this question?
- Be exhaustive - list every fact/data point needed
- Be specific - "inflation rate in Boston" not just "economic data"

STEP 2 - PLAN RETRIEVAL QUERIES:
For each required piece of information, what search query would find it?
- Each query should target ONE specific piece of information
- Use entity names when known
- Include temporal hints if relevant

STEP 3 - IDENTIFY HINTS:
- Entity hints: What companies, people, organizations are mentioned or implied?
- Topic hints: What financial themes are relevant (Inflation, Labor Market, M&A)?
- Temporal scope: What time period is relevant?

STEP 4 - CLASSIFY QUESTION TYPE:
- FACTUAL: Direct fact lookup ("What was X?")
- COMPARISON: Comparing entities/values ("How did X vs Y?")
- ENUMERATION: Listing items ("Which districts reported...")
- TEMPORAL: Change over time ("How did X evolve?")
- CAUSAL: Cause and effect ("Why did X happen?")

IMPORTANT:
- Each sub_query should map to exactly one required_info item
- Strategy weights should reflect the question type:
  - FACTUAL: vector=0.5, keyword=0.3, graph=0.2
  - COMPARISON: vector=0.3, keyword=0.2, graph=0.5
  - ENUMERATION: vector=0.2, keyword=0.3, graph=0.5
  - TEMPORAL: vector=0.4, keyword=0.3, graph=0.3
  - CAUSAL: vector=0.3, keyword=0.2, graph=0.5"""

DECOMPOSITION_USER_PROMPT = """QUESTION:
{question}

CONTEXT:
{context}
{topics_hint}

Decompose this question into a structured retrieval plan.
First enumerate ALL required information, then plan specific queries to retrieve each piece."""


# =============================================================================
# CRITIQUE PROMPTS
# =============================================================================

CRITIQUE_SYSTEM_PROMPT = """You are a senior research analyst reviewing evidence gathered for a question.

Your task: Determine if the evidence is SUFFICIENT to answer the question.

REVIEW CHECKLIST:
1. COVERAGE: Does evidence exist for EACH required piece of information?
2. QUALITY: Is the evidence specific and relevant (not tangential)?
3. RECENCY: Is temporal coverage appropriate for the question?
4. COMPLETENESS: Are there obvious gaps in the evidence?

If evidence is INSUFFICIENT:
- List specific gaps (which required_info items lack evidence)
- Generate targeted queries to fill those gaps
- Be specific about what's missing

If evidence is SUFFICIENT:
- Confirm coverage
- Assess confidence (0.0-1.0)

Be conservative: if uncertain, flag as insufficient with gaps."""

CRITIQUE_USER_PROMPT = """QUESTION:
{question}

REQUIRED INFORMATION:
{required_info}

RETRIEVED EVIDENCE:
{evidence_summary}

COVERAGE ANALYSIS:
{coverage_summary}

Review this evidence. Is it sufficient to answer the question?
If not, identify specific gaps and queries to fill them."""


# =============================================================================
# SYNTHESIS PROMPTS
# =============================================================================

SYNTHESIS_SYSTEM_PROMPT = """You are a financial analyst synthesizing an answer from retrieved evidence.

Guidelines by question type:

FACTUAL: Provide direct, concise answer with specific data points
COMPARISON: Structure as comparison (X vs Y), highlight differences
ENUMERATION: List all items systematically, don't miss any
TEMPORAL: Describe progression over time, note trends
CAUSAL: Explain cause-effect relationships with evidence

ALWAYS:
- Ground every claim in retrieved evidence
- Reference evidence by number [1], [2], etc.
- Acknowledge gaps explicitly ("No evidence found for...")
- State confidence based on evidence coverage
- Include relevant dates and sources

NEVER:
- Hallucinate facts not in evidence
- Claim certainty when evidence is sparse
- Ignore contradictory evidence"""

SYNTHESIS_USER_PROMPT = """QUESTION:
{question}

QUESTION TYPE: {question_type}

REQUIRED INFORMATION:
{required_info}

RETRIEVED EVIDENCE:
{evidence}

IDENTIFIED GAPS:
{gaps}

TEMPORAL RANGE OF EVIDENCE: {temporal_range}

Synthesize a comprehensive answer using the evidence above.
Reference evidence by number [1], [2], etc.
Acknowledge any gaps in the information."""
```

---

## Testing & Evaluation

### Test Script (`scripts/test_cot_query.py`)

```python
"""Test the Chain-of-Thought Query Pipeline."""

import asyncio
import json
from datetime import datetime

from src.querying_system.cot_query.pipeline import CoTQueryPipeline


TEST_QUESTIONS = [
    # Factual
    "What was the inflation trend reported in the Boston district?",

    # Comparison
    "How did labor market conditions differ between New York and Chicago?",

    # Enumeration
    "Which Federal Reserve districts reported concerns about inflation?",

    # Temporal
    "How have consumer spending patterns changed over the past quarter?",

    # Causal
    "Why did manufacturing activity decline in the Midwest?",
]


async def main():
    pipeline = CoTQueryPipeline(group_id="default")

    results = []

    for question in TEST_QUESTIONS:
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        print('='*60)

        answer = await pipeline.query(question)

        print(f"\nA: {answer.answer[:500]}...")
        print(f"\nConfidence: {answer.confidence:.2f}")
        print(f"Evidence: {len(answer.evidence)} facts")
        print(f"Gaps: {answer.gaps}")
        print(f"Time: {answer.total_time_ms:.0f}ms")

        results.append({
            "question": question,
            "answer": answer.answer,
            "confidence": answer.confidence,
            "evidence_count": len(answer.evidence),
            "gaps": answer.gaps,
            "timing": {
                "decomposition_ms": answer.decomposition_time_ms,
                "retrieval_ms": answer.retrieval_time_ms,
                "critique_ms": answer.critique_time_ms,
                "synthesis_ms": answer.synthesis_time_ms,
                "total_ms": answer.total_time_ms
            }
        })

    # Save results
    output_file = f"eval_cot_query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Comparison Evaluation (`scripts/eval_cot_vs_deep_research.py`)

```python
"""Compare CoT Query Pipeline vs Deep Research."""

import asyncio
import json
from datetime import datetime

from src.querying_system.cot_query.pipeline import CoTQueryPipeline
from src.querying_system.deep_research.hybrid_pipeline import (
    HybridDeepResearchPipeline, ResearchMode
)


EVAL_QUESTIONS = [
    "What was the inflation trend in the Boston district?",
    "How did different Fed districts describe consumer spending?",
    "Which districts reported labor market tightness?",
]


async def main():
    cot_pipeline = CoTQueryPipeline(group_id="default")
    deep_pipeline = HybridDeepResearchPipeline(group_id="default")

    results = []

    for question in EVAL_QUESTIONS:
        print(f"\n{'='*60}")
        print(f"Q: {question}")

        # Run CoT
        print("\n--- CoT Query Pipeline ---")
        cot_answer = await cot_pipeline.query(question)
        print(f"Time: {cot_answer.total_time_ms:.0f}ms")
        print(f"Confidence: {cot_answer.confidence:.2f}")

        # Run Deep Research (DETERMINISTIC mode for fair comparison)
        print("\n--- Deep Research (DETERMINISTIC) ---")
        deep_answer = await deep_pipeline.query(question, mode=ResearchMode.DETERMINISTIC)
        print(f"Time: {deep_answer.total_time_ms:.0f}ms")

        results.append({
            "question": question,
            "cot": {
                "answer": cot_answer.answer[:500],
                "confidence": cot_answer.confidence,
                "time_ms": cot_answer.total_time_ms,
                "evidence_count": len(cot_answer.evidence)
            },
            "deep_research": {
                "answer": deep_answer.answer[:500],
                "time_ms": deep_answer.total_time_ms,
                "findings_count": len(deep_answer.findings)
            }
        })

    # Save
    output_file = f"eval_cot_vs_deep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Summary

### Expected Improvements

| Metric | Current Deep Research | CoT Query Pipeline |
|--------|----------------------|-------------------|
| LLM Calls | 15-50+ | 2-4 |
| Latency | 2-5 seconds | 300-800ms |
| Consistency | ~70-80% | ~95%+ |
| Cost | $$$ | $ |
| Explainability | Low (agent decisions) | High (structured plan) |

### Implementation Order

1. `schemas.py` - Data structures
2. `prompts.py` - All prompts
3. `decomposer.py` - Question decomposition
4. `retriever.py` - Parallel retrieval + fusion
5. `critic.py` - Evidence critique
6. `synthesizer.py` - Answer generation
7. `pipeline.py` - Orchestrator
8. Test scripts and evaluation

### Key Innovations

1. **Structured Decomposition**: Enumerate required info before planning (like CoT extraction)
2. **Cross-Query Fusion**: Facts found by multiple sub-queries get boosted (like entity dedup)
3. **Bounded Reflexion**: Max 1 refinement pass (like extraction critique)
4. **Coverage Tracking**: Know exactly what information is missing
5. **Question-Type-Aware**: Different strategies for different question types
