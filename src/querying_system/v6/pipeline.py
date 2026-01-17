"""
V6 Pipeline: Threshold-Only Deep Research Orchestrator.

V6 CHANGES:
- Uses threshold-only retrieval (no LLM scoring)
- OpenAI text-embedding-3-large (3072 dims) for better score separation
- ~5-10x faster query times

Minimal orchestration following the deep research pattern:
1. Decompose question into sub-queries
2. Run parallel researchers (each produces a sub-answer)
3. Synthesize sub-answers into final answer

The orchestrator just orchestrates - all heavy lifting is in Researcher.
"""

import asyncio
import os
import time
from typing import Optional

from src.util.llm_client import get_critique_llm
from src.querying_system.shared.decomposer import QueryDecomposer
from src.querying_system.shared.schemas import QuestionType, EntityHint

from .schemas import (
    ResearcherConfig,
    SubAnswer,
    PipelineResult,
    Evidence,
    FinalSynthesis,
)
from .graph_store import GraphStore
from .researcher import Researcher
from .prompts import (
    FINAL_SYNTHESIS_SYSTEM_PROMPT,
    FINAL_SYNTHESIS_USER_PROMPT,
    format_sub_answers_for_final,
    get_question_type_instructions,
)

VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"


def log(msg: str):
    if VERBOSE:
        print(f"[V6Pipeline] {msg}")


class V6Pipeline:
    """
    Threshold-only deep research pipeline.

    V6 KEY CHANGE: No LLM scoring - uses vector score threshold (0.65) instead.

    Three phases:
    1. Decompose: Break question into sub-queries
    2. Research: Parallel researchers (each produces sub-answer)
    3. Synthesize: Combine sub-answers into final answer
    """

    def __init__(
        self,
        group_id: str = "default",
        config: Optional[ResearcherConfig] = None,
        max_concurrent: int = 5,
    ):
        self.group_id = group_id
        self.config = config or ResearcherConfig()
        self.max_concurrent = max_concurrent

        # Components
        self.graph_store = GraphStore(group_id)
        self.decomposer = QueryDecomposer()

        # LLM for final synthesis
        self.synthesis_llm = get_critique_llm()
        self.final_synthesizer = self.synthesis_llm.with_structured_output(FinalSynthesis)

    async def query(self, question: str) -> PipelineResult:
        """
        Process a question through the full pipeline.

        Returns PipelineResult with:
        - Final answer
        - Sub-query breakdown (the deep research pattern)
        - Evidence (deduplicated across sub-answers)
        - Timing breakdown
        """
        log(f"Processing: {question[:80]}...")
        total_start = time.time()

        # Phase 1: Decompose
        decomposition_start = time.time()
        decomposition, _ = self.decomposer.decompose(question)
        decomposition_time = int((time.time() - decomposition_start) * 1000)

        log(f"Decomposed into {len(decomposition.sub_queries)} sub-queries "
            f"(type={decomposition.question_type.value}) in {decomposition_time}ms")

        # Phase 2: Parallel sub-query research
        research_start = time.time()
        global_topic_hints = list(decomposition.topic_hints or [])
        topic_definitions = {
            h.name.lower(): h.definition
            for h in global_topic_hints
            if hasattr(h, "name")
        }
        sub_answers = await self._research_parallel(
            decomposition.sub_queries,
            question_context=question,
            question_type=decomposition.question_type,
            global_topic_hints=global_topic_hints,
            topic_definitions=topic_definitions,
        )
        research_time = int((time.time() - research_start) * 1000)

        log(f"Researched {len(sub_answers)} sub-queries in {research_time}ms")

        # Phase 3: Final synthesis
        synthesis_start = time.time()
        final = await self._synthesize_final(
            question=question,
            question_type=decomposition.question_type,
            sub_answers=sub_answers,
            required_info=decomposition.required_info,
        )
        synthesis_time = int((time.time() - synthesis_start) * 1000)

        log(f"Final synthesis in {synthesis_time}ms")

        # Collect evidence from all sub-answers
        evidence = self._collect_evidence(sub_answers)

        total_time = int((time.time() - total_start) * 1000)
        log(f"Total pipeline time: {total_time}ms")

        return PipelineResult(
            question=question,
            answer=final.answer,
            confidence=final.confidence,
            sub_answers=sub_answers,
            evidence=evidence,
            question_type=decomposition.question_type.value,
            gaps=final.gaps,
            decomposition_time_ms=decomposition_time,
            research_time_ms=research_time,
            synthesis_time_ms=synthesis_time,
        )

    async def _research_parallel(
        self,
        sub_queries,
        question_context: str,
        question_type: QuestionType,
        global_topic_hints: list[EntityHint],
        topic_definitions: dict[str, str],
    ) -> list[SubAnswer]:
        """
        Run researchers in parallel with semaphore limiting.

        Each researcher handles one sub-query end-to-end and produces a sub-answer.
        """
        if not sub_queries:
            # No sub-queries - create a default one from the question
            from src.querying_system.shared.schemas import SubQuery
            sub_queries = [SubQuery(
                query_text=question_context[:100],
                target_info="Answer the question",
                entity_hints=[],
                topic_hints=[h.name if hasattr(h, "name") else h for h in global_topic_hints],
            )]

        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def bounded_research(sq) -> SubAnswer:
            async with semaphore:
                researcher = Researcher(self.graph_store, self.config)
                raw_topic_hints = getattr(sq, "topic_hints", None) or []
                if raw_topic_hints:
                    topic_hints = [
                        EntityHint(
                            name=t,
                            definition=topic_definitions.get(
                                t.lower(), f"Topic related to: {sq.target_info}"
                            ),
                        )
                        for t in raw_topic_hints
                    ]
                else:
                    topic_hints = global_topic_hints
                return await researcher.research(
                    sq,
                    question_context,
                    question_type,
                    topic_hints=topic_hints,
                )

        # Run all in parallel
        sub_answers = await asyncio.gather(*[
            bounded_research(sq) for sq in sub_queries
        ])

        return list(sub_answers)

    async def _synthesize_final(
        self,
        question: str,
        question_type: QuestionType,
        sub_answers: list[SubAnswer],
        required_info: list[str],
    ) -> FinalSynthesis:
        """
        Combine sub-answers into final answer.

        This is the key innovation from deep research: we combine
        synthesized sub-answers, not raw facts.
        """
        # Format sub-answers
        sub_answers_text = format_sub_answers_for_final(sub_answers)

        # Identify gaps
        gap_notes = self._identify_coverage_gaps(sub_answers, required_info)

        # Type-specific instructions
        type_instructions = get_question_type_instructions(question_type.value)

        prompt = FINAL_SYNTHESIS_USER_PROMPT.format(
            question=question,
            question_type=f"{question_type.value.upper()}\n\nInstructions: {type_instructions}",
            sub_answers=sub_answers_text,
            gap_notes=gap_notes,
        )

        try:
            result = await asyncio.to_thread(
                self.final_synthesizer.invoke,
                [
                    ("system", FINAL_SYNTHESIS_SYSTEM_PROMPT),
                    ("human", prompt),
                ]
            )
            return result

        except Exception as e:
            log(f"Final synthesis error: {e}")
            # Fallback: concatenate sub-answers
            fallback_answer = "\n\n".join([
                f"**{sa.sub_query}**: {sa.answer}"
                for sa in sub_answers
            ])
            return FinalSynthesis(
                answer=fallback_answer,
                confidence=0.5,
                gaps=["Synthesis failed, showing raw sub-answers"],
            )

    def _identify_coverage_gaps(
        self,
        sub_answers: list[SubAnswer],
        required_info: list[str],
    ) -> str:
        """Identify which required info items may not be covered."""
        if not required_info:
            return ""

        # Simple heuristic: check if required info keywords appear in sub-answers
        all_answer_text = " ".join(sa.answer.lower() for sa in sub_answers)

        uncovered = []
        for info in required_info:
            # Check if key words from required info appear
            key_words = info.lower().split()[:3]  # First 3 words
            if not any(kw in all_answer_text for kw in key_words):
                uncovered.append(info)

        if not uncovered:
            return ""

        return f"**Note**: The following information may not be fully covered:\n" + \
               "\n".join(f"- {info}" for info in uncovered)

    def _collect_evidence(self, sub_answers: list[SubAnswer]) -> list[Evidence]:
        """Collect and deduplicate evidence from all sub-answers."""
        seen_fact_ids = set()
        evidence = []

        for sa in sub_answers:
            for fact in sa.facts_used:
                if fact.fact_id in seen_fact_ids:
                    continue
                seen_fact_ids.add(fact.fact_id)

                evidence.append(Evidence(
                    fact_id=fact.fact_id,
                    content=fact.content,
                    subject=fact.subject,
                    edge_type=fact.edge_type,
                    object=fact.object,
                    source_chunk=fact.chunk_id,
                    chunk_header=fact.chunk_header,
                    source_doc=fact.doc_id,
                    document_date=fact.document_date,
                    score=fact.final_score,
                ))

        # Sort by score
        evidence.sort(key=lambda e: e.score, reverse=True)

        return evidence


# =============================================================================
# Convenience function
# =============================================================================

async def query_v6(
    question: str,
    group_id: str = "default",
    config: Optional[ResearcherConfig] = None,
) -> PipelineResult:
    """
    Convenience function to run a single query through V6 pipeline.

    V6 uses threshold-only retrieval (no LLM scoring) for ~5-10x faster queries.

    Example:
        result = await query_v6("What economic conditions did Boston report?")
        print(result.answer)
    """
    pipeline = V6Pipeline(group_id=group_id, config=config)
    return await pipeline.query(question)
