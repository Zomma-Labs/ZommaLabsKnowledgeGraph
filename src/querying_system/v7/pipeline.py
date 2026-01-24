"""
V7 Pipeline: Main orchestrator for the GraphRAG-aligned query pipeline.

V7 Design Principles:
- Always decompose questions (even simple ones)
- Parallel sub-query research with semaphore limiting
- Final synthesis merges sub-answers with question-type formatting
- Timing data at every phase for analysis

Pipeline Flow:
1. Decomposition: Break question into sub-queries with entity/topic hints
2. Parallel Research: Each sub-query researched independently (respects max_concurrent)
3. Final Synthesis: Merge sub-answers into coherent final answer
"""

import asyncio
import os
import time
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI

from src.querying_system.shared.decomposer import QueryDecomposer
from src.querying_system.shared.schemas import (
    QueryDecomposition,
    QuestionType,
    SubQuery,
    EntityHint,
)

from .schemas import (
    V7Config,
    SubAnswer,
    PipelineResult,
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
        print(f"[V7 Pipeline] {msg}")


class V7Pipeline:
    """
    V7 GraphRAG-aligned Query Pipeline.

    Orchestrates:
    1. Query decomposition (always runs)
    2. Parallel sub-query research (respects max_concurrent)
    3. Final answer synthesis (question-type-aware)

    Features:
    - Chunk-centric retrieval via GraphStore
    - Wide-net resolution (one hint -> many entities)
    - 1-hop expansion for related context
    - Gemini-3-pro for synthesis
    """

    def __init__(
        self,
        group_id: str = "default",
        config: Optional[V7Config] = None,
        max_concurrent: int = 5,
    ):
        """
        Initialize the V7 pipeline.

        Args:
            group_id: Multi-tenant group identifier
            config: V7 configuration (creates default if None)
            max_concurrent: Max concurrent sub-query research tasks
        """
        self.config = config or V7Config(group_id=group_id)
        self.group_id = self.config.group_id
        self.max_concurrent = max_concurrent

        # Components
        self.graph_store = GraphStore(group_id=self.group_id)

        # Decomposition LLM (use gemini-3-flash-preview from config - fast)
        decomp_llm = ChatGoogleGenerativeAI(
            model=self.config.decomposition_model,
            temperature=0,
        )
        self.decomposer = QueryDecomposer(llm=decomp_llm)

        # Final synthesis LLM - use OpenAI for higher quality synthesis
        from langchain_openai import ChatOpenAI
        self.synthesis_llm = ChatOpenAI(
            model=self.config.synthesis_model,
            temperature=0,
        )
        self.final_synthesizer = self.synthesis_llm.with_structured_output(FinalSynthesis)

    async def query(self, question: str) -> PipelineResult:
        """
        Process a question through the full pipeline.

        Args:
            question: User's natural language question

        Returns:
            PipelineResult with answer, sub-answers, and timing data
        """
        log(f"Processing question: {question[:80]}...")

        # Phase 1: Decompose the question
        decomposition, decomposition_time_ms = self.decomposer.decompose(question)
        log(f"Decomposition complete: {len(decomposition.sub_queries)} sub-queries")

        # Phase 2: Run parallel research
        sub_answers, resolution_time_ms, retrieval_time_ms = await self._research_parallel(
            decomposition=decomposition,
            question=question,
        )
        log(f"Research complete: {len(sub_answers)} sub-answers")

        # Phase 3: Merge sub-answers into final answer
        synthesis_start = time.time()
        final_answer, confidence = await self._merge_answers(
            question=question,
            sub_answers=sub_answers,
            question_type=decomposition.question_type,
        )
        synthesis_time_ms = int((time.time() - synthesis_start) * 1000)
        log(f"Synthesis complete in {synthesis_time_ms}ms")

        return PipelineResult(
            question=question,
            answer=final_answer,
            confidence=confidence,
            sub_answers=sub_answers,
            question_type=decomposition.question_type.value,
            decomposition_time_ms=decomposition_time_ms,
            resolution_time_ms=resolution_time_ms,
            retrieval_time_ms=retrieval_time_ms,
            synthesis_time_ms=synthesis_time_ms,
        )

    async def _research_parallel(
        self,
        decomposition: QueryDecomposition,
        question: str,
    ) -> tuple[list[SubAnswer], int, int]:
        """
        Run sub-query research in parallel with concurrency limiting.

        Args:
            decomposition: Decomposed question with sub-queries and hints
            question: Original question for context

        Returns:
            Tuple of (sub_answers, total_resolution_time_ms, total_retrieval_time_ms)
        """
        sub_queries = decomposition.sub_queries

        # Create default sub-query if none provided
        if not sub_queries:
            log("No sub-queries from decomposition, creating default")
            sub_queries = [
                SubQuery(
                    query_text=question[:100],
                    target_info="Answer to the question",
                    entity_hints=[
                        h.name if isinstance(h, EntityHint) else h
                        for h in (decomposition.entity_hints or [])
                    ],
                    topic_hints=[
                        h.name if isinstance(h, EntityHint) else h
                        for h in (decomposition.topic_hints or [])
                    ],
                )
            ]

        # Build topic hints from decomposition's topic_definitions
        topic_hints = decomposition.topic_hints or []

        # Create semaphore for concurrency limiting
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def research_with_semaphore(sub_query: SubQuery) -> SubAnswer:
            async with semaphore:
                researcher = Researcher(
                    graph_store=self.graph_store,
                    config=self.config,
                )
                return await researcher.research(
                    sub_query=sub_query,
                    question_context=question,
                    question_type=decomposition.question_type,
                    topic_hints=topic_hints,
                )

        # Run all sub-queries in parallel (respecting semaphore)
        tasks = [research_with_semaphore(sq) for sq in sub_queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results and timing
        sub_answers = []
        total_resolution_time_ms = 0
        total_retrieval_time_ms = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                log(f"Sub-query {i} failed: {result}")
                # Create fallback sub-answer
                sub_answers.append(SubAnswer(
                    sub_query=sub_queries[i].query_text,
                    target_info=sub_queries[i].target_info,
                    answer=f"Error during research: {result}",
                    confidence=0.0,
                ))
            else:
                sub_answers.append(result)
                total_resolution_time_ms += result.resolution_time_ms
                total_retrieval_time_ms += result.retrieval_time_ms

        return sub_answers, total_resolution_time_ms, total_retrieval_time_ms

    async def _merge_answers(
        self,
        question: str,
        sub_answers: list[SubAnswer],
        question_type: QuestionType,
    ) -> tuple[str, float]:
        """
        Merge sub-answers into a coherent final answer.

        Uses Gemini-3-pro with question-type-specific formatting instructions.

        Args:
            question: Original question
            sub_answers: List of SubAnswer objects from research
            question_type: Classification for formatting

        Returns:
            Tuple of (final_answer, confidence)
        """
        if not sub_answers:
            return "No information was found to answer this question.", 0.0

        # Check if all sub-answers are errors
        valid_answers = [sa for sa in sub_answers if sa.confidence > 0.0]
        if not valid_answers:
            return "All research attempts failed. Please try again.", 0.0

        # Format sub-answers for the prompt
        sub_answers_text = format_sub_answers_for_final(sub_answers)

        # Get type-specific instructions
        type_instructions = get_question_type_instructions(question_type.value)

        # Build the prompt
        prompt = FINAL_SYNTHESIS_USER_PROMPT.format(
            question=question,
            question_type=question_type.value,
            sub_answers=sub_answers_text,
            type_instructions=type_instructions,
        )

        try:
            result = await asyncio.to_thread(
                self.final_synthesizer.invoke,
                [
                    ("system", FINAL_SYNTHESIS_SYSTEM_PROMPT),
                    ("human", prompt),
                ]
            )

            return result.answer, result.confidence

        except Exception as e:
            log(f"Final synthesis error: {e}")
            # Fallback: concatenate sub-answers
            fallback_answer = self._create_fallback_answer(sub_answers, question_type)
            # Average confidence of valid answers
            avg_confidence = (
                sum(sa.confidence for sa in valid_answers) / len(valid_answers)
                if valid_answers else 0.0
            )
            return fallback_answer, avg_confidence * 0.8  # Reduce confidence for fallback

    def _create_fallback_answer(
        self,
        sub_answers: list[SubAnswer],
        question_type: QuestionType,
    ) -> str:
        """
        Create a simple concatenated answer as fallback.

        Args:
            sub_answers: List of SubAnswer objects
            question_type: For minimal formatting

        Returns:
            Concatenated answer string
        """
        valid_answers = [sa for sa in sub_answers if sa.confidence > 0.0 and sa.answer]

        if not valid_answers:
            return "Unable to synthesize an answer from the available information."

        if len(valid_answers) == 1:
            return valid_answers[0].answer

        # Simple concatenation with headers
        parts = []
        for i, sa in enumerate(valid_answers, 1):
            parts.append(f"**Finding {i}** ({sa.sub_query}):\n{sa.answer}")

        return "\n\n".join(parts)


# =============================================================================
# Convenience Function
# =============================================================================

async def query_v7(
    question: str,
    group_id: str = "default",
    config: Optional[V7Config] = None,
) -> PipelineResult:
    """
    Convenience function to query the V7 pipeline.

    Creates a V7Pipeline instance and runs the query.

    Args:
        question: User's natural language question
        group_id: Multi-tenant group identifier
        config: Optional V7Config (creates default if None)

    Returns:
        PipelineResult with answer, sub-answers, and timing data

    Example:
        result = await query_v7("What economic conditions did Boston report?")
        print(result.answer)
    """
    pipeline = V7Pipeline(group_id=group_id, config=config)
    return await pipeline.query(question)
