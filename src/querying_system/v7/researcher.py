"""
V7 Researcher: Per-sub-query research agent with chunk-centric retrieval.

V7 Design Principles:
- Chunk-centric retrieval (EpisodicNodes as primary context)
- Wide-net entity/topic resolution (one-to-many)
- 1-hop expansion for related entities
- Parallel execution wherever possible
- Gemini-3-pro for synthesis

The Researcher handles one sub-query end-to-end:
1. Resolution: Entities + Topics (parallel)
2. Retrieval: Entity chunks, facts, 1-hop neighbors, topic chunks, global search (parallel)
3. Context building: Assemble structured context
4. Synthesis: Generate answer with Gemini-3-pro
"""

import asyncio
import os
import time
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI

from src.querying_system.shared.schemas import SubQuery, QuestionType, EntityHint

from .schemas import (
    V7Config,
    ResolvedContext,
    ResolvedEntity,
    ResolvedTopic,
    StructuredContext,
    SubAnswer,
    SubAnswerSynthesis,
)
from .graph_store import GraphStore
from .context_builder import ContextBuilder
from .prompts import (
    SUB_ANSWER_SYSTEM_PROMPT,
    SUB_ANSWER_USER_PROMPT,
)

VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"


def log(msg: str):
    if VERBOSE:
        print(f"[V7 Researcher] {msg}")


class Researcher:
    """
    V7 Per-sub-query research agent with chunk-centric retrieval.

    Handles one sub-query end-to-end with:
    - Parallel entity/topic resolution
    - Parallel chunk + fact retrieval
    - 1-hop neighbor expansion
    - Structured context assembly
    - Gemini-3-pro synthesis
    """

    def __init__(
        self,
        graph_store: GraphStore,
        config: Optional[V7Config] = None,
    ):
        self.graph_store = graph_store
        self.config = config or V7Config()

        # Context builder with config-driven limits
        self.context_builder = ContextBuilder(
            high_relevance_threshold=self.config.high_relevance_threshold,
            max_high_relevance_chunks=self.config.max_high_relevance_chunks,
            max_facts=self.config.max_facts,
            max_topic_chunks=self.config.max_topic_chunks,
            max_low_relevance_chunks=self.config.max_low_relevance_chunks,
        )

        # Synthesis LLM - use OpenAI for higher quality synthesis
        from langchain_openai import ChatOpenAI
        self.synthesis_llm = ChatOpenAI(
            model=self.config.synthesis_model,
            temperature=0,
        )
        self.structured_synthesizer = self.synthesis_llm.with_structured_output(
            SubAnswerSynthesis
        )

    async def research(
        self,
        sub_query: SubQuery,
        question_context: str,
        question_type: QuestionType,
        topic_hints: Optional[list[EntityHint]] = None,
    ) -> SubAnswer:
        """
        Research a sub-query and produce a synthesized answer.

        Args:
            sub_query: The focused sub-query to research
            question_context: The original question for context
            question_type: Classification of the question type
            topic_hints: Additional topic hints from decomposition

        Returns:
            SubAnswer with synthesized answer, confidence, and timing data
        """
        log(f"Researching: {sub_query.query_text}")

        # Step 1: Resolution (entities + topics in parallel)
        resolution_start = time.time()
        resolved = await self._resolve(
            sub_query=sub_query,
            question_context=question_context,
            topic_hints=topic_hints,
        )
        resolution_time_ms = int((time.time() - resolution_start) * 1000)
        log(f"Resolved {len(resolved.entities)} entities, {len(resolved.topics)} topics in {resolution_time_ms}ms")

        # Step 2: Retrieval (chunks, facts, neighbors, topics in parallel)
        retrieval_start = time.time()
        context = await self._retrieve(
            resolved=resolved,
            query_text=sub_query.query_text,
        )
        retrieval_time_ms = int((time.time() - retrieval_start) * 1000)
        log(f"Retrieved context in {retrieval_time_ms}ms")

        # Step 3: Synthesis
        synthesis_start = time.time()
        answer, confidence, entities_mentioned = await self._synthesize(
            sub_query=sub_query,
            context=context,
        )
        synthesis_time_ms = int((time.time() - synthesis_start) * 1000)
        log(f"Synthesized answer (confidence: {confidence:.2f}) in {synthesis_time_ms}ms")

        return SubAnswer(
            sub_query=sub_query.query_text,
            target_info=sub_query.target_info,
            answer=answer,
            confidence=confidence,
            context=context,
            entities_found=entities_mentioned,
            resolution_time_ms=resolution_time_ms,
            retrieval_time_ms=retrieval_time_ms,
            synthesis_time_ms=synthesis_time_ms,
        )

    # =========================================================================
    # Step 1: Resolution
    # =========================================================================

    async def _resolve(
        self,
        sub_query: SubQuery,
        question_context: str,
        topic_hints: Optional[list[EntityHint]] = None,
    ) -> ResolvedContext:
        """
        Resolve entity and topic hints to graph nodes.

        Runs entity and topic resolution in parallel for efficiency.

        Args:
            sub_query: Contains entity_hints and topic_hints from decomposition
            question_context: The original question for context
            topic_hints: Additional topic hints from pipeline-level decomposition

        Returns:
            ResolvedContext with resolved entities and topics
        """
        # Build entity hints from sub_query
        entity_hint_names = sub_query.entity_hints or []

        # Build topic hints from both sources (sub_query + pipeline-level)
        topic_hint_names = list(sub_query.topic_hints or [])
        if topic_hints:
            for hint in topic_hints:
                name = hint.name if hasattr(hint, 'name') else str(hint)
                if name and name not in topic_hint_names:
                    topic_hint_names.append(name)

        # Run resolution in parallel
        entity_task = self.graph_store.resolve_entities(
            hints=entity_hint_names,
            question_context=question_context,
            threshold=self.config.entity_threshold,
        )

        topic_task = self.graph_store.resolve_topics(
            hints=topic_hint_names,
            question_context=question_context,
            threshold=self.config.topic_threshold,
        )

        entities, topics = await asyncio.gather(
            entity_task,
            topic_task,
            return_exceptions=True,
        )

        # Handle errors gracefully
        if isinstance(entities, Exception):
            log(f"Entity resolution error: {entities}")
            entities = []
        if isinstance(topics, Exception):
            log(f"Topic resolution error: {topics}")
            topics = []

        return ResolvedContext(
            entities=entities,
            topics=topics,
        )

    # =========================================================================
    # Step 2: Retrieval
    # =========================================================================

    async def _retrieve(
        self,
        resolved: ResolvedContext,
        query_text: str,
    ) -> StructuredContext:
        """
        Retrieve chunks, facts, and related content from the graph.

        Executes multiple retrieval paths in parallel:
        - Entity chunks (per resolved entity)
        - Entity facts (per resolved entity)
        - 1-hop neighbors (if enabled)
        - Topic chunks (per resolved topic)
        - Global search (if enabled)

        Args:
            resolved: Resolved entities and topics from resolution phase
            query_text: The query text for embedding

        Returns:
            StructuredContext with organized, deduplicated content
        """
        # Embed query once for reuse
        query_embedding = await self.graph_store.embed_text(query_text)

        # Collect all retrieval tasks
        entity_chunk_tasks = []
        entity_fact_tasks = []
        neighbor_tasks = []

        for entity in resolved.entities:
            # Entity chunks
            entity_chunk_tasks.append(
                self.graph_store.get_entity_chunks(
                    entity_name=entity.resolved_name,
                    query_embedding=query_embedding,
                    threshold=self.config.entity_threshold,
                )
            )

            # Entity facts
            entity_fact_tasks.append(
                self.graph_store.get_entity_facts(
                    entity_name=entity.resolved_name,
                    query_embedding=query_embedding,
                    threshold=self.config.entity_threshold,
                )
            )

            # 1-hop neighbors (if enabled)
            if self.config.enable_1hop_expansion:
                neighbor_tasks.append(
                    self.graph_store.get_1hop_neighbors(
                        entity_name=entity.resolved_name,
                    )
                )

        # Topic chunk tasks
        topic_chunk_tasks = []
        for topic in resolved.topics:
            topic_chunk_tasks.append(
                self.graph_store.get_topic_chunks(
                    topic_name=topic.resolved_name,
                )
            )

        # Global search task (if enabled)
        global_task = None
        if self.config.enable_global_search:
            global_task = self.graph_store.global_chunk_search(
                query_embedding=query_embedding,
                top_k=self.config.global_search_top_k,
            )

        # Execute all tasks in parallel
        all_tasks = (
            entity_chunk_tasks +
            entity_fact_tasks +
            neighbor_tasks +
            topic_chunk_tasks +
            ([global_task] if global_task else [])
        )

        if not all_tasks:
            # No resolved entities or topics - fall back to global search only
            if global_task:
                global_results = await global_task
                return self.context_builder.build(
                    entity_chunks=[],
                    neighbor_chunks=[],
                    facts=[],
                    resolved_entities=[],
                    topic_chunks=[],
                    global_chunks=global_results if not isinstance(global_results, Exception) else [],
                )
            return StructuredContext()

        results = await asyncio.gather(*all_tasks, return_exceptions=True)

        # Parse results back into categories
        idx = 0
        num_entities = len(resolved.entities)

        # Entity chunks
        entity_chunks = []
        for i in range(num_entities):
            result = results[idx + i]
            if not isinstance(result, Exception):
                entity_chunks.extend(result)
            else:
                log(f"Entity chunk retrieval error: {result}")
        idx += num_entities

        # Entity facts
        entity_facts = []
        for i in range(num_entities):
            result = results[idx + i]
            if not isinstance(result, Exception):
                entity_facts.extend(result)
            else:
                log(f"Entity fact retrieval error: {result}")
        idx += num_entities

        # 1-hop neighbors
        all_neighbors = []
        if self.config.enable_1hop_expansion:
            for i in range(num_entities):
                result = results[idx + i]
                if not isinstance(result, Exception):
                    all_neighbors.extend(result)
                else:
                    log(f"1-hop neighbor retrieval error: {result}")
            idx += num_entities

        # Topic chunks
        topic_chunks = []
        for i in range(len(resolved.topics)):
            result = results[idx + i]
            if not isinstance(result, Exception):
                topic_chunks.extend(result)
            else:
                log(f"Topic chunk retrieval error: {result}")
        idx += len(resolved.topics)

        # Global chunks
        global_chunks = []
        if self.config.enable_global_search:
            result = results[idx]
            if not isinstance(result, Exception):
                global_chunks = result
            else:
                log(f"Global search error: {result}")

        # Get neighbor chunks if we have neighbors
        neighbor_chunks = []
        if all_neighbors and self.config.enable_1hop_expansion:
            # Dedupe neighbor names
            seen = set()
            unique_neighbors = []
            for n in all_neighbors:
                if n.name not in seen:
                    seen.add(n.name)
                    unique_neighbors.append(n)

            # Limit neighbors
            neighbor_names = [n.name for n in unique_neighbors[:10]]

            neighbor_chunks = await self.graph_store.get_neighbor_chunks(
                neighbor_names=neighbor_names,
                query_embedding=query_embedding,
            )
            log(f"Retrieved {len(neighbor_chunks)} chunks from {len(neighbor_names)} neighbors")

        # Build structured context
        return self.context_builder.build(
            entity_chunks=entity_chunks,
            neighbor_chunks=neighbor_chunks,
            facts=entity_facts,
            resolved_entities=resolved.entities,
            topic_chunks=topic_chunks,
            global_chunks=global_chunks,
        )

    # =========================================================================
    # Step 3: Synthesis
    # =========================================================================

    async def _synthesize(
        self,
        sub_query: SubQuery,
        context: StructuredContext,
    ) -> tuple[str, float, list[str]]:
        """
        Synthesize an answer from the structured context.

        Args:
            sub_query: The sub-query being answered
            context: Structured context with chunks, facts, entities

        Returns:
            Tuple of (answer_text, confidence, entities_mentioned)
        """
        # Convert context to prompt text
        context_text = context.to_prompt_text()

        # Check for empty context
        if context_text == "No context available.":
            log("No context available - returning fallback answer")
            return (
                f"Insufficient information available to answer: {sub_query.target_info}",
                0.1,
                [],
            )

        # Format the prompt
        prompt = SUB_ANSWER_USER_PROMPT.format(
            sub_query=sub_query.query_text,
            target_info=sub_query.target_info,
            context=context_text,
        )

        try:
            result = await asyncio.to_thread(
                self.structured_synthesizer.invoke,
                [
                    ("system", SUB_ANSWER_SYSTEM_PROMPT),
                    ("human", prompt),
                ]
            )

            return (
                result.answer,
                result.confidence,
                result.entities_mentioned or [],
            )

        except Exception as e:
            log(f"Synthesis error: {e}")
            # Fallback answer
            return (
                f"Unable to synthesize answer: {e}",
                0.0,
                [],
            )
