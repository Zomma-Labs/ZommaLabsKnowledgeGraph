"""
V6 Researcher: Per-subquery research agent with threshold-only retrieval.

V6 CHANGES:
- REMOVED: _score_and_identify_gaps() - replaced with _filter_by_threshold()
- REMOVED: self.scoring_llm, self.scoring_and_gap
- REMOVED: imports for SCORING_AND_GAP_* prompts and format_facts_for_scoring

Each researcher handles one sub-query end-to-end:
1. Resolve entities/topics from hints
2. Retrieve facts (scoped + global in parallel)
3. Filter facts by vector score threshold (NO LLM scoring)
4. Expand if too few facts (dynamic multi-hop)
5. Entity drill-down for ENUMERATION questions
6. Synthesize a focused sub-answer
"""

import asyncio
import os
import re
import time
from typing import Optional

from src.util.llm_client import get_nano_gpt_llm, get_critique_llm
from src.querying_system.shared.schemas import SubQuery, QuestionType, EntityHint

from src.util.llm_client import get_llm
from .schemas import (
    ResearcherConfig,
    SubAnswer,
    RawFact,
    ScoredFact,
    Gap,
    GapDetectionResult,
    SubAnswerSynthesis,
    DrilldownSelection,
    VaguenessDetectionResult,
    FactFilterResult,
    FactFilterCritique,
)
from .graph_store import GraphStore
from .prompts import (
    SUB_ANSWER_SYSTEM_PROMPT,
    SUB_ANSWER_USER_PROMPT,
    GAP_DETECTION_SYSTEM_PROMPT,
    GAP_DETECTION_USER_PROMPT,
    DRILLDOWN_SYSTEM_PROMPT,
    DRILLDOWN_USER_PROMPT,
    VAGUENESS_DETECTION_SYSTEM_PROMPT,
    VAGUENESS_DETECTION_USER_PROMPT,
    REFINEMENT_SYNTHESIS_SYSTEM_PROMPT,
    REFINEMENT_SYNTHESIS_USER_PROMPT,
    FACT_FILTER_SYSTEM_PROMPT,
    FACT_FILTER_USER_PROMPT,
    FACT_FILTER_CRITIQUE_SYSTEM_PROMPT,
    FACT_FILTER_CRITIQUE_USER_PROMPT,
    format_facts_for_gap_detection,
    format_evidence_for_synthesis,
)

VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"

# Cross-source boost for facts found by multiple retrieval paths
CROSS_SOURCE_BOOST = 0.15


def log(msg: str):
    if VERBOSE:
        print(f"[Researcher] {msg}")


def extract_keywords(text: str) -> list[str]:
    """Extract keywords from text for keyword search."""
    # Simple keyword extraction - remove common words
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
        "from", "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "under", "again", "further", "then", "once", "here",
        "there", "when", "where", "why", "how", "all", "each", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own",
        "same", "so", "than", "too", "very", "just", "and", "but", "if", "or",
        "because", "until", "while", "what", "which", "who", "whom", "this",
        "that", "these", "those", "am", "it", "its", "they", "them", "their",
        "we", "us", "our", "you", "your", "he", "him", "his", "she", "her",
    }

    # Extract words, filter stop words, keep meaningful ones
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    keywords = [w for w in words if w not in stop_words]

    # Dedupe while preserving order
    seen = set()
    result = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            result.append(kw)

    return result[:10]  # Limit to 10 keywords


class Researcher:
    """
    Per-subquery research agent with threshold-only retrieval.

    V6 CHANGE: No LLM scoring - uses vector score threshold (0.65) instead.
    Handles one sub-query end-to-end with resolve -> retrieve -> filter -> expand -> synthesize.
    """

    def __init__(
        self,
        graph_store: GraphStore,
        config: Optional[ResearcherConfig] = None,
    ):
        self.graph_store = graph_store
        self.config = config or ResearcherConfig()

        # LLMs (V6: removed scoring_llm)
        self.synthesis_llm = get_critique_llm()  # Better for synthesis
        self.drilldown_llm = get_nano_gpt_llm()  # Cheap for drilldown selection
        self.fact_filter_llm = get_llm()  # gpt-5.2 for precise fact filtering

        # Structured outputs (V6: removed scoring_and_gap)
        self.sub_synthesizer = self.synthesis_llm.with_structured_output(SubAnswerSynthesis)
        self.vagueness_detector = self.drilldown_llm.with_structured_output(VaguenessDetectionResult)
        self.gap_detector = self.drilldown_llm.with_structured_output(GapDetectionResult)
        self.drilldown_selector = self.drilldown_llm.with_structured_output(DrilldownSelection)
        self.fact_filter = self.fact_filter_llm.with_structured_output(FactFilterResult)
        self.fact_filter_critic = self.fact_filter_llm.with_structured_output(FactFilterCritique)

    async def research(
        self,
        sub_query: SubQuery,
        question_context: str,
        question_type: QuestionType,
        topic_hints: Optional[list] = None,
    ) -> SubAnswer:
        """
        Research a single sub-query end-to-end.

        Returns a SubAnswer with:
        - Synthesized answer text
        - Confidence score
        - Facts used as evidence
        - Entities found
        - Timing breakdown
        """
        log(f"Researching: {sub_query.query_text}")

        # Timing
        resolution_start = time.time()

        # Step 1: Resolve entities/topics from hints
        entity_hints = [
            EntityHint(name=h, definition=f"Entity related to: {sub_query.target_info}")
            for h in sub_query.entity_hints
        ]
        raw_topic_hints = topic_hints if topic_hints is not None else (
            getattr(sub_query, "topic_hints", []) or []
        )
        topic_hints_structured = []
        for t in raw_topic_hints:
            if hasattr(t, "name") and hasattr(t, "definition"):
                topic_hints_structured.append(t)
                continue
            if isinstance(t, dict):
                name = t.get("name", "")
                definition = t.get("definition", "")
            else:
                name = str(t)
                definition = ""
            name = name.strip()
            if not name:
                continue
            if not definition:
                definition = f"Topic related to: {sub_query.target_info}"
            topic_hints_structured.append(EntityHint(name=name, definition=definition))
        # Use query_text as context for resolution
        resolved = await self.graph_store.resolve(
            entity_hints=entity_hints,
            topic_hints=topic_hints_structured,
            question_context=sub_query.query_text,
        )

        resolution_time = int((time.time() - resolution_start) * 1000)
        log(f"Resolved {len(resolved.entities)} entities, {len(resolved.topics)} topics in {resolution_time}ms")

        # Step 2: Retrieve facts (scoped + global in parallel)
        # Cache query embedding for reuse in drilldown/expansion
        retrieval_start = time.time()
        query_embedding = await self.graph_store.embed_text(sub_query.query_text)
        raw_facts = await self._retrieve_dual_path(resolved, sub_query.query_text, query_embedding)
        retrieval_time = int((time.time() - retrieval_start) * 1000)
        log(f"Retrieved {len(raw_facts)} facts in {retrieval_time}ms")

        # Step 3: Filter facts by threshold (V6: NO LLM SCORING)
        scoring_start = time.time()
        scored_facts, gaps = await self._filter_by_threshold(
            raw_facts, sub_query.target_info
        )
        scoring_time = int((time.time() - scoring_start) * 1000)
        log(f"Filtered to {len(scored_facts)} facts (threshold={self.config.relevance_threshold}), {len(gaps)} gaps in {scoring_time}ms")

        # Step 4: Dynamic expansion if too few facts AND enabled
        expansion_time = 0
        if gaps and self.config.enable_gap_expansion:
            expansion_start = time.time()
            expanded = await self._expand_for_gaps(gaps, scored_facts)
            scored_facts = self._merge_and_rescore(scored_facts, expanded)
            expansion_time = int((time.time() - expansion_start) * 1000)
            log(f"Expanded from gaps, now have {len(scored_facts)} facts in {expansion_time}ms")

        # Step 5: Entity drill-down for ENUMERATION (configurable)
        if (question_type == QuestionType.ENUMERATION and
            self.config.enable_entity_drilldown):
            drilldown_start = time.time()
            # Pass cached embedding to avoid re-computing
            drilldown_facts = await self._entity_drilldown(scored_facts, question_context, query_embedding)
            scored_facts = self._merge_and_rescore(scored_facts, drilldown_facts)
            expansion_time += int((time.time() - drilldown_start) * 1000)
            log(f"Drill-down complete, now have {len(scored_facts)} facts")

        # Step 5.5: LLM fact filter for precision (if enabled)
        llm_filter_time = 0
        if self.config.enable_llm_fact_filter:
            llm_filter_start = time.time()
            scored_facts = await self._filter_facts_with_llm(scored_facts, question_context)
            llm_filter_time = int((time.time() - llm_filter_start) * 1000)
            log(f"LLM filter complete in {llm_filter_time}ms, now have {len(scored_facts)} facts")

        # Step 6: Synthesize sub-answer
        synthesis_start = time.time()
        sub_answer = await self._synthesize(sub_query, scored_facts, question_type)
        synthesis_time = int((time.time() - synthesis_start) * 1000)
        log(f"Synthesized answer in {synthesis_time}ms")

        # Step 7: Refinement loop for vague answers (if enabled AND confidence below threshold)
        refinement_time = 0
        should_refine = (
            self.config.enable_refinement_loop and
            sub_answer.confidence < self.config.refinement_confidence_threshold
        )
        if should_refine:
            refinement_start = time.time()
            sub_answer, additional_facts = await self._refine_if_vague(
                sub_answer,
                question_context,
                scored_facts,
                question_type,
            )
            refinement_time = int((time.time() - refinement_start) * 1000)
            if refinement_time > 100:  # Only log if refinement actually happened
                log(f"Refinement completed in {refinement_time}ms")
            # Merge any additional facts
            if additional_facts:
                scored_facts = self._merge_and_rescore(scored_facts, additional_facts)
                top_k = self._get_top_k_evidence(question_type)
                sub_answer.facts_used = scored_facts[:top_k]
        elif self.config.enable_refinement_loop:
            log(f"Skipping refinement - confidence {sub_answer.confidence:.2f} >= threshold {self.config.refinement_confidence_threshold}")

        # Add timing
        sub_answer.resolution_time_ms = resolution_time
        sub_answer.retrieval_time_ms = retrieval_time
        sub_answer.scoring_time_ms = scoring_time
        sub_answer.expansion_time_ms = expansion_time + refinement_time
        sub_answer.synthesis_time_ms = synthesis_time

        return sub_answer

    async def _retrieve_dual_path(
        self,
        resolved,
        query_text: str,
        query_embedding: list[float] | None = None,
    ) -> list[RawFact]:
        """
        Always runs BOTH scoped and global search in parallel.

        This ensures we don't miss relevant facts even if resolution fails.
        Accepts pre-computed query_embedding to avoid duplicate embed calls.
        """
        # Use provided embedding or compute (should be cached by caller)
        if query_embedding is None:
            query_embedding = await self.graph_store.embed_text(query_text)

        tasks = []

        # Scoped searches (per resolved entity/topic)
        for entity in resolved.entities:
            tasks.append(self.graph_store.search_entity_facts(
                entity.resolved_name,
                query_embedding,
                self.config.scoped_threshold,
            ))

        for topic in resolved.topics:
            tasks.append(self.graph_store.search_topic_facts(
                topic.resolved_name,
                query_embedding,
                self.config.scoped_threshold,
            ))

        # Global vector search (ALWAYS run, even if entities resolved)
        if self.config.enable_global_search:
            tasks.append(self.graph_store.search_all_facts_vector(
                query_embedding,
                self.config.global_top_k,
            ))

        # Execute all in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten and handle errors
        all_facts = []
        for result in results:
            if isinstance(result, Exception):
                log(f"Retrieval error: {result}")
                continue
            all_facts.extend(result)

        # Merge and dedupe by fact_id, apply cross-source boost
        return self._merge_facts(all_facts)

    def _merge_facts(self, facts: list[RawFact]) -> list[RawFact]:
        """Merge facts by fact_id, tracking sources for cross-source boost."""
        fact_map: dict[str, RawFact] = {}
        source_map: dict[str, set[str]] = {}

        for fact in facts:
            if fact.fact_id not in fact_map:
                fact_map[fact.fact_id] = fact
                source_map[fact.fact_id] = {fact.source} if fact.source else set()
            else:
                # Keep higher score
                if fact.vector_score > fact_map[fact.fact_id].vector_score:
                    fact_map[fact.fact_id] = fact
                # Track source
                if fact.source:
                    source_map[fact.fact_id].add(fact.source)

        # Attach aggregated sources to each fact
        for fact_id, fact in fact_map.items():
            fact.found_by_sources = sorted(source_map.get(fact_id, set()))

        return list(fact_map.values())

    def _get_top_k_evidence(self, question_type: QuestionType) -> int:
        """Select evidence count based on question type."""
        if question_type == QuestionType.ENUMERATION:
            return self.config.top_k_evidence_enumeration
        return self.config.top_k_evidence

    async def _filter_by_threshold(
        self,
        facts: list[RawFact],
        target_info: str,
    ) -> tuple[list[ScoredFact], list[Gap]]:
        """
        V6: Filter facts by embedding score threshold.

        OpenAI text-embedding-3-large has ~3x better score separation than Voyage,
        enabling threshold-only retrieval without LLM scoring.

        Returns (scored_facts sorted by score, gaps to expand from).
        """
        if not facts:
            return [], []

        # Filter by threshold + apply cross-source boost
        filtered: list[ScoredFact] = []
        for f in facts:
            if f.vector_score < self.config.relevance_threshold:
                continue
            boost = 0.0
            if f.found_by_sources:
                boost = CROSS_SOURCE_BOOST * max(0, len(f.found_by_sources) - 1)
            filtered.append(
                ScoredFact.from_raw(
                    f,
                    cross_source_boost=boost,
                    final_score=f.vector_score + boost,
                )
            )

        # Sort by score descending
        filtered.sort(key=lambda f: f.final_score, reverse=True)

        # Limit to max_facts_to_score
        filtered = filtered[:self.config.max_facts_to_score]

        # Gap detection (LLM-guided), fallback to simple heuristic
        gaps: list[Gap] = []
        if self.config.enable_gap_expansion:
            if filtered:
                gaps = await self._detect_gaps(target_info, filtered)
                if gaps:
                    # Ensure we have an expansion anchor
                    for g in gaps:
                        if not g.expand_from and filtered:
                            g.expand_from = filtered[0].subject
                if not gaps:
                    # Fallback heuristic if LLM finds no gaps but evidence is thin
                    if len(filtered) < 5:
                        for fact in filtered[:3]:
                            if fact.subject:
                                gaps.append(Gap(missing="more context", expand_from=fact.subject))
            else:
                gaps = [Gap(missing=target_info, expand_from="")]

        return filtered, gaps

    async def _detect_gaps(
        self,
        target_info: str,
        facts: list[ScoredFact],
    ) -> list[Gap]:
        """Use LLM to detect information gaps from retrieved facts."""
        facts_summary = format_facts_for_gap_detection(facts, max_facts=15)

        prompt = GAP_DETECTION_USER_PROMPT.format(
            target_info=target_info,
            facts_summary=facts_summary if facts_summary else "No facts available.",
        )

        try:
            result = await asyncio.to_thread(
                self.gap_detector.invoke,
                [
                    ("system", GAP_DETECTION_SYSTEM_PROMPT),
                    ("human", prompt),
                ]
            )
            if getattr(result, "sufficient", False):
                return []

            gaps: list[Gap] = []
            for g in result.gaps:
                missing = getattr(g, "missing", "").strip()
                expand_from = getattr(g, "expand_from", "").strip()
                if missing:
                    gaps.append(Gap(missing=missing, expand_from=expand_from))
            return gaps

        except Exception as e:
            log(f"Gap detection error: {e}")
            return []

    async def _expand_for_gaps(
        self,
        gaps: list[Gap],
        current_facts: list[ScoredFact],
    ) -> list[RawFact]:
        """Expand from entities identified in gaps."""
        if not gaps:
            return []

        # Get entities to expand from
        expand_entities = set()
        for gap in gaps:
            if gap.expand_from:
                expand_entities.add(gap.expand_from)

        # Also expand from entities marked should_expand (from previous logic)
        for fact in current_facts:
            if fact.should_expand:
                expand_entities.add(fact.subject)
                expand_entities.add(fact.object)

        # Limit expansion
        expand_entities = list(expand_entities)[:self.config.drilldown_max_entities]

        if not expand_entities:
            return []

        log(f"Expanding from {len(expand_entities)} entities")

        # Get query embedding (reuse from any fact or create new)
        query_embedding = await self.graph_store.embed_text(
            " ".join([g.missing for g in gaps])
        )

        # Expand in parallel
        tasks = [
            self.graph_store.expand_from_entity(entity, query_embedding, max_facts=5)
            for entity in expand_entities
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        expanded_facts = []
        for result in results:
            if isinstance(result, Exception):
                continue
            expanded_facts.extend(result)

        return expanded_facts

    async def _entity_drilldown(
        self,
        scored_facts: list[ScoredFact],
        question_context: str,
        query_embedding: list[float] | None = None,
    ) -> list[RawFact]:
        """
        Entity drill-down for ENUMERATION questions.

        Identifies entities in current facts that might have more relevant info.
        Accepts pre-computed query_embedding to avoid duplicate embed calls.
        """
        if not scored_facts:
            return []

        # Collect unique entities from facts
        entities = set()
        for fact in scored_facts[:20]:  # Top 20 facts
            if fact.subject:
                entities.add(fact.subject)
            if fact.object:
                entities.add(fact.object)

        if not entities:
            return []

        # Pre-compute embedding if not provided (cache for reuse)
        if query_embedding is None:
            query_embedding = await self.graph_store.embed_text(question_context)

        # Use LLM to select which entities to drill down on
        entities_text = "\n".join(f"- {e}" for e in list(entities)[:30])
        prompt = DRILLDOWN_USER_PROMPT.format(
            question=question_context,
            entities=entities_text,
        )

        try:
            # Simple string list response
            result = await asyncio.to_thread(
                self.drilldown_selector.invoke,
                [
                    ("system", DRILLDOWN_SYSTEM_PROMPT),
                    ("human", prompt),
                ]
            )

            # Normalize selection to known entities
            entity_map = {e.lower(): e for e in entities}
            selected_entities = []
            for name in getattr(result, "entities", []) or []:
                key = name.lower().strip()
                if key in entity_map:
                    selected_entities.append(entity_map[key])

            # Limit
            selected_entities = selected_entities[:self.config.drilldown_max_entities]

            if not selected_entities:
                return []

            log(f"Drilling down on {len(selected_entities)} entities")

            # Expand from selected entities (reuse cached embedding)
            tasks = [
                self.graph_store.expand_from_entity(entity, query_embedding, max_facts=3)
                for entity in selected_entities
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            drilldown_facts = []
            for result in results:
                if isinstance(result, Exception):
                    continue
                drilldown_facts.extend(result)

            return drilldown_facts

        except Exception as e:
            log(f"Drilldown error: {e}")
            # Fallback to substring matching if structured output fails
            try:
                response = await asyncio.to_thread(
                    self.drilldown_llm.invoke,
                    [
                        ("system", DRILLDOWN_SYSTEM_PROMPT),
                        ("human", prompt),
                    ]
                )
                response_text = response.content if hasattr(response, 'content') else str(response)
                selected_entities = []
                for entity in entities:
                    if entity.lower() in response_text.lower():
                        selected_entities.append(entity)
                selected_entities = selected_entities[:self.config.drilldown_max_entities]
                if not selected_entities:
                    return []

                log(f"Drilling down on {len(selected_entities)} entities (fallback)")
                # Reuse cached embedding instead of re-computing
                tasks = [
                    self.graph_store.expand_from_entity(entity, query_embedding, max_facts=3)
                    for entity in selected_entities
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                drilldown_facts = []
                for result in results:
                    if isinstance(result, Exception):
                        continue
                    drilldown_facts.extend(result)
                return drilldown_facts
            except Exception:
                return []

    def _merge_and_rescore(
        self,
        existing: list[ScoredFact],
        new_facts: list[RawFact],
    ) -> list[ScoredFact]:
        """Merge new facts with existing, applying cross-source boost."""
        # Convert new facts to scored facts
        new_scored = [
            ScoredFact.from_raw(f, final_score=f.vector_score * 0.8)  # Slight penalty for expansion
            for f in new_facts
        ]

        # Merge by fact_id
        fact_map = {f.fact_id: f for f in existing}

        for fact in new_scored:
            if fact.fact_id in fact_map:
                # Boost existing fact for any new sources
                existing = fact_map[fact.fact_id]
                existing_sources = set(existing.found_by_sources)
                incoming_sources = set(fact.found_by_sources or [])
                new_sources = incoming_sources - existing_sources
                if new_sources:
                    existing.found_by_sources.extend(sorted(new_sources))
                    boost = CROSS_SOURCE_BOOST * len(new_sources)
                    existing.cross_source_boost += boost
                    existing.final_score += boost
            else:
                fact_map[fact.fact_id] = fact

        # Re-sort
        merged = list(fact_map.values())
        merged.sort(key=lambda f: f.final_score, reverse=True)

        return merged

    async def _synthesize(
        self,
        sub_query: SubQuery,
        scored_facts: list[ScoredFact],
        question_type: QuestionType,
    ) -> SubAnswer:
        """Synthesize a focused sub-answer from the facts."""
        # Get top facts for evidence
        top_k = self._get_top_k_evidence(question_type)
        top_facts = scored_facts[:top_k]

        # Format evidence
        evidence_text = format_evidence_for_synthesis(top_facts)

        prompt = SUB_ANSWER_USER_PROMPT.format(
            sub_query=sub_query.query_text,
            target_info=sub_query.target_info,
            evidence=evidence_text if evidence_text else "No relevant evidence found.",
        )

        try:
            result = await asyncio.to_thread(
                self.sub_synthesizer.invoke,
                [
                    ("system", SUB_ANSWER_SYSTEM_PROMPT),
                    ("human", prompt),
                ]
            )

            return SubAnswer(
                sub_query=sub_query.query_text,
                target_info=sub_query.target_info,
                answer=result.answer,
                confidence=result.confidence,
                facts_used=top_facts,
                entities_found=result.entities_mentioned,
            )

        except Exception as e:
            log(f"Synthesis error: {e}")
            # Fallback answer
            return SubAnswer(
                sub_query=sub_query.query_text,
                target_info=sub_query.target_info,
                answer=f"Unable to synthesize answer: {e}",
                confidence=0.0,
                facts_used=top_facts,
                entities_found=[],
            )

    async def _refine_if_vague(
        self,
        sub_answer: SubAnswer,
        question_context: str,
        current_facts: list[ScoredFact],
        question_type: QuestionType,
    ) -> tuple[SubAnswer, list[RawFact]]:
        """
        Check if answer is vague and refine with targeted searches.

        Single pass: detect all vague references, fix them in parallel, re-synthesize once.
        No looping - we refine once and exit.

        Returns (refined_answer, additional_facts_found).
        """
        # Detect vagueness by comparing answer against evidence
        vagueness = await self._detect_vagueness(
            question_context,
            sub_answer.answer,
            sub_answer.facts_used,
        )

        if not vagueness.is_vague or not vagueness.vague_references:
            return sub_answer, []

        log(f"Refinement: found {len(vagueness.vague_references)} vague references")

        # Run targeted searches for all vague references in parallel
        new_facts = await self._search_for_specifics(vagueness)

        if not new_facts:
            log("No new facts found in refinement")
            return sub_answer, []

        # Re-synthesize with new evidence
        combined_facts = self._merge_and_rescore(current_facts, new_facts)

        # Format vague references for the prompt
        vague_text = "\n".join(
            f"- \"{ref.vague_text}\": need {ref.what_is_missing}"
            for ref in vagueness.vague_references
        )

        # Format new evidence
        new_evidence = format_evidence_for_synthesis(
            [ScoredFact.from_raw(f, final_score=f.vector_score) for f in new_facts],
            max_facts=10
        )

        # Refine the answer
        prompt = REFINEMENT_SYNTHESIS_USER_PROMPT.format(
            question=question_context,
            original_answer=sub_answer.answer,
            vague_references=vague_text,
            new_evidence=new_evidence if new_evidence else "No additional specific evidence found.",
        )

        try:
            result = await asyncio.to_thread(
                self.sub_synthesizer.invoke,
                [
                    ("system", REFINEMENT_SYNTHESIS_SYSTEM_PROMPT),
                    ("human", prompt),
                ]
            )

            top_k = self._get_top_k_evidence(question_type)
            refined_answer = SubAnswer(
                sub_query=sub_answer.sub_query,
                target_info=sub_answer.target_info,
                answer=result.answer,
                confidence=result.confidence,
                facts_used=combined_facts[:top_k],
                entities_found=result.entities_mentioned,
            )

            log(f"Refined answer, new confidence: {result.confidence:.2f}")
            return refined_answer, new_facts

        except Exception as e:
            log(f"Refinement synthesis error: {e}")
            return sub_answer, []

    async def _detect_vagueness(
        self,
        question: str,
        answer: str,
        facts_used: list[ScoredFact],
    ) -> VaguenessDetectionResult:
        """Detect vague references in an answer by comparing against evidence."""
        # Format evidence so LLM can compare counts/details
        evidence_summary = format_facts_for_gap_detection(facts_used, max_facts=15)

        prompt = VAGUENESS_DETECTION_USER_PROMPT.format(
            question=question,
            evidence=evidence_summary,
            answer=answer,
        )

        try:
            result = await asyncio.to_thread(
                self.vagueness_detector.invoke,
                [
                    ("system", VAGUENESS_DETECTION_SYSTEM_PROMPT),
                    ("human", prompt),
                ]
            )
            return result

        except Exception as e:
            log(f"Vagueness detection error: {e}")
            return VaguenessDetectionResult(is_vague=False, vague_references=[], reasoning=str(e))

    async def _search_for_specifics(
        self,
        vagueness: VaguenessDetectionResult,
    ) -> list[RawFact]:
        """Run targeted searches based on vague references."""
        # Collect queries - only 1 per vague reference to limit searches
        all_queries: list[str] = []
        for ref in vagueness.vague_references:
            if ref.search_queries:
                all_queries.append(ref.search_queries[0])  # Just first query

        if not all_queries:
            return []

        # Batch embed all queries at once
        try:
            all_embeddings = await self.graph_store.batch_embed(all_queries)
        except Exception as e:
            log(f"Batch embedding error: {e}")
            return []

        # Run all vector searches in parallel
        query_to_embedding = dict(zip(all_queries, all_embeddings))
        search_tasks = [
            self.graph_store.search_all_facts_vector(
                query_to_embedding[query],
                top_k=self.config.refinement_search_top_k,
            )
            for query in all_queries
        ]

        results = await asyncio.gather(*search_tasks, return_exceptions=True)

        all_facts: list[RawFact] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                log(f"Refinement search error: {result}")
                continue
            # Mark source
            for f in result:
                f.source = f"refinement:{all_queries[i][:30]}"
            all_facts.extend(result)

        # Dedupe
        return self._merge_facts(all_facts)

    # =========================================================================
    # LLM Fact Filter (Precision Improvement)
    # =========================================================================

    async def _filter_facts_with_llm(
        self,
        facts: list[ScoredFact],
        question: str,
    ) -> list[ScoredFact]:
        """
        Use LLM to filter facts to only those directly relevant to the question.

        This dramatically improves precision by removing tangentially related facts
        that passed the embedding threshold but don't actually answer the question.
        """
        if not facts or not self.config.enable_llm_fact_filter:
            return facts

        log(f"LLM filtering {len(facts)} facts...")

        # Format facts for the prompt
        facts_text = self._format_facts_for_filter(facts)

        try:
            # Initial filter
            result = await asyncio.to_thread(
                self.fact_filter.invoke,
                [
                    ("system", FACT_FILTER_SYSTEM_PROMPT),
                    ("human", FACT_FILTER_USER_PROMPT.format(
                        question=question,
                        facts=facts_text,
                    )),
                ]
            )

            selected_indices = set(result.relevant_indices)
            log(f"Initial filter selected {len(selected_indices)} facts")

            # Critique loop (if enabled)
            if self.config.enable_fact_filter_critique and selected_indices:
                critique = await asyncio.to_thread(
                    self.fact_filter_critic.invoke,
                    [
                        ("system", FACT_FILTER_CRITIQUE_SYSTEM_PROMPT),
                        ("human", FACT_FILTER_CRITIQUE_USER_PROMPT.format(
                            question=question,
                            facts=facts_text,
                            selected_indices=sorted(selected_indices),
                        )),
                    ]
                )

                # Apply critique corrections
                if critique.missed_facts:
                    log(f"Critique found {len(critique.missed_facts)} missed facts")
                    selected_indices.update(critique.missed_facts)
                if critique.wrong_inclusions:
                    log(f"Critique found {len(critique.wrong_inclusions)} wrong inclusions")
                    selected_indices -= set(critique.wrong_inclusions)

            # Filter facts by selected indices
            filtered = [f for i, f in enumerate(facts) if i in selected_indices]
            log(f"LLM filter: {len(facts)} -> {len(filtered)} facts")

            return filtered if filtered else facts[:10]  # Fallback to top 10 if filter too aggressive

        except Exception as e:
            log(f"LLM fact filter error: {e}")
            return facts  # Return original on error

    def _format_facts_for_filter(self, facts: list[ScoredFact], max_facts: int = 50) -> str:
        """Format facts for the LLM filter prompt."""
        lines = []
        for i, fact in enumerate(facts[:max_facts]):
            lines.append(f"[{i}] {fact.subject} -> {fact.object}: {fact.content[:200]}")
        return "\n".join(lines)
