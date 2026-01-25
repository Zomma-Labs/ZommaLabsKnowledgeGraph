"""
V7 ContextBuilder: Assembles retrieved content into structured context for synthesis.

GraphRAG-aligned ordering:
1. High-relevance chunks (vector_score >= threshold)
2. Entity summaries + fact relationships
3. Topic-linked chunks
4. Lower-relevance chunks

This module handles deduplication and prioritization of retrieved content.
"""

from .schemas import (
    StructuredContext,
    RetrievedChunk,
    RetrievedFact,
    RetrievedEntity,
    ResolvedEntity,
)


class ContextBuilder:
    """
    Assembles retrieved content into structured context for LLM synthesis.

    Handles:
    - Deduplication of chunks by chunk_id (keeping highest score)
    - Deduplication of facts by fact_id
    - Splitting chunks into high/low relevance by threshold
    - Converting resolved entities to RetrievedEntity format
    - Filtering topic chunks to avoid duplication
    - Applying limits to each category
    """

    def __init__(
        self,
        high_relevance_threshold: float = 0.45,
        max_high_relevance_chunks: int = 30,
        max_facts: int = 40,
        max_topic_chunks: int = 15,
        max_low_relevance_chunks: int = 20,
    ):
        """
        Initialize the context builder.

        Args:
            high_relevance_threshold: Vector score threshold for high relevance.
                Chunks with score >= threshold go to high_relevance_chunks.
                Chunks with score < threshold go to low_relevance_chunks.
            max_high_relevance_chunks: Max chunks in high relevance section.
            max_facts: Max facts to include.
            max_topic_chunks: Max topic-related chunks.
            max_low_relevance_chunks: Max supporting context chunks.
        """
        self.high_relevance_threshold = high_relevance_threshold
        self.max_high_relevance_chunks = max_high_relevance_chunks
        self.max_facts = max_facts
        self.max_topic_chunks = max_topic_chunks
        self.max_low_relevance_chunks = max_low_relevance_chunks

    def build(
        self,
        entity_chunks: list[RetrievedChunk],
        neighbor_chunks: list[RetrievedChunk],
        facts: list[RetrievedFact],
        resolved_entities: list[ResolvedEntity],
        topic_chunks: list[RetrievedChunk],
        global_chunks: list[RetrievedChunk],
    ) -> StructuredContext:
        """
        Assemble all retrieved content into structured context.

        GraphRAG-aligned ordering:
        1. High-relevance chunks (vector_score >= threshold)
        2. Entity summaries + fact relationships
        3. Topic-linked chunks
        4. Lower-relevance chunks

        Args:
            entity_chunks: Chunks connected to resolved entities
            neighbor_chunks: Chunks from 1-hop neighbor entities
            facts: Retrieved facts with subject/object relationships
            resolved_entities: Entities resolved from the knowledge graph
            topic_chunks: Chunks connected to resolved topics
            global_chunks: Chunks from global vector search

        Returns:
            StructuredContext with organized, deduplicated content
        """
        # Step 1: Combine entity_chunks + neighbor_chunks + global_chunks
        all_chunks = entity_chunks + neighbor_chunks + global_chunks

        # Step 2: Dedupe by chunk_id, keeping highest score
        deduped_chunks = self._dedupe_chunks(all_chunks)

        # Step 3: Sort by score descending
        sorted_chunks = sorted(
            deduped_chunks,
            key=lambda c: c.vector_score,
            reverse=True
        )

        # Step 4: Split into high_relevance and low_relevance by threshold
        high_relevance = []
        low_relevance = []
        for chunk in sorted_chunks:
            if chunk.vector_score >= self.high_relevance_threshold:
                high_relevance.append(chunk)
            else:
                low_relevance.append(chunk)

        # Step 5: Convert resolved_entities to RetrievedEntity format
        # Only include entities that have summaries
        entities = self._convert_resolved_entities(resolved_entities)

        # Step 6: Dedupe facts by fact_id
        unique_facts = self._dedupe_facts(facts)

        # Step 7: Filter topic_chunks to exclude those already in entity/neighbor/global chunks
        chunk_ids_seen = {c.chunk_id for c in deduped_chunks}
        unique_topic_chunks = [
            c for c in topic_chunks
            if c.chunk_id not in chunk_ids_seen
        ]

        # Apply limits and return StructuredContext
        return StructuredContext(
            high_relevance_chunks=high_relevance[:self.max_high_relevance_chunks],
            entities=entities,
            facts=unique_facts[:self.max_facts],
            topic_chunks=unique_topic_chunks[:self.max_topic_chunks],
            low_relevance_chunks=low_relevance[:self.max_low_relevance_chunks],
        )

    def _dedupe_chunks(
        self,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """
        Deduplicate chunks by chunk_id, keeping the one with highest vector_score.

        Args:
            chunks: List of chunks potentially containing duplicates

        Returns:
            Deduplicated list of chunks
        """
        chunk_map: dict[str, RetrievedChunk] = {}

        for chunk in chunks:
            if not chunk.chunk_id:
                continue

            if chunk.chunk_id not in chunk_map:
                chunk_map[chunk.chunk_id] = chunk
            elif chunk.vector_score > chunk_map[chunk.chunk_id].vector_score:
                chunk_map[chunk.chunk_id] = chunk

        return list(chunk_map.values())

    def _dedupe_facts(
        self,
        facts: list[RetrievedFact],
    ) -> list[RetrievedFact]:
        """
        Deduplicate facts by fact_id, keeping the one with highest vector_score.

        Args:
            facts: List of facts potentially containing duplicates

        Returns:
            Deduplicated list of facts, sorted by score descending
        """
        fact_map: dict[str, RetrievedFact] = {}

        for fact in facts:
            if not fact.fact_id:
                continue

            if fact.fact_id not in fact_map:
                fact_map[fact.fact_id] = fact
            elif fact.vector_score > fact_map[fact.fact_id].vector_score:
                fact_map[fact.fact_id] = fact

        # Sort by score descending
        return sorted(
            fact_map.values(),
            key=lambda f: f.vector_score,
            reverse=True
        )

    def _convert_resolved_entities(
        self,
        resolved_entities: list[ResolvedEntity],
    ) -> list[RetrievedEntity]:
        """
        Convert ResolvedEntity objects to RetrievedEntity format.

        Only includes entities that have summaries, as entities without
        summaries provide little value in the context.

        Args:
            resolved_entities: List of resolved entities from graph

        Returns:
            List of RetrievedEntity objects with summaries
        """
        entities = []
        seen_names: set[str] = set()

        for resolved in resolved_entities:
            # Skip entities without summaries
            if not resolved.summary:
                continue

            # Skip duplicates by name
            if resolved.resolved_name in seen_names:
                continue

            seen_names.add(resolved.resolved_name)
            entities.append(RetrievedEntity(
                name=resolved.resolved_name,
                summary=resolved.summary,
                entity_type="UNKNOWN",  # ResolvedEntity doesn't have entity_type
            ))

        return entities
