"""
Phase 2: Entity and Topic Resolution.

Resolves extracted entity/topic hints to actual graph nodes.
Uses vector search for candidates + LLM verification.

Key feature: Hints now include definitions for better embedding matching.
When searching, we embed "name: definition" which matches the stored
entity embeddings format ("name: summary").
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Union

from pydantic import BaseModel, Field

from src.util.services import get_services
from src.util.llm_client import get_nano_gpt_llm
from src.querying_system.shared.schemas import EntityHint

VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"


def log(msg: str):
    if VERBOSE:
        print(f"[Resolver] {msg}")


class ResolvedNode(BaseModel):
    """A resolved graph node."""
    name: str = Field(..., description="Node name in the graph")
    node_type: str = Field(..., description="EntityNode or TopicNode")
    match_reason: str = Field(..., description="Why this matches the query term")


class ResolutionResult(BaseModel):
    """LLM output for resolution verification."""
    resolved_nodes: list[ResolvedNode] = Field(
        default_factory=list,
        description="Nodes that match the query term"
    )
    no_match: bool = Field(
        default=False,
        description="True if none of the candidates match"
    )


@dataclass
class ResolvedEntities:
    """Result of resolution phase."""
    entity_nodes: list[str]  # Names of resolved EntityNodes
    topic_nodes: list[str]   # Names of resolved TopicNodes


RESOLUTION_SYSTEM_PROMPT = """You are resolving query terms to knowledge graph nodes.

Given a query term, the original question context, and candidate nodes from the graph, determine which candidates match.

RULES:
- Use the question context to understand what the query term refers to
- Match based on semantic equivalence in the context of the question
- Generic/plural terms should match all relevant specific instances
- A term can match multiple candidates
- If no candidates match the query term in context, set no_match to true
- Include match_reason explaining why each node matches
"""

RESOLUTION_USER_PROMPT = """QUESTION CONTEXT: {question}

QUERY TERM: {term}

CANDIDATE NODES FROM GRAPH:
{candidates}

Which candidates match the query term in the context of this question? Return all that apply."""


class Resolver:
    """
    Resolves extracted entities/topics to actual graph nodes.

    Flow:
    1. Query graph for candidate nodes matching the term
    2. LLM verifies which candidates actually match
    3. Return resolved node names for downstream search
    """

    def __init__(self, group_id: str = "default"):
        self.group_id = group_id
        self.services = get_services()
        self.llm = get_nano_gpt_llm()
        self.structured_resolver = self.llm.with_structured_output(ResolutionResult)

    async def resolve(
        self,
        entity_hints: Union[list[str], list[EntityHint]],
        topic_hints: Union[list[str], list[EntityHint]],
        question: str = "",
        top_k_candidates: int = 20
    ) -> ResolvedEntities:
        """
        Resolve entity and topic hints to actual graph nodes.

        Accepts either:
        - list[str]: Plain names (backward compatible)
        - list[EntityHint]: Names with definitions (preferred for better matching)
        """
        log(f"Resolving {len(entity_hints)} entities, {len(topic_hints)} topics")

        # Run entity and topic resolution in parallel
        entity_task = self._resolve_entities(entity_hints, question, top_k_candidates)
        topic_task = self._resolve_topics(topic_hints, question, top_k_candidates)

        entity_nodes, topic_nodes = await asyncio.gather(entity_task, topic_task)

        log(f"Resolved to {len(entity_nodes)} entities, {len(topic_nodes)} topics")

        return ResolvedEntities(
            entity_nodes=entity_nodes,
            topic_nodes=topic_nodes
        )

    def _hint_to_embed_text(self, hint: Union[str, EntityHint]) -> tuple[str, str]:
        """
        Convert hint to embed text and display name.

        Returns (embed_text, display_name)
        - For EntityHint: embed "name: definition", display "name"
        - For str: embed and display the string as-is
        """
        if isinstance(hint, EntityHint):
            return f"{hint.name}: {hint.definition}", hint.name
        return hint, hint

    async def _resolve_entities(
        self, hints: Union[list[str], list[EntityHint]], question: str, top_k: int
    ) -> list[str]:
        """Resolve entity hints to EntityNode names."""
        if not hints:
            return []

        all_resolved = []
        for hint in hints:
            embed_text, display_name = self._hint_to_embed_text(hint)
            candidates = await self._get_entity_candidates(embed_text, top_k)
            if not candidates:
                log(f"No entity candidates for: {display_name}")
                continue

            resolved = await self._verify_candidates(display_name, candidates, "EntityNode", question)
            all_resolved.extend(resolved)

        return list(set(all_resolved))

    async def _resolve_topics(
        self, hints: Union[list[str], list[EntityHint]], question: str, top_k: int
    ) -> list[str]:
        """Resolve topic hints to TopicNode names."""
        if not hints:
            return []

        all_resolved = []
        for hint in hints:
            embed_text, display_name = self._hint_to_embed_text(hint)
            candidates = await self._get_topic_candidates(embed_text, top_k)
            if not candidates:
                log(f"No topic candidates for: {display_name}")
                continue

            resolved = await self._verify_candidates(display_name, candidates, "TopicNode", question)
            all_resolved.extend(resolved)

        return list(set(all_resolved))

    async def _get_entity_candidates(self, embed_text: str, top_k: int) -> list[str]:
        """
        Get candidate EntityNodes via vector similarity search.

        Args:
            embed_text: Text to embed (either "name" or "name: definition")
            top_k: Number of candidates to retrieve
        """
        embedding = await asyncio.to_thread(
            self.services.embeddings.embed_query, embed_text
        )

        def _query():
            results = self.services.neo4j.query(
                """
                CALL db.index.vector.queryNodes('entity_name_embeddings', $top_k, $vec)
                YIELD node, score
                WHERE node.group_id = $uid AND score > 0.3
                RETURN DISTINCT node.name as name, score
                ORDER BY score DESC
                """,
                {"vec": embedding, "uid": self.group_id, "top_k": top_k}
            )
            return [r["name"] for r in results]

        return await asyncio.to_thread(_query)

    async def _get_topic_candidates(self, embed_text: str, top_k: int) -> list[str]:
        """
        Get candidate TopicNodes via vector similarity search.

        Args:
            embed_text: Text to embed (either "name" or "name: definition")
            top_k: Number of candidates to retrieve
        """
        embedding = await asyncio.to_thread(
            self.services.embeddings.embed_query, embed_text
        )

        def _query():
            results = self.services.neo4j.query(
                """
                CALL db.index.vector.queryNodes('topic_embeddings', $top_k, $vec)
                YIELD node, score
                WHERE node.group_id = $uid AND score > 0.3
                RETURN DISTINCT node.name as name, score
                ORDER BY score DESC
                """,
                {"vec": embedding, "uid": self.group_id, "top_k": top_k}
            )
            return [r["name"] for r in results]

        return await asyncio.to_thread(_query)

    async def _verify_candidates(
        self, hint: str, candidates: list[str], node_type: str, question: str
    ) -> list[str]:
        """Use LLM to verify which candidates match the hint in context."""
        if not candidates:
            return []

        candidates_text = "\n".join(f"- {name}" for name in candidates)

        prompt = RESOLUTION_USER_PROMPT.format(
            question=question,
            term=hint,
            candidates=candidates_text
        )

        try:
            result = await asyncio.to_thread(
                self.structured_resolver.invoke,
                [
                    ("system", RESOLUTION_SYSTEM_PROMPT),
                    ("human", prompt)
                ]
            )

            if result.no_match:
                log(f"LLM found no matches for '{hint}'")
                return []

            resolved = [node.name for node in result.resolved_nodes]
            log(f"Resolved '{hint}' -> {resolved}")
            return resolved

        except Exception as e:
            log(f"Resolution error for '{hint}': {e}")
            return candidates[:3]
