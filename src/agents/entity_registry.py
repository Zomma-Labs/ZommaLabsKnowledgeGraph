"""
MODULE: Entity Registry
DESCRIPTION: LLM-based entity deduplication with precedent system.

Key principles:
- In-chunk dedup: Cluster similar entity names within a chunk first
- Graph lookup: Check if entity already exists (vector search, 25 candidates)
- LLM verification: Use LLM to confirm matches (no auto-merge at threshold)
- Precedent system: First occurrence establishes canonical name
- Evolving summaries: Merge new info with source tracking

Subsidiaries are SEPARATE entities (Google != Alphabet)
"""

import os
from uuid import uuid4
from typing import Dict, List, Optional, TYPE_CHECKING

from src.schemas.extraction import EntityMatchDecision, EntityResolution
from src.util.llm_client import get_nano_gpt_llm, get_nano_llm

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from src.tools.neo4j_client import Neo4jClient

# Control verbose output
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"

# Vector search limit for thorough matching
ENTITY_SEARCH_LIMIT = 25


def log(msg: str):
    """Print only if VERBOSE mode is enabled."""
    if VERBOSE:
        print(f"[EntityRegistry] {msg}")


# ============================================================================
# PROMPTS - Financial Analyst Perspective (System + User separation)
# ============================================================================

# --- ENTITY MATCHING PROMPTS ---

MATCH_SYSTEM_PROMPT = """You are a financial analyst maintaining an entity registry for a knowledge graph.

Your task: Determine if a new entity is the EXACT SAME real-world entity as any existing one.

MERGE CRITERIA (same entity, different names):
- "Apple Inc." = "Apple" = "Apple headquarters" = "AAPL" (same company)
- "Tim Cook" = "Timothy Cook" = "Apple CEO Tim Cook" (same person)
- "Federal Reserve" = "The Fed" = "Fed" (same institution)
- "Exxon Mobil" = "ExxonMobil" = "XOM" (same company)

DO NOT MERGE (related but distinct entities):
- "AWS" is NOT "Amazon" (AWS is a subsidiary/division - keep parent and child separate!)
- "Instagram" is NOT "Meta" (Instagram is owned by Meta - ownership is a relationship, not identity)
- "YouTube" is NOT its parent company (subsidiaries are separate entities with edges between them)
- "Tim Cook" is NOT "Apple" (person vs company - different entity types)
- "Apple Inc." is NOT "Apple Records" (same name, completely different companies)

CRITICAL: Subsidiary/ownership relationships should be captured as EDGES in the graph, NOT by merging entities.

Your decision:
- If SAME as an existing entity: is_same=True, match_index=[number 1-based]
- If DISTINCT (even if related): is_same=False, match_index=None"""

MATCH_USER_PROMPT = """NEW ENTITY TO RESOLVE:
Name: {entity_name}
Type: {entity_type}
Summary: {entity_summary}

EXISTING ENTITIES IN REGISTRY:
{candidates_text}

Is the new entity the EXACT SAME real-world entity as any existing one?"""


# --- SUMMARY MERGING PROMPTS ---

MERGE_SYSTEM_PROMPT = """You are maintaining entity summaries for a financial knowledge graph.

When merging summaries:
- Combine non-redundant information
- Keep existing source annotations [Source: ...]
- Add new source annotation for new facts
- Avoid duplication
- Keep it concise
- Ensure it is factually correct
- Ensure Specificity: Do not include information that is not directly related to the entity"""

MERGE_USER_PROMPT = """EXISTING SUMMARY:
{existing_summary}

NEW INFORMATION FROM CHUNK {chunk_uuid}:
{new_summary}

Merge these into a single comprehensive summary."""


class EntityRegistry:
    """
    Entity resolution and deduplication using LLM verification.

    Two-level deduplication:
    1. In-chunk: Cluster similar entity names within a chunk
    2. Graph lookup: Vector search + LLM verification against existing entities
    """

    def __init__(
        self,
        neo4j_client: Optional["Neo4jClient"] = None,
        embeddings=None,
        llm: Optional["BaseChatModel"] = None,
        group_id: str = "default"
    ):
        if neo4j_client is None or embeddings is None:
            from src.util.services import get_services
            services = get_services()
            neo4j_client = neo4j_client or services.neo4j
            embeddings = embeddings or services.embeddings

        self.neo4j = neo4j_client
        self.embeddings = embeddings
        self.group_id = group_id

        # Use gpt-5-mini for entity matching (simpler task)
        self.match_llm = get_nano_gpt_llm()
        self.structured_matcher = self.match_llm.with_structured_output(EntityMatchDecision)

        # Use gemini-2.5-flash-lite for summary merging (simple text task)
        self.merge_llm = get_nano_llm()

    def resolve(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        chunk_uuid: str,
        precomputed_embedding: Optional[List[float]] = None
    ) -> EntityResolution:
        """
        Resolve an entity against the graph, using LLM to verify matches.

        Args:
            entity_name: The entity name from extraction
            entity_type: Entity type (Company, Person, etc.)
            entity_summary: Description of the entity
            chunk_uuid: UUID of the source chunk (for provenance)
            precomputed_embedding: Optional pre-computed embedding to avoid API call

        Returns:
            EntityResolution with uuid, canonical_name, is_new, updated_summary
        """
        # Step 1: Embed the entity (use precomputed if available)
        if precomputed_embedding is not None:
            embedding = precomputed_embedding
        else:
            embed_text = f"{entity_name}: {entity_summary}"
            embedding = self.embeddings.embed_query(embed_text)

        # Step 2: Vector search in graph
        candidates = self._search_candidates(embedding)
        log(f"Found {len(candidates)} candidates for '{entity_name}'")

        # Step 3: LLM verification if candidates exist
        if candidates:
            decision = self._llm_verify_match(
                entity_name=entity_name,
                entity_type=entity_type,
                entity_summary=entity_summary,
                candidates=candidates
            )

            if decision.is_same and decision.match_index is not None:
                # Match found - use existing entity
                matched = candidates[decision.match_index - 1]  # 1-based index
                log(f"LLM matched '{entity_name}' -> '{matched['name']}' ({decision.reasoning})")

                # Merge summaries
                merged_summary = self._merge_summaries(
                    existing_summary=matched.get("summary", ""),
                    new_summary=entity_summary,
                    chunk_uuid=chunk_uuid
                )

                # Update source_chunks list
                existing_chunks = matched.get("source_chunks", [])
                if chunk_uuid not in existing_chunks:
                    existing_chunks.append(chunk_uuid)

                # Update aliases list
                existing_aliases = matched.get("aliases", [])
                if entity_name not in existing_aliases and entity_name != matched["name"]:
                    existing_aliases.append(entity_name)

                return EntityResolution(
                    uuid=matched["uuid"],
                    canonical_name=matched["name"],
                    is_new=False,
                    updated_summary=merged_summary,
                    source_chunks=existing_chunks,
                    aliases=existing_aliases
                )

        # Step 4: No match - create new entity
        log(f"Creating new entity: '{entity_name}'")
        new_uuid = str(uuid4())
        initial_summary = f"{entity_summary}\n[Source: {chunk_uuid}]"

        return EntityResolution(
            uuid=new_uuid,
            canonical_name=entity_name,
            is_new=True,
            updated_summary=initial_summary,
            source_chunks=[chunk_uuid],
            aliases=[]
        )

    def resolve_batch(
        self,
        entities: List[Dict],
        chunk_uuid: str
    ) -> Dict[str, EntityResolution]:
        """
        Resolve multiple entities from a chunk.

        Args:
            entities: List of dicts with 'name', 'type', 'summary'
            chunk_uuid: UUID of the source chunk

        Returns:
            Dict mapping entity names to EntityResolution
        """
        results = {}
        for entity in entities:
            name = entity.get("name", "")
            entity_type = entity.get("type", "Unknown")
            summary = entity.get("summary", "")

            if name:
                resolution = self.resolve(
                    entity_name=name,
                    entity_type=entity_type,
                    entity_summary=summary,
                    chunk_uuid=chunk_uuid
                )
                results[name] = resolution

        return results

    def _search_candidates(self, embedding: List[float]) -> List[Dict]:
        """Search for candidate entities in the graph."""
        try:
            results = self.neo4j.vector_search(
                index_name="entity_name_embeddings",
                query_vector=embedding,
                top_k=ENTITY_SEARCH_LIMIT,
                filters={"group_id": self.group_id}
            )

            # Extract node properties from results
            candidates = []
            for record in results:
                node = record.get("node", {})
                candidates.append({
                    "uuid": node.get("uuid", ""),
                    "name": node.get("name", ""),
                    "summary": node.get("summary", ""),
                    "entity_type": node.get("entity_type", ""),
                    "source_chunks": node.get("source_chunks", []),
                    "aliases": node.get("aliases", []),
                    "score": record.get("score", 0)
                })

            return candidates

        except Exception as e:
            log(f"Vector search error: {e}")
            return []

    def _llm_verify_match(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        candidates: List[Dict]
    ) -> EntityMatchDecision:
        """Use LLM to verify if entity matches any candidate."""
        # Format candidates for prompt
        candidates_text = ""
        for i, cand in enumerate(candidates, 1):
            aliases_str = f" (aliases: {', '.join(cand['aliases'])})" if cand.get("aliases") else ""
            candidates_text += (
                f"{i}. {cand['name']}{aliases_str}\n"
                f"   Type: {cand.get('entity_type', 'Unknown')}\n"
                f"   Summary: {cand.get('summary', 'No summary')[:200]}\n\n"
            )

        user_prompt = MATCH_USER_PROMPT.format(
            entity_name=entity_name,
            entity_type=entity_type,
            entity_summary=entity_summary,
            candidates_text=candidates_text
        )
        messages = [
            ("system", MATCH_SYSTEM_PROMPT),
            ("human", user_prompt)
        ]

        try:
            decision = self.structured_matcher.invoke(messages)
            return decision
        except Exception as e:
            log(f"LLM match error: {e}")
            # On error, assume distinct to avoid incorrect merges
            return EntityMatchDecision(is_same=False, match_index=None, reasoning="Error during matching")

    def _merge_summaries(
        self,
        existing_summary: str,
        new_summary: str,
        chunk_uuid: str
    ) -> str:
        """Merge existing and new summaries with source tracking."""
        # If existing is empty, just use new with source
        if not existing_summary.strip():
            return f"{new_summary}\n[Source: {chunk_uuid}]"

        # If summaries are very similar, don't bother merging
        if new_summary.strip() in existing_summary or existing_summary in new_summary.strip():
            return existing_summary

        user_prompt = MERGE_USER_PROMPT.format(
            existing_summary=existing_summary,
            new_summary=new_summary,
            chunk_uuid=chunk_uuid
        )
        messages = [
            ("system", MERGE_SYSTEM_PROMPT),
            ("human", user_prompt)
        ]

        try:
            result = self.merge_llm.invoke(messages)
            return result.content.strip()
        except Exception as e:
            log(f"Summary merge error: {e}")
            # Fallback: append new info
            return f"{existing_summary}\n{new_summary}\n[Source: {chunk_uuid}]"
