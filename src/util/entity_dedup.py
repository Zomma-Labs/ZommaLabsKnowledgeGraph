"""
MODULE: Deferred Deduplication (Before Neo4j Write)
DESCRIPTION:
    Collects entities during ingestion, clusters them by similarity,
    and writes ONLY canonical entities to Neo4j. Duplicates never touch the graph.

    Uses a two-stage approach:
    1. Connected components to find CANDIDATE duplicate groups (wide net)
    2. LLM verification to identify truly distinct entities within each component

USAGE:
    1. Set DEFER_DEDUPLICATION=true to enable
    2. During ingestion, entities are tracked via register_entity() - NOT written to Neo4j
    3. Facts are tracked via register_fact() - NOT written to Neo4j
    4. After all chunks processed, call finalize() to:
       a. Build similarity graph + find connected components
       b. LLM identifies distinct entities per component
       c. Write canonical entities to Neo4j
       d. Write facts with remapped UUIDs
"""

import os
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
from pydantic import BaseModel, Field
import numpy as np

# Check if deferred deduplication is enabled (default: true)
DEFER_DEDUPLICATION = os.getenv("DEFER_DEDUPLICATION", "true").lower() == "true"
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"


def log(msg: str):
    """Print only if VERBOSE mode is enabled."""
    if VERBOSE:
        print(msg)


# ============================================
# Pydantic Models for LLM Structured Output
# ============================================

class DistinctEntity(BaseModel):
    """A distinct real-world entity identified by the LLM."""
    canonical_name: str = Field(
        description="The best/most complete name for this entity (e.g., 'Apple Inc.' not 'Apple')"
    )
    member_indices: List[int] = Field(
        description="List of indices (0-based) from the input that refer to this same entity"
    )
    merged_summary: str = Field(
        description="Combined summary incorporating information from all members"
    )


class DeduplicationResult(BaseModel):
    """LLM output identifying distinct entities from a list of candidates."""
    distinct_entities: List[DistinctEntity] = Field(
        description="List of distinct real-world entities. Each input entity should appear in exactly one group."
    )


# ============================================
# Union-Find for Connected Components
# ============================================

class UnionFind:
    """Disjoint Set Union (Union-Find) with path compression and union by rank."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """Find with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        """Union by rank."""
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

    def get_components(self) -> Dict[int, List[int]]:
        """Return dict mapping root -> list of members."""
        components = defaultdict(list)
        for i in range(len(self.parent)):
            components[self.find(i)].append(i)
        return dict(components)


# ============================================
# Data Classes
# ============================================

@dataclass
class PendingEntity:
    """Entity tracked during ingestion for clustering before Neo4j write."""
    uuid: str
    name: str
    node_type: str  # "Entity" or "Topic"
    summary: str
    embedding: Optional[List[float]]
    group_id: str


@dataclass
class PendingFact:
    """Fact tracked during ingestion for writing after entity clustering."""
    fact_data: Dict[str, Any]
    precomputed: Dict[str, Any]
    episode_uuid: str
    group_id: str


@dataclass
class MergeRecord:
    """Record of entities that were merged together."""
    canonical_uuid: str
    canonical_name: str
    merged_uuids: List[str]
    merged_names: List[str]
    original_summaries: List[str]
    final_summary: str


# ============================================
# Main Manager Class
# ============================================

class DeferredDeduplicationManager:
    """
    Collects entities and facts during ingestion, deduplicates entities via
    connected components + LLM verification, then writes to Neo4j.
    """
    _instance: Optional["DeferredDeduplicationManager"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._pending_entities: Dict[str, PendingEntity] = {}
            cls._instance._pending_facts: List[PendingFact] = []
            cls._instance._uuid_remap: Dict[str, str] = {}
            cls._instance._merge_history: List[MergeRecord] = []  # Track what was merged
            cls._instance._llm = None
        return cls._instance

    @classmethod
    def get_instance(cls) -> "DeferredDeduplicationManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset for testing or between documents."""
        if cls._instance:
            cls._instance._pending_entities = {}
            cls._instance._pending_facts = []
            cls._instance._uuid_remap = {}
            cls._instance._merge_history = []

    def _get_llm(self):
        """Lazy-load LLM client for entity disambiguation."""
        if self._llm is None:
            from src.util.llm_client import get_dedup_llm
            self._llm = get_dedup_llm()  # GPT-5.1 for dedup (cheaper than 5.2)
        return self._llm

    def register_entity(self,
                        uuid: str,
                        name: str,
                        node_type: str,
                        summary: str,
                        embedding: Optional[List[float]],
                        group_id: str) -> str:
        """Register a pending entity. NOT written to Neo4j yet."""
        normalized_name = name.strip().title()
        entity = PendingEntity(
            uuid=uuid,
            name=normalized_name,
            node_type=node_type,
            summary=summary or "",
            embedding=embedding,
            group_id=group_id
        )
        self._pending_entities[uuid] = entity
        return uuid

    def register_fact(self,
                      fact_data: Dict[str, Any],
                      precomputed: Dict[str, Any],
                      episode_uuid: str,
                      group_id: str) -> None:
        """Register a pending fact. NOT written to Neo4j yet."""
        self._pending_facts.append(PendingFact(
            fact_data=fact_data,
            precomputed=precomputed,
            episode_uuid=episode_uuid,
            group_id=group_id
        ))

    def get_remapped_uuid(self, original_uuid: str) -> str:
        """Get the canonical UUID for a potentially duplicate entity."""
        return self._uuid_remap.get(original_uuid, original_uuid)

    def _compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarity matrix."""
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        normalized = embeddings / norms

        # Cosine similarity = dot product of normalized vectors
        similarity = np.dot(normalized, normalized.T)
        return similarity

    def _build_connected_components(self,
                                     entities: List[PendingEntity],
                                     similarity_threshold: float = 0.85) -> Dict[int, List[int]]:
        """
        Build similarity graph and find connected components.

        Args:
            entities: List of entities with embeddings
            similarity_threshold: Entities with similarity > this get an edge

        Returns:
            Dict mapping component_id -> list of entity indices
        """
        n = len(entities)
        if n < 2:
            return {0: list(range(n))} if n == 1 else {}

        # Build embedding matrix
        embeddings = np.array([e.embedding for e in entities])

        # Compute similarity matrix
        similarity = self._compute_similarity_matrix(embeddings)

        # Build Union-Find and union similar entities
        uf = UnionFind(n)
        for i in range(n):
            for j in range(i + 1, n):
                if similarity[i, j] > similarity_threshold:
                    uf.union(i, j)

        return uf.get_components()

    def _llm_identify_distinct_entities(self,
                                         entities: List[PendingEntity],
                                         max_batch_size: int = 30) -> List[DistinctEntity]:
        """
        Use LLM to identify distinct real-world entities from a list of candidates.

        For large components, processes in batches with overlap detection.
        """
        if len(entities) == 1:
            # Single entity - no deduplication needed
            return [DistinctEntity(
                canonical_name=entities[0].name,
                member_indices=[0],
                merged_summary=entities[0].summary
            )]

        # For small components, process directly
        if len(entities) <= max_batch_size:
            return self._llm_dedupe_batch(entities)

        # For large components, process in overlapping batches
        # and merge results
        log(f"      Large component ({len(entities)} entities), processing in batches...")
        return self._llm_dedupe_large_component(entities, max_batch_size)

    def _llm_dedupe_batch(self, entities: List[PendingEntity]) -> List[DistinctEntity]:
        """Process a batch of entities through LLM to identify distinct entities."""
        llm = self._get_llm()
        structured_llm = llm.with_structured_output(DeduplicationResult)

        # Build the entity list for the prompt
        entity_list = "\n".join([
            f"{i}. Name: \"{e.name}\"\n   Definition: {e.summary or 'No definition available'}"
            for i, e in enumerate(entities)
        ])

        prompt = f"""You are deduplicating entities extracted from financial documents for a knowledge graph.

TASK: Group entity names that refer to THE SAME real-world entity.
This is NOT grouping similar or related entities - only TRUE duplicates (same entity, different names).

If unsure, do NOT merge. False negatives are better than false positives.

ENTITIES:
{entity_list}

ENTITY TYPES (cannot merge across different types):
- PERSON: Individuals (executives, analysts, politicians, employees)
- ORGANIZATION: Companies, agencies, institutions, funds
- LOCATION: Countries, cities, regions, addresses
- PRODUCT: Products, services, platforms, technologies
- EVENT: Conferences, announcements, filings
- OTHER: Indices, currencies, commodities, concepts

MERGE - same entity, different names:
- Ticker ↔ company: "AAPL" = "Apple Inc." = "Apple"
- Abbreviation ↔ full name: "Fed" = "Federal Reserve" = "The Fed"
- Location variants: "NYC" = "New York City" = "New York, NY"
- Person name variants: "Tim Cook" = "Timothy D. Cook"
- Product variants: "AWS" = "Amazon Web Services"

DO NOT MERGE - related but different entities:
- Person ≠ their company: "Tim Cook" ≠ "Apple" (CEO is not the company)
- Parent ≠ subsidiary: "Alphabet" ≠ "Google" ≠ "YouTube"
- Competitors: "Goldman Sachs" ≠ "Morgan Stanley"
- Product ≠ company: "iPhone" ≠ "Apple"
- Location ≠ org there: "Silicon Valley" ≠ "Google"

DECISION TEST: If you swapped one name for the other in a sentence, would the meaning stay exactly the same?

IMPORTANT:
- Every input index (0 to {len(entities) - 1}) must appear in exactly ONE group
- USE THE DEFINITIONS to understand what type of entity each one is
- When in doubt, keep entities separate

Return the distinct real-world entities found."""

        try:
            result = structured_llm.invoke([("human", prompt)])

            # Validate that all indices are covered
            covered = set()
            for de in result.distinct_entities:
                covered.update(de.member_indices)

            expected = set(range(len(entities)))
            if covered != expected:
                missing = expected - covered
                log(f"      ⚠️ LLM missed indices {missing}, adding as singletons")
                for idx in missing:
                    result.distinct_entities.append(DistinctEntity(
                        canonical_name=entities[idx].name,
                        member_indices=[idx],
                        merged_summary=entities[idx].summary
                    ))

            return result.distinct_entities

        except Exception as e:
            log(f"      ⚠️ LLM deduplication failed: {e}, treating all as distinct")
            return [
                DistinctEntity(
                    canonical_name=e.name,
                    member_indices=[i],
                    merged_summary=e.summary
                )
                for i, e in enumerate(entities)
            ]

    def _llm_dedupe_large_component(self,
                                     entities: List[PendingEntity],
                                     batch_size: int = 30) -> List[DistinctEntity]:
        """
        Handle large components by processing in batches and merging results.

        Strategy:
        1. Sort entities by name for better locality
        2. Process in batches with some overlap
        3. Merge results, detecting cross-batch duplicates by canonical name
        """
        # Sort by name to group similar names together
        indexed_entities = list(enumerate(entities))
        indexed_entities.sort(key=lambda x: x[1].name.lower())

        # Track: canonical_name -> (canonical_uuid, list of original indices, merged_summary)
        canonical_map: Dict[str, Tuple[str, List[int], str]] = {}

        # Process in batches
        for batch_start in range(0, len(indexed_entities), batch_size - 5):  # Overlap of 5
            batch_end = min(batch_start + batch_size, len(indexed_entities))
            batch = indexed_entities[batch_start:batch_end]

            # Extract entities for this batch
            batch_entities = [e for _, e in batch]
            original_indices = [i for i, _ in batch]

            # LLM dedupe this batch
            distinct = self._llm_dedupe_batch(batch_entities)

            # Map back to original indices and merge with existing results
            for de in distinct:
                # Convert batch indices to original indices
                orig_member_indices = [original_indices[i] for i in de.member_indices]

                # Check if this canonical name already exists
                canonical_key = de.canonical_name.lower().strip()

                if canonical_key in canonical_map:
                    # Merge with existing
                    existing_indices = canonical_map[canonical_key][1]
                    existing_summary = canonical_map[canonical_key][2]

                    # Add new indices (avoid duplicates from overlap)
                    for idx in orig_member_indices:
                        if idx not in existing_indices:
                            existing_indices.append(idx)

                    # Merge summaries if new info (no truncation to preserve full context)
                    if de.merged_summary and de.merged_summary not in existing_summary:
                        merged = f"{existing_summary} {de.merged_summary}"
                        canonical_map[canonical_key] = (
                            canonical_map[canonical_key][0],
                            existing_indices,
                            merged
                        )
                else:
                    # New canonical entity
                    canonical_map[canonical_key] = (
                        de.canonical_name,
                        orig_member_indices,
                        de.merged_summary
                    )

        # Convert back to DistinctEntity list
        return [
            DistinctEntity(
                canonical_name=name,
                member_indices=indices,
                merged_summary=summary
            )
            for name, (_, indices, summary) in canonical_map.items()
        ]

    def cluster_and_remap(self, similarity_threshold: float = 0.85) -> Dict[str, Any]:
        """
        Main deduplication logic:
        1. Separate entities by type and group
        2. Build connected components (candidate groups)
        3. LLM identifies distinct entities per component
        4. Create UUID remapping

        Args:
            similarity_threshold: Connect entities with similarity > this
        """
        entities = list(self._pending_entities.values())

        if len(entities) < 2:
            return {"components_found": 0, "distinct_entities": len(entities), "duplicates_merged": 0}

        # Separate by node_type and group_id (don't merge across these)
        by_type_group: Dict[Tuple[str, str], List[PendingEntity]] = defaultdict(list)
        for e in entities:
            if e.embedding:
                by_type_group[(e.node_type, e.group_id)].append(e)

        stats = {
            "components_found": 0,
            "distinct_entities": 0,
            "duplicates_merged": 0,
            "llm_calls": 0
        }

        for (node_type, group_id), type_entities in by_type_group.items():
            if len(type_entities) < 2:
                stats["distinct_entities"] += len(type_entities)
                continue

            log(f"   Processing {len(type_entities)} {node_type}s...")

            # Step 1: Find connected components
            components = self._build_connected_components(type_entities, similarity_threshold)

            multi_entity_components = [c for c in components.values() if len(c) > 1]
            stats["components_found"] += len(multi_entity_components)

            log(f"      Found {len(multi_entity_components)} components with potential duplicates")

            # Step 2: Process each component
            for comp_id, member_indices in components.items():
                comp_entities = [type_entities[i] for i in member_indices]

                if len(comp_entities) == 1:
                    # Singleton - no deduplication needed
                    stats["distinct_entities"] += 1
                    continue

                # LLM identifies distinct entities
                stats["llm_calls"] += 1
                distinct = self._llm_identify_distinct_entities(comp_entities)

                log(f"      Component with {len(comp_entities)} candidates → {len(distinct)} distinct entities")
                stats["distinct_entities"] += len(distinct)
                stats["duplicates_merged"] += len(comp_entities) - len(distinct)

                # Step 3: Create UUID remapping and record merge history
                self._apply_distinct_entities(comp_entities, distinct)

        return stats

    def _apply_distinct_entities(self, comp_entities: List[PendingEntity], distinct: List[DistinctEntity]) -> None:
        """Apply deduplication results: create UUID remapping and record merge history."""
        for de in distinct:
            if len(de.member_indices) > 1:
                # Pick the member with longest summary as canonical
                member_entities = [comp_entities[i] for i in de.member_indices]
                canonical_entity = max(member_entities, key=lambda e: len(e.summary))

                # Record original summaries before merging
                original_summaries = [e.summary for e in member_entities]
                merged_names = [e.name for e in member_entities]
                merged_uuids = [e.uuid for e in member_entities]

                # Update canonical entity's name and summary
                canonical_entity.name = de.canonical_name
                canonical_entity.summary = de.merged_summary

                # Record the merge
                self._merge_history.append(MergeRecord(
                    canonical_uuid=canonical_entity.uuid,
                    canonical_name=de.canonical_name,
                    merged_uuids=merged_uuids,
                    merged_names=merged_names,
                    original_summaries=original_summaries,
                    final_summary=de.merged_summary
                ))

                # Map all other UUIDs to canonical
                for me in member_entities:
                    if me.uuid != canonical_entity.uuid:
                        self._uuid_remap[me.uuid] = canonical_entity.uuid

    async def cluster_and_remap_async(self, similarity_threshold: float = 0.85, concurrency: int = 20) -> Dict[str, Any]:
        """
        Async version with parallel LLM calls for deduplication.

        Args:
            similarity_threshold: Connect entities with similarity > this
            concurrency: Max concurrent LLM deduplication calls
        """
        import asyncio

        entities = list(self._pending_entities.values())

        if len(entities) < 2:
            return {"components_found": 0, "distinct_entities": len(entities), "duplicates_merged": 0}

        # Separate by node_type and group_id (don't merge across these)
        by_type_group: Dict[Tuple[str, str], List[PendingEntity]] = defaultdict(list)
        for e in entities:
            if e.embedding:
                by_type_group[(e.node_type, e.group_id)].append(e)

        stats = {
            "components_found": 0,
            "distinct_entities": 0,
            "duplicates_merged": 0,
            "llm_calls": 0
        }

        # Collect all multi-entity components for parallel processing
        components_to_process: List[List[PendingEntity]] = []

        for (node_type, group_id), type_entities in by_type_group.items():
            if len(type_entities) < 2:
                stats["distinct_entities"] += len(type_entities)
                continue

            log(f"   Processing {len(type_entities)} {node_type}s...")

            # Build connected components
            components = self._build_connected_components(type_entities, similarity_threshold)

            multi_entity_components = [c for c in components.values() if len(c) > 1]
            stats["components_found"] += len(multi_entity_components)

            log(f"      Found {len(multi_entity_components)} components with potential duplicates")

            for comp_id, member_indices in components.items():
                comp_entities = [type_entities[i] for i in member_indices]

                if len(comp_entities) == 1:
                    stats["distinct_entities"] += 1
                else:
                    components_to_process.append(comp_entities)

        if not components_to_process:
            return stats

        # Process all components in parallel
        sem = asyncio.Semaphore(concurrency)

        async def process_component(comp_entities: List[PendingEntity]):
            async with sem:
                distinct = await asyncio.to_thread(
                    self._llm_identify_distinct_entities,
                    comp_entities
                )
                return comp_entities, distinct

        log(f"      Running {len(components_to_process)} LLM dedup calls (concurrency={concurrency})...")

        tasks = [process_component(comp) for comp in components_to_process]
        results = await asyncio.gather(*tasks)

        # Apply all results
        for comp_entities, distinct in results:
            stats["llm_calls"] += 1
            log(f"      Component {len(comp_entities)} candidates → {len(distinct)} distinct")
            stats["distinct_entities"] += len(distinct)
            stats["duplicates_merged"] += len(comp_entities) - len(distinct)
            self._apply_distinct_entities(comp_entities, distinct)

        return stats

    def get_merge_history(self) -> List[MergeRecord]:
        """Get the list of all merges that were performed."""
        return self._merge_history.copy()

    def write_canonical_entities(self, neo4j_client) -> int:
        """Write only canonical entities to Neo4j (skip duplicates)."""
        written = 0

        for uuid, entity in self._pending_entities.items():
            # Skip if this entity is a duplicate (remapped to another)
            if uuid in self._uuid_remap:
                continue

            label = "TopicNode" if entity.node_type == "Topic" else "EntityNode"

            cypher = f"""
            MERGE (n:{label} {{name: $name, group_id: $group_id}})
            ON CREATE SET
                n.uuid = $uuid,
                n.summary = $summary,
                n.embedding = $embedding,
                n.created_at = datetime()
            ON MATCH SET
                n.uuid = COALESCE(n.uuid, $uuid),
                n.embedding = CASE WHEN n.embedding IS NULL THEN $embedding ELSE n.embedding END,
                n.summary = CASE WHEN n.summary IS NULL OR n.summary = "" THEN $summary ELSE n.summary END
            """

            try:
                neo4j_client.query(cypher, {
                    "name": entity.name,
                    "group_id": entity.group_id,
                    "uuid": entity.uuid,
                    "summary": entity.summary,
                    "embedding": entity.embedding
                })
                written += 1
            except Exception as e:
                log(f"   ⚠️ Failed to write entity {entity.name}: {e}")

        return written

    def write_facts_with_remapping(self, assembler, neo4j_client) -> int:
        """Write all pending facts to Neo4j with UUID remapping applied."""
        written = 0

        for pending in self._pending_facts:
            fact_data = pending.fact_data.copy()
            precomputed = pending.precomputed

            # Remap subject UUID
            if fact_data.get("subject_uuid"):
                fact_data["subject_uuid"] = self.get_remapped_uuid(fact_data["subject_uuid"])

            # Remap object UUID
            if fact_data.get("object_uuid"):
                fact_data["object_uuid"] = self.get_remapped_uuid(fact_data["object_uuid"])

            # Remap topic UUIDs
            if fact_data.get("topics"):
                for topic in fact_data["topics"]:
                    if topic.get("uuid"):
                        topic["uuid"] = self.get_remapped_uuid(topic["uuid"])

            try:
                with neo4j_client.driver.session() as session:
                    with session.begin_transaction() as tx:
                        fact_obj = fact_data["fact_obj"]
                        relationship_classification = fact_data.get("relationship_classification")

                        fact_uuid = precomputed.get("resolved_fact_uuid") or precomputed.get("new_fact_uuid")

                        assembler._assemble_fact_in_transaction_precomputed(
                            tx=tx,
                            fact_uuid=fact_uuid,
                            fact_obj=fact_obj,
                            fact_embedding=precomputed.get("fact_embedding"),
                            is_new=(precomputed.get("resolved_fact_uuid") is None),
                            subject_uuid=fact_data["subject_uuid"],
                            subject_label=fact_data["subject_label"],
                            subject_summary=fact_data.get("subject_summary", ""),
                            subject_type=fact_data.get("subject_type", "Entity"),
                            subj_embedding=precomputed.get("subj_embedding"),
                            object_uuid=fact_data.get("object_uuid"),
                            object_label=fact_data.get("object_label"),
                            object_summary=fact_data.get("object_summary", ""),
                            object_type=fact_data.get("object_type", "Entity"),
                            obj_embedding=precomputed.get("obj_embedding"),
                            episode_uuid=pending.episode_uuid,
                            group_id=pending.group_id,
                            relationship_classification=relationship_classification
                        )

                        # Link topics
                        for topic in fact_data.get("topics", []):
                            if topic.get("uuid"):
                                assembler._link_topic_in_transaction_precomputed(
                                    tx=tx,
                                    topic_uuid=topic["uuid"],
                                    topic_label=topic["label"],
                                    topic_summary=topic.get("summary", ""),
                                    topic_embedding=precomputed.get("topic_embeddings", {}).get(topic["uuid"]),
                                    episode_uuid=pending.episode_uuid,
                                    group_id=pending.group_id
                                )

                        tx.commit()
                        written += 1

            except Exception as e:
                log(f"   ⚠️ Failed to write fact: {e}")

        return written

    def finalize(self, neo4j_client, assembler, group_id: str,
                 similarity_threshold: float = 0.70) -> Dict[str, Any]:
        """
        Main entry point after all chunks processed:
        1. Build connected components (candidate groups)
        2. LLM identifies distinct entities per component
        3. Write canonical entities to Neo4j
        4. Write facts with remapped UUIDs
        """
        log(f"   📊 Finalizing: {len(self._pending_entities)} entities, {len(self._pending_facts)} facts")

        # Step 1 & 2: Cluster and create remapping
        cluster_stats = self.cluster_and_remap(similarity_threshold)
        log(f"   🔍 Found {cluster_stats['distinct_entities']} distinct entities ({cluster_stats['duplicates_merged']} duplicates merged)")

        # Step 3: Write canonical entities
        entities_written = self.write_canonical_entities(neo4j_client)
        log(f"   ✅ Wrote {entities_written} canonical entities")

        # Step 4: Write facts with remapping
        facts_written = self.write_facts_with_remapping(assembler, neo4j_client)
        log(f"   ✅ Wrote {facts_written} facts")

        stats = {
            **cluster_stats,
            "entities_written": entities_written,
            "facts_written": facts_written,
            "merge_history": self._merge_history.copy()  # Include merge history
        }

        # Clear state
        self.reset()

        return stats


# Global instance accessor
def get_dedup_manager() -> DeferredDeduplicationManager:
    return DeferredDeduplicationManager.get_instance()


def is_deferred_mode() -> bool:
    """Check if deferred deduplication is enabled."""
    return DEFER_DEDUPLICATION
