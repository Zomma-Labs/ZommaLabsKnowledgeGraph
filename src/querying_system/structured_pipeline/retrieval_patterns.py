"""
Retrieval patterns for different query types.
Each pattern is a deterministic sequence of graph operations.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .models import QueryPlan, RetrievedChunk, ResolvedEntity, RetrievalResult

if TYPE_CHECKING:
    from .retriever import GraphOperations


class BaseRetrievalPattern(ABC):
    """Base class for retrieval patterns."""

    def __init__(self, graph_ops: "GraphOperations"):
        self.graph = graph_ops

    @property
    @abstractmethod
    def name(self) -> str:
        """Pattern name for logging."""
        pass

    @abstractmethod
    def execute(self, plan: QueryPlan) -> RetrievalResult:
        """Execute the retrieval pattern."""
        pass


class EntityAttributePattern(BaseRetrievalPattern):
    """
    Pattern for ENTITY_ATTRIBUTE queries.

    Flow: resolve entity → get_entity_info → get related chunks if needed

    Example: "What is the Beige Book?" → resolve → get info → get key relationships
    """

    @property
    def name(self) -> str:
        return "entity_attribute"

    def execute(self, plan: QueryPlan) -> RetrievalResult:
        result = RetrievalResult(plan=plan, retrieval_pattern_used=self.name)

        for entity_query in plan.anchor_entities:
            # Step 1: Resolve entity
            resolved = self.graph.resolve_entity(entity_query)
            result.resolved_entities.append(resolved)

            if not resolved.resolved_name:
                result.warnings.append(f"Could not resolve entity: {entity_query}")
                continue

            entity_name = resolved.resolved_name

            # Step 2: Get entity info
            info = self.graph.get_entity_info(entity_name)
            if info:
                # Create a synthetic chunk from entity summary
                chunk = RetrievedChunk(
                    chunk_id=f"entity_info_{entity_name[:20]}",
                    doc_id="graph_metadata",
                    content=f"Entity: {entity_name}\nType: {info.get('entity_type', 'Unknown')}\nSummary: {info.get('summary', 'No summary available')}",
                    source_entity=entity_name,
                    edge_type="ENTITY_INFO",
                )
                result.chunks.append(chunk)

            # Step 3: Get key relationships for context
            edges = self.graph.explore_neighbors(entity_name, query_hint="")
            for edge in edges[:3]:
                chunks = self.graph.get_chunks_by_edge(
                    entity_name=entity_name,
                    edge_type=edge["type"],
                    direction="both",
                    limit=1,
                )
                result.chunks.extend(chunks)

        result.total_candidates_found = len(result.chunks)
        return result


class EntityRelationshipPattern(BaseRetrievalPattern):
    """
    Pattern for ENTITY_RELATIONSHIP queries.

    Flow: resolve entity → explore neighbors (filtered) → get chunks

    Example: "What happened in the Chicago District?"
             → resolve Chicago → explore all edges → get chunks
    """

    @property
    def name(self) -> str:
        return "entity_relationship"

    def execute(self, plan: QueryPlan) -> RetrievalResult:
        result = RetrievalResult(plan=plan, retrieval_pattern_used=self.name)

        for entity_query in plan.anchor_entities:
            # Step 1: Resolve entity
            resolved = self.graph.resolve_entity(entity_query)
            result.resolved_entities.append(resolved)

            if not resolved.resolved_name:
                result.warnings.append(f"Could not resolve entity: {entity_query}")
                continue

            entity_name = resolved.resolved_name

            # Step 2: Get chunks
            if plan.target_relationship:
                # Specific relationship requested
                chunks = self.graph.get_chunks_by_edge(
                    entity_name=entity_name,
                    edge_type=plan.target_relationship,
                    direction=plan.relationship_direction or "both",
                    limit=10,
                )
                result.chunks.extend(chunks)

                # If no chunks, try semantic exploration
                if not chunks:
                    query_hint = plan.target_relationship.replace("_", " ").lower()
                    edges = self.graph.explore_neighbors(entity_name, query_hint)
                    for edge in edges[:5]:
                        edge_chunks = self.graph.get_chunks_by_edge(
                            entity_name=entity_name,
                            edge_type=edge["type"],
                            direction=edge.get("direction", "both"),
                            limit=2,
                        )
                        result.chunks.extend(edge_chunks)
            else:
                # No specific relationship - explore and get top chunks
                if plan.fallback_search_terms:
                    query_hint = plan.fallback_search_terms[0]
                    edges = self.graph.explore_neighbors(entity_name, query_hint)
                else:
                    edges = self.graph.explore_neighbors(entity_name, "")

                for edge in edges[:5]:
                    chunks = self.graph.get_chunks_by_edge(
                        entity_name=entity_name,
                        edge_type=edge["type"],
                        limit=2,
                    )
                    result.chunks.extend(chunks)

        result.total_candidates_found = len(result.chunks)
        return result


class ComparisonPattern(BaseRetrievalPattern):
    """
    Pattern for COMPARISON queries.

    Flow: resolve all entities → get comparable data for each

    Example: "Compare Chicago and Dallas districts"
             → resolve both → get similar relationships for each
    """

    @property
    def name(self) -> str:
        return "comparison"

    def execute(self, plan: QueryPlan) -> RetrievalResult:
        result = RetrievalResult(plan=plan, retrieval_pattern_used=self.name)

        entities_to_compare = plan.comparison_entities or plan.anchor_entities

        for entity_query in entities_to_compare:
            resolved = self.graph.resolve_entity(entity_query)
            result.resolved_entities.append(resolved)

            if not resolved.resolved_name:
                result.warnings.append(f"Could not resolve: {entity_query}")
                continue

            entity_name = resolved.resolved_name

            # Get data for comparison aspects
            if plan.comparison_aspects:
                for aspect in plan.comparison_aspects:
                    edges = self.graph.explore_neighbors(entity_name, aspect)
                    for edge in edges[:3]:
                        chunks = self.graph.get_chunks_by_edge(
                            entity_name=entity_name,
                            edge_type=edge["type"],
                            limit=2,
                        )
                        for chunk in chunks:
                            # Tag with entity for comparison
                            chunk.header_path = f"[{entity_name}] {chunk.header_path}"
                        result.chunks.extend(chunks)
            else:
                # No specific aspects - get general info
                edges = self.graph.explore_neighbors(entity_name, "")
                for edge in edges[:3]:
                    chunks = self.graph.get_chunks_by_edge(
                        entity_name=entity_name,
                        edge_type=edge["type"],
                        limit=2,
                    )
                    for chunk in chunks:
                        chunk.header_path = f"[{entity_name}] {chunk.header_path}"
                    result.chunks.extend(chunks)

        result.total_candidates_found = len(result.chunks)
        return result


class TemporalPattern(BaseRetrievalPattern):
    """
    Pattern for TEMPORAL queries.

    Flow: resolve entity → explore with temporal awareness → filter by date

    Example: "What happened in October 2025?"
             → explore all edges → filter by date → get chunks
    """

    @property
    def name(self) -> str:
        return "temporal"

    def execute(self, plan: QueryPlan) -> RetrievalResult:
        result = RetrievalResult(plan=plan, retrieval_pattern_used=self.name)

        if plan.anchor_entities:
            # Entity-scoped temporal query
            for entity_query in plan.anchor_entities:
                resolved = self.graph.resolve_entity(entity_query)
                result.resolved_entities.append(resolved)

                if not resolved.resolved_name:
                    continue

                entity_name = resolved.resolved_name
                edges = self.graph.explore_neighbors(entity_name, "")

                for edge in edges[:10]:
                    chunks = self.graph.get_chunks_by_edge(
                        entity_name=entity_name,
                        edge_type=edge["type"],
                        limit=2,
                    )
                    result.chunks.extend(chunks)
        else:
            # Global temporal query - use fallback search
            for term in plan.fallback_search_terms:
                facts = self.graph.search_relationships(term, top_k=10)
                for fact in facts:
                    chunk = self.graph.get_chunk_by_id(fact.get("chunk_id"))
                    if chunk:
                        result.chunks.append(chunk)

        result.total_candidates_found = len(result.chunks)
        return result


class GlobalThemePattern(BaseRetrievalPattern):
    """
    Pattern for GLOBAL_THEME queries.

    Flow: search facts directly → aggregate results

    Example: "What are the main economic trends?"
             → search_relationships with various terms → aggregate
    """

    @property
    def name(self) -> str:
        return "global_theme"

    def execute(self, plan: QueryPlan) -> RetrievalResult:
        result = RetrievalResult(plan=plan, retrieval_pattern_used=self.name)

        seen_chunk_ids = set()

        # Search using fallback terms (fact-level search)
        for search_term in plan.fallback_search_terms:
            facts = self.graph.search_relationships(search_term, top_k=10)

            for fact in facts:
                chunk_id = fact.get("chunk_id")
                if not chunk_id or chunk_id in seen_chunk_ids:
                    continue

                chunk = self.graph.get_chunk_by_id(chunk_id)
                if chunk:
                    chunk.relevance_score = fact.get("score", 0)
                    result.chunks.append(chunk)
                    seen_chunk_ids.add(chunk_id)

        # Also try anchor entities if provided
        for entity_query in plan.anchor_entities[:2]:
            resolved = self.graph.resolve_entity(entity_query)
            result.resolved_entities.append(resolved)

            if resolved.resolved_name:
                edges = self.graph.explore_neighbors(resolved.resolved_name, "")
                for edge in edges[:3]:
                    chunks = self.graph.get_chunks_by_edge(
                        entity_name=resolved.resolved_name,
                        edge_type=edge["type"],
                        limit=2,
                    )
                    for chunk in chunks:
                        if chunk.chunk_id not in seen_chunk_ids:
                            result.chunks.append(chunk)
                            seen_chunk_ids.add(chunk.chunk_id)

        # Sort by relevance
        result.chunks.sort(key=lambda c: c.relevance_score, reverse=True)
        result.total_candidates_found = len(result.chunks)
        return result


class MultiHopPattern(BaseRetrievalPattern):
    """
    Pattern for MULTI_HOP queries.

    Flow: resolve anchor → find first-hop relationships → explore second hop

    Example: "What sectors in declining districts saw growth?"
             → find declining districts → find sectors in each
    """

    @property
    def name(self) -> str:
        return "multi_hop"

    def execute(self, plan: QueryPlan) -> RetrievalResult:
        result = RetrievalResult(plan=plan, retrieval_pattern_used=self.name)

        for entity_query in plan.anchor_entities:
            resolved = self.graph.resolve_entity(entity_query)
            result.resolved_entities.append(resolved)

            if not resolved.resolved_name:
                continue

            entity_name = resolved.resolved_name

            # First hop
            if plan.target_relationship:
                first_edges = self.graph.get_edges_by_type(
                    entity_name=entity_name,
                    edge_type=plan.target_relationship,
                    direction=plan.relationship_direction or "outgoing",
                )
            else:
                first_edges = self.graph.explore_neighbors(entity_name, "")

            # For each first-hop target, explore second hop
            for edge in first_edges[:5]:
                target = edge["target"]

                # Get chunk for first hop
                chunks = self.graph.get_chunk(
                    entity_one=entity_name,
                    entity_two=target,
                    edge_type=edge["type"],
                )
                if chunks:
                    result.chunks.extend(chunks if isinstance(chunks, list) else [chunks])

                # Second hop exploration
                if plan.target_entity_type:
                    second_edges = self.graph.explore_neighbors(target, plan.target_entity_type)
                else:
                    second_edges = self.graph.explore_neighbors(target, "")

                for second_edge in second_edges[:2]:
                    second_chunks = self.graph.get_chunk(
                        entity_one=target,
                        entity_two=second_edge["target"],
                        edge_type=second_edge["type"],
                    )
                    if second_chunks:
                        result.chunks.extend(
                            second_chunks if isinstance(second_chunks, list) else [second_chunks]
                        )

        result.total_candidates_found = len(result.chunks)
        return result


class FallbackSearchPattern(BaseRetrievalPattern):
    """
    Fallback pattern when other patterns fail or for UNKNOWN query types.

    Flow: search facts directly with various query formulations
    """

    @property
    def name(self) -> str:
        return "fallback_search"

    def execute(self, plan: QueryPlan) -> RetrievalResult:
        result = RetrievalResult(plan=plan, retrieval_pattern_used=self.name, fallback_used=True)

        seen_chunk_ids = set()

        # Try all fallback search terms
        for search_term in plan.fallback_search_terms:
            facts = self.graph.search_relationships(search_term, top_k=5)

            for fact in facts:
                chunk_id = fact.get("chunk_id")
                if chunk_id and chunk_id not in seen_chunk_ids:
                    chunk = self.graph.get_chunk_by_id(chunk_id)
                    if chunk:
                        chunk.relevance_score = fact.get("score", 0)
                        result.chunks.append(chunk)
                        seen_chunk_ids.add(chunk_id)

        # Also try resolving entities and exploring
        for entity_query in plan.anchor_entities:
            resolved = self.graph.resolve_entity(entity_query)
            result.resolved_entities.append(resolved)

            if resolved.resolved_name:
                edges = self.graph.explore_neighbors(resolved.resolved_name, "")
                for edge in edges[:3]:
                    chunks = self.graph.get_chunks_by_edge(
                        entity_name=resolved.resolved_name,
                        edge_type=edge["type"],
                        limit=1,
                    )
                    for chunk in chunks:
                        if chunk.chunk_id not in seen_chunk_ids:
                            result.chunks.append(chunk)
                            seen_chunk_ids.add(chunk.chunk_id)

        result.total_candidates_found = len(result.chunks)
        return result
