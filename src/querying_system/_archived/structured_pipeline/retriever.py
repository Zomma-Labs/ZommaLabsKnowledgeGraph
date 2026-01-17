"""
Step 2: Deterministic Retriever
Routes queries to appropriate retrieval patterns and executes them.
"""

import time
from typing import Optional

from .models import QueryPlan, QueryType, RetrievalResult, RetrievedChunk, ResolvedEntity
from .retrieval_patterns import (
    EntityAttributePattern,
    EntityRelationshipPattern,
    ComparisonPattern,
    TemporalPattern,
    GlobalThemePattern,
    MultiHopPattern,
    FallbackSearchPattern,
)

# Import existing MCP logic functions directly
from src.querying_system.mcp_server import (
    _resolve_entity_or_topic_logic,
    _get_entity_info_logic,
    _explore_neighbors_logic,
    _explore_neighbors_semantic_logic,
    _get_chunk_logic,
    _get_chunks_by_edge_logic,
    _search_relationships_logic,
)


class GraphOperations:
    """
    Wrapper around MCP server logic functions.
    Provides a clean interface for retrieval patterns.
    """

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id

    def resolve_entity(self, query: str, context: str = "") -> ResolvedEntity:
        """Resolve an entity name to its canonical form."""
        result = _resolve_entity_or_topic_logic(query, "Entity", self.user_id, context)

        if result["found"]:
            return ResolvedEntity(
                query=query,
                resolved_name=result["results"][0],
                match_type="semantic",
                score=0.9,
                alternatives=result["results"][1:5],
            )
        else:
            return ResolvedEntity(
                query=query,
                resolved_name=None,
                match_type="not_found",
                score=0.0,
                alternatives=[],
            )

    def resolve_topic(self, query: str) -> list[str]:
        """Resolve a topic name."""
        result = _resolve_entity_or_topic_logic(query, "Topic", self.user_id)
        return result["results"] if result["found"] else []

    def get_entity_info(self, entity_name: str) -> Optional[dict]:
        """Get entity metadata."""
        result = _get_entity_info_logic(entity_name, self.user_id)
        return result if result.get("found") else None

    def explore_neighbors(self, entity_name: str, query_hint: str = "") -> list[dict]:
        """Explore relationships from an entity."""
        if query_hint:
            result_str = _explore_neighbors_semantic_logic(entity_name, self.user_id, query_hint)
        else:
            result_str = _explore_neighbors_logic(entity_name, self.user_id)

        return self._parse_edges_str(result_str)

    def get_chunk(
        self, entity_one: str, entity_two: str, edge_type: str
    ) -> Optional[RetrievedChunk]:
        """Get a single chunk for a relationship."""
        result = _get_chunk_logic(entity_one, entity_two, edge_type, self.user_id)

        if result.get("found"):
            return self._parse_chunk_str(
                result["chunk"],
                source_entity=entity_one,
                target_entity=entity_two,
                edge_type=edge_type,
            )
        return None

    def get_chunks_by_edge(
        self,
        entity_name: str,
        edge_type: str,
        direction: str = "both",
        limit: int = 10,
    ) -> list[RetrievedChunk]:
        """Get all chunks for an entity + edge type."""
        result = _get_chunks_by_edge_logic(entity_name, edge_type, self.user_id, direction)

        if not result.get("found"):
            return []

        chunks = []
        for item in result["results"][:limit]:
            chunk = self._parse_chunk_str(
                item["chunk"],
                source_entity=entity_name,
                edge_type=edge_type,
                direction=item.get("direction", direction),
            )
            if chunk:
                chunks.append(chunk)

        return chunks

    def get_edges_by_type(
        self,
        entity_name: str,
        edge_type: str,
        direction: str = "outgoing",
    ) -> list[dict]:
        """Get edges of a specific type from an entity."""
        all_edges = self.explore_neighbors(entity_name, "")
        return [
            e
            for e in all_edges
            if e.get("type") == edge_type and e.get("direction", "both") in [direction, "both"]
        ]

    def search_relationships(self, query: str, top_k: int = 10) -> list[dict]:
        """Search for facts/relationships by semantic similarity."""
        result = _search_relationships_logic(query, self.user_id, top_k)
        return result.get("results", []) if result.get("found") else []

    def get_chunk_by_id(self, chunk_id: str) -> Optional[RetrievedChunk]:
        """Get a chunk by its ID."""
        from src.util.services import get_services

        services = get_services()

        result = services.neo4j.query(
            """
            MATCH (c:EpisodicNode {uuid: $chunk_id, group_id: $uid})
            OPTIONAL MATCH (d:DocumentNode)-[:CONTAINS_CHUNK]->(c)
            RETURN c, d.name as doc_name
        """,
            {"chunk_id": chunk_id, "uid": self.user_id},
        )

        if not result:
            return None

        chunk_data = result[0]["c"]
        return RetrievedChunk(
            chunk_id=chunk_data.get("uuid", chunk_id),
            doc_id=result[0].get("doc_name") or chunk_data.get("doc_id", "unknown"),
            content=chunk_data.get("content", ""),
            header_path=chunk_data.get("header_path", ""),
            doc_date=chunk_data.get("created_at"),
        )

    def _parse_edges_str(self, edges_str: str) -> list[dict]:
        """Parse the edge list string from explore_neighbors."""
        edges = []

        for line in edges_str.split("\n"):
            if not line.startswith("- "):
                continue

            line = line[2:]  # Remove "- "

            # Determine direction
            if " → " in line:
                parts = line.split(" → ")
                direction = "outgoing"
            elif " ← " in line:
                parts = line.split(" ← ")
                direction = "incoming"
            else:
                continue

            if len(parts) < 2:
                continue

            edge_type = parts[0].strip()
            rest = parts[1]

            # Parse target and optional date/score
            target = rest
            score = None

            if " | " in rest:
                target, date_part = rest.split(" | ", 1)
                target = target.strip()

                # Check for score in parentheses
                if "(" in date_part and ")" in date_part:
                    score_str = date_part.split("(")[1].rstrip(")")
                    try:
                        score = float(score_str)
                    except ValueError:
                        pass

            edge = {
                "type": edge_type,
                "target": target,
                "direction": direction,
            }

            if score is not None:
                edge["score"] = score

            edges.append(edge)

        return edges

    def _parse_chunk_str(
        self,
        chunk_str: str,
        source_entity: str = "",
        target_entity: str = "",
        edge_type: str = "",
        direction: str = "",
    ) -> Optional[RetrievedChunk]:
        """Parse the formatted chunk string into a RetrievedChunk object."""
        if not chunk_str:
            return None

        lines = chunk_str.strip().split("\n")

        doc_id = ""
        chunk_id = ""
        header_path = ""
        date = ""
        content_lines = []

        in_content = False

        for line in lines:
            if line.startswith('"""'):
                in_content = not in_content
                continue

            if not in_content:
                continue

            if line.startswith("DOCUMENT:"):
                doc_id = line.replace("DOCUMENT:", "").strip()
            elif line.startswith("CHUNK_id:") or line.startswith("CHUNK_ID:"):
                chunk_id = line.replace("CHUNK_id:", "").replace("CHUNK_ID:", "").strip()
            elif line.startswith("Header:") or line.startswith("HEADER:"):
                header_path = line.replace("Header:", "").replace("HEADER:", "").strip()
            elif line.startswith("Date:") or line.startswith("DATE:"):
                date = line.replace("Date:", "").replace("DATE:", "").strip()
            elif line.startswith("RELATIONSHIP:"):
                # Skip - we have this info already
                pass
            else:
                content_lines.append(line)

        content = "\n".join(content_lines).strip()

        return RetrievedChunk(
            chunk_id=chunk_id or "unknown",
            doc_id=doc_id or "unknown",
            content=content,
            header_path=header_path,
            source_entity=source_entity,
            target_entity=target_entity,
            edge_type=edge_type,
            direction=direction,
            fact_date=date if date and date != "N/A" else None,
        )


class Retriever:
    """
    Main retriever class that routes queries to appropriate patterns.
    """

    def __init__(self, user_id: str = "default"):
        self.graph = GraphOperations(user_id)

        # Initialize all patterns
        self.patterns = {
            QueryType.ENTITY_ATTRIBUTE: EntityAttributePattern(self.graph),
            QueryType.ENTITY_RELATIONSHIP: EntityRelationshipPattern(self.graph),
            QueryType.COMPARISON: ComparisonPattern(self.graph),
            QueryType.TEMPORAL: TemporalPattern(self.graph),
            QueryType.GLOBAL_THEME: GlobalThemePattern(self.graph),
            QueryType.MULTI_HOP: MultiHopPattern(self.graph),
            QueryType.UNKNOWN: FallbackSearchPattern(self.graph),
        }

        self.fallback_pattern = FallbackSearchPattern(self.graph)

    def retrieve(self, plan: QueryPlan) -> tuple[RetrievalResult, int]:
        """
        Execute retrieval based on the query plan.

        Args:
            plan: QueryPlan from the analyzer

        Returns:
            tuple of (RetrievalResult, elapsed_time_ms)
        """
        start_time = time.time()

        # Get the appropriate pattern
        pattern = self.patterns.get(plan.query_type, self.fallback_pattern)

        # Execute the pattern
        result = pattern.execute(plan)

        # If no results and confidence is low, try fallback
        if not result.chunks and plan.confidence < 0.7:
            fallback_result = self.fallback_pattern.execute(plan)
            if fallback_result.chunks:
                result = fallback_result

        # If still no results, try a broader fallback
        if not result.chunks and plan.fallback_search_terms:
            result = self.fallback_pattern.execute(plan)

        # Deduplicate chunks by chunk_id
        seen_ids = set()
        unique_chunks = []
        for chunk in result.chunks:
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                unique_chunks.append(chunk)
        result.chunks = unique_chunks

        elapsed_ms = int((time.time() - start_time) * 1000)

        return result, elapsed_ms
