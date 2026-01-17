"""
Phase 2b: Retrieval with Resolution-Based Search.

Flow:
1. Takes resolved entity/topic nodes from Resolver
2. Does vector search localized around each resolved node
3. Returns facts connected to those nodes
"""

import asyncio
import os
from dataclasses import dataclass

from .schemas import QueryDecomposition, ScoredFact
from .resolver import ResolvedEntities
from src.util.services import get_services
from src.util.deterministic_retrieval import extract_keywords
from src.util.fact_vector_store import get_fact_store

VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"


def log(msg: str):
    if VERBOSE:
        print(f"[HybridRetrieverV2] {msg}")


@dataclass
class ScopedRetrievalResult:
    """Separates scoped (high confidence) from global (coverage) results."""
    scoped_facts: dict[str, ScoredFact]
    global_facts: dict[str, ScoredFact]
    all_facts: dict[str, ScoredFact]


class HybridRetrieverV2:
    """
    Parallel retriever with separated scoped/global results.

    All searches (scoped + global) run in parallel for speed.
    """

    def __init__(self, group_id: str = "default"):
        self.group_id = group_id
        self.services = get_services()

    async def retrieve_separated(
        self, decomposition: QueryDecomposition, top_k_per_query: int = 15
    ) -> ScopedRetrievalResult:
        """
        Retrieve with scoped and global searches running IN PARALLEL.
        """
        entities = decomposition.entity_hints or []
        topics = decomposition.topic_hints or []
        query_texts = [sq.query_text for sq in decomposition.sub_queries]

        log(f"Entities: {len(entities)}, Topics: {len(topics)}, Sub-queries: {len(query_texts)}")

        # Step 1: Batch ALL embeddings upfront (topics + query texts)
        all_texts_to_embed = list(set(topics + query_texts))
        log(f"Embedding {len(all_texts_to_embed)} texts...")

        all_embeddings = await self._batch_embed(all_texts_to_embed)

        # Create lookup for embeddings
        embedding_map = {text: emb for text, emb in zip(all_texts_to_embed, all_embeddings)}

        # Step 2: Build ALL tasks (scoped + global)
        scoped_task_list = []  # (name, coroutine)
        global_task_list = []  # (name, coroutine)

        # Scoped tasks: entity × topic combinations
        if entities and topics:
            for entity in entities:
                for topic in topics:
                    coro = self._scoped_entity_topic_search(entity, topic, top_k_per_query)
                    scoped_task_list.append((f"scoped:{entity}+{topic}", coro))

            for topic in topics:
                if topic in embedding_map:
                    coro = self._topic_vector_search(embedding_map[topic], topic, top_k_per_query)
                    scoped_task_list.append((f"topic_vec:{topic}", coro))

        elif entities:
            for entity in entities:
                coro = self._entity_graph_search([entity], top_k_per_query)
                scoped_task_list.append((f"entity:{entity}", coro))

        elif topics:
            for topic in topics:
                if topic in embedding_map:
                    coro = self._topic_vector_search(embedding_map[topic], topic, top_k_per_query)
                    scoped_task_list.append((f"topic:{topic}", coro))

        # Global tasks: vector + keyword search for each sub-query
        for sq in decomposition.sub_queries:
            if sq.query_text in embedding_map:
                coro = self._vector_search(embedding_map[sq.query_text], top_k_per_query)
                global_task_list.append((f"global_vec:{sq.query_text[:30]}", coro))

            coro = self._keyword_search(sq.query_text, top_k_per_query)
            global_task_list.append((f"global_kw:{sq.query_text[:30]}", coro))

        # Step 3: Run ALL searches in parallel
        all_tasks = scoped_task_list + global_task_list
        log(f"Running {len(scoped_task_list)} scoped + {len(global_task_list)} global searches IN PARALLEL...")

        all_results = await asyncio.gather(*[t[1] for t in all_tasks])

        # Step 4: Process results - separate scoped from global
        scoped_facts: dict[str, ScoredFact] = {}
        global_facts: dict[str, ScoredFact] = {}

        num_scoped = len(scoped_task_list)

        # Process scoped results
        for i, ((query_name, _), results) in enumerate(zip(all_tasks[:num_scoped], all_results[:num_scoped])):
            for r in results:
                fact_id = r.get("fact_id")
                if not fact_id or fact_id in scoped_facts:
                    continue
                fact = self._dict_to_scored_fact(r)
                fact.found_by_queries.append(query_name)
                scoped_facts[fact_id] = fact

        # Process global results
        for (query_name, _), results in zip(all_tasks[num_scoped:], all_results[num_scoped:]):
            for r in results:
                fact_id = r.get("fact_id")
                if not fact_id:
                    continue

                # If already in scoped, add to found_by_queries but don't duplicate
                if fact_id in scoped_facts:
                    if query_name not in scoped_facts[fact_id].found_by_queries:
                        scoped_facts[fact_id].found_by_queries.append(query_name)
                    continue

                if fact_id not in global_facts:
                    fact = self._dict_to_scored_fact(r)
                    fact.found_by_queries.append(query_name)
                    global_facts[fact_id] = fact
                else:
                    if query_name not in global_facts[fact_id].found_by_queries:
                        global_facts[fact_id].found_by_queries.append(query_name)

        # Calculate cross-query boost
        all_facts = {**scoped_facts, **global_facts}
        num_queries = len(all_tasks)
        for fact in all_facts.values():
            queries_found = len(fact.found_by_queries)
            if num_queries > 0:
                fact.cross_query_boost = min(1.0, (queries_found - 1) * 0.3)
            fact.rrf_score = fact.vector_score

        log(
            f"Retrieved {len(all_facts)} facts "
            f"(scoped: {len(scoped_facts)}, global: {len(global_facts)})"
        )

        return ScopedRetrievalResult(
            scoped_facts=scoped_facts,
            global_facts=global_facts,
            all_facts=all_facts,
        )

    def _dict_to_scored_fact(self, r: dict) -> ScoredFact:
        return ScoredFact(
            fact_id=r.get("fact_id", ""),
            content=r.get("content", ""),
            subject=r.get("subject", ""),
            edge_type=r.get("edge_type", ""),
            object=r.get("object", ""),
            chunk_id=r.get("chunk_id"),
            chunk_content=r.get("chunk_content"),
            chunk_header=r.get("chunk_header"),
            doc_id=r.get("doc_id"),
            document_date=r.get("document_date"),
            vector_score=r.get("score", 0.0),
        )

    async def _batch_embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(
            self.services.embeddings.embed_documents, texts
        )

    async def _scoped_entity_topic_search(
        self, entity: str, topic: str, top_k: int
    ) -> list[dict]:
        def _query():
            return self.services.neo4j.query(
                """
                CALL db.index.fulltext.queryNodes('entity_fulltext', $entity_name)
                YIELD node as e, score as match_score
                WHERE (e:EntityNode OR e:TopicNode) AND e.group_id = $uid AND match_score > 0.5
                WITH e LIMIT 3

                MATCH (e)-[r1]->(c:EpisodicNode {group_id: $uid})-[r2]->(target)
                WHERE (target:EntityNode OR target:TopicNode) AND r1.fact_id = r2.fact_id

                MATCH (f:FactNode {uuid: r1.fact_id, group_id: $uid})

                WHERE toLower(f.content) CONTAINS toLower($topic)
                   OR toLower(target.name) CONTAINS toLower($topic)
                   OR toLower(e.name) CONTAINS toLower($topic)

                OPTIONAL MATCH (d:DocumentNode {group_id: $uid})-[:CONTAINS_CHUNK]->(c)

                RETURN DISTINCT f.uuid as fact_id,
                       f.content as content,
                       1.0 as score,
                       e.name as subject,
                       type(r1) as edge_type,
                       target.name as object,
                       c.uuid as chunk_id,
                       c.content as chunk_content,
                       c.header_path as chunk_header,
                       d.name as doc_id,
                       d.document_date as document_date
                LIMIT $top_k
                """,
                {"entity_name": entity, "topic": topic, "uid": self.group_id, "top_k": top_k},
            )
        results = await asyncio.to_thread(_query)
        return [dict(r) for r in results]

    async def _topic_vector_search(
        self, embedding: list[float], topic: str, top_k: int
    ) -> list[dict]:
        def _query():
            return self.services.neo4j.query(
                """
                CALL db.index.vector.queryNodes('fact_embeddings', $top_k * 2, $vec)
                YIELD node, score
                WHERE node.group_id = $uid AND score > 0.25
                  AND toLower(node.content) CONTAINS toLower($topic)

                OPTIONAL MATCH (subj)-[r1 {fact_id: node.uuid}]->(c:EpisodicNode {group_id: $uid})-[r2 {fact_id: node.uuid}]->(obj)
                WHERE (subj:EntityNode OR subj:TopicNode) AND (obj:EntityNode OR obj:TopicNode)

                OPTIONAL MATCH (d:DocumentNode {group_id: $uid})-[:CONTAINS_CHUNK]->(c)

                RETURN node.uuid as fact_id,
                       node.content as content,
                       score,
                       subj.name as subject,
                       type(r1) as edge_type,
                       obj.name as object,
                       c.uuid as chunk_id,
                       c.content as chunk_content,
                       c.header_path as chunk_header,
                       d.name as doc_id,
                       d.document_date as document_date
                ORDER BY score DESC
                LIMIT $top_k
                """,
                {"vec": embedding, "uid": self.group_id, "top_k": top_k, "topic": topic},
            )
        results = await asyncio.to_thread(_query)
        return [dict(r) for r in results]

    async def _vector_search(self, embedding: list[float], top_k: int) -> list[dict]:
        def _query():
            return self.services.neo4j.query(
                """
                CALL db.index.vector.queryNodes('fact_embeddings', $top_k, $vec)
                YIELD node, score
                WHERE node.group_id = $uid AND score > 0.25

                OPTIONAL MATCH (subj)-[r1 {fact_id: node.uuid}]->(c:EpisodicNode {group_id: $uid})-[r2 {fact_id: node.uuid}]->(obj)
                WHERE (subj:EntityNode OR subj:TopicNode) AND (obj:EntityNode OR obj:TopicNode)

                OPTIONAL MATCH (d:DocumentNode {group_id: $uid})-[:CONTAINS_CHUNK]->(c)

                RETURN node.uuid as fact_id,
                       node.content as content,
                       score,
                       subj.name as subject,
                       type(r1) as edge_type,
                       obj.name as object,
                       c.uuid as chunk_id,
                       c.content as chunk_content,
                       c.header_path as chunk_header,
                       d.name as doc_id,
                       d.document_date as document_date
                ORDER BY score DESC
                """,
                {"vec": embedding, "uid": self.group_id, "top_k": top_k},
            )
        results = await asyncio.to_thread(_query)
        return [dict(r) for r in results]

    async def _keyword_search(self, query: str, top_k: int) -> list[dict]:
        keywords = extract_keywords(query)
        if not keywords:
            return []

        def _query():
            return self.services.neo4j.query(
                """
                CALL db.index.fulltext.queryNodes('fact_fulltext', $keywords)
                YIELD node, score
                WHERE node.group_id = $uid

                OPTIONAL MATCH (subj)-[r1 {fact_id: node.uuid}]->(c:EpisodicNode {group_id: $uid})-[r2 {fact_id: node.uuid}]->(obj)
                WHERE (subj:EntityNode OR subj:TopicNode) AND (obj:EntityNode OR obj:TopicNode)

                OPTIONAL MATCH (d:DocumentNode {group_id: $uid})-[:CONTAINS_CHUNK]->(c)

                RETURN node.uuid as fact_id,
                       node.content as content,
                       score,
                       subj.name as subject,
                       type(r1) as edge_type,
                       obj.name as object,
                       c.uuid as chunk_id,
                       c.content as chunk_content,
                       c.header_path as chunk_header,
                       d.name as doc_id,
                       d.document_date as document_date
                ORDER BY score DESC
                LIMIT $top_k
                """,
                {"keywords": keywords, "uid": self.group_id, "top_k": top_k},
            )
        results = await asyncio.to_thread(_query)
        return [dict(r) for r in results]

    async def _entity_graph_search(self, entities: list[str], top_k: int) -> list[dict]:
        def _query():
            return self.services.neo4j.query(
                """
                UNWIND $entities as entity_name

                CALL db.index.fulltext.queryNodes('entity_fulltext', entity_name)
                YIELD node as e, score as match_score
                WHERE e.group_id = $uid AND match_score > 0.5

                WITH e LIMIT 3

                MATCH (e)-[r1]->(c:EpisodicNode {group_id: $uid})-[r2]->(target)
                WHERE (target:EntityNode OR target:TopicNode) AND r1.fact_id = r2.fact_id

                OPTIONAL MATCH (f:FactNode {uuid: r1.fact_id, group_id: $uid})
                OPTIONAL MATCH (d:DocumentNode {group_id: $uid})-[:CONTAINS_CHUNK]->(c)

                RETURN DISTINCT r1.fact_id as fact_id,
                       COALESCE(f.content, e.name + ' ' + type(r1) + ' ' + target.name) as content,
                       1.0 as score,
                       e.name as subject,
                       type(r1) as edge_type,
                       target.name as object,
                       c.uuid as chunk_id,
                       c.content as chunk_content,
                       c.header_path as chunk_header,
                       d.name as doc_id,
                       d.document_date as document_date
                LIMIT $top_k
                """,
                {"entities": entities, "uid": self.group_id, "top_k": top_k},
            )
        results = await asyncio.to_thread(_query)
        return [dict(r) for r in results]

    async def retrieve_from_resolved(
        self,
        resolved: ResolvedEntities,
        decomposition: QueryDecomposition,
        question: str = "",
        top_k_per_node: int = 10
    ) -> dict[str, ScoredFact]:
        """
        Get facts connected to resolved nodes using question-based vector search.

        For each resolved entity/topic:
        1. Use the full question as the search query (provides full semantic context)
        2. Vector search on fact_embeddings
        3. Filter to facts connected to the resolved node
        """
        # Use the full question as search query - it provides the richest semantic context
        # e.g., "Which districts reported slight to modest economic growth?" tells us:
        # - We want districts (subject)
        # - With a specific relationship (reported slight to modest)
        # - To a topic (economic growth)
        search_query = question if question else " ".join(
            sq.query_text for sq in decomposition.sub_queries[:2]
        )
        log(f"Search query: '{search_query}'")

        # Embed the search query
        query_embedding = None
        if search_query:
            query_embedding = await asyncio.to_thread(
                self.services.embeddings.embed_query, search_query
            )

        # Collect all nodes to search
        nodes_to_search = []
        for entity_name in resolved.entity_nodes:
            nodes_to_search.append((f"entity:{entity_name}", entity_name))
        for topic_name in resolved.topic_nodes:
            nodes_to_search.append((f"topic:{topic_name}", topic_name))

        if not nodes_to_search:
            log("No resolved nodes to search around")
            return {}

        log(f"Searching around {len(nodes_to_search)} resolved nodes...")

        # Run Qdrant searches sequentially to avoid concurrency issues
        # (local Qdrant doesn't support concurrent access)
        all_facts: dict[str, ScoredFact] = {}
        for query_name, node_name in nodes_to_search:
            try:
                if query_embedding:
                    results = self._search_facts_for_node_sync(node_name, query_embedding, top_k_per_node)
                else:
                    # Fallback to non-vector search
                    results = await self._get_facts_for_node(node_name, top_k_per_node)

                for r in results:
                    fact_id = r.get("fact_id")
                    if not fact_id or fact_id in all_facts:
                        continue
                    fact = self._dict_to_scored_fact(r)
                    fact.found_by_queries.append(query_name)
                    all_facts[fact_id] = fact
            except Exception as e:
                log(f"Search {query_name} failed: {e}")
                continue

        log(f"Retrieved {len(all_facts)} facts from resolved nodes")
        return all_facts

    def _search_facts_for_node_sync(
        self, node_name: str, query_embedding: list[float], top_k: int
    ) -> list[dict]:
        """
        Synchronous search for facts connected to a node using Qdrant.

        Note: Uses singleton fact_store - must be called sequentially or
        the caller must handle synchronization.
        """
        fact_store = get_fact_store()
        results = fact_store.search_facts_for_entity(
            entity_name=node_name,
            query_embedding=query_embedding,
            group_id=self.group_id,
            top_k=top_k
        )

        # Convert to expected dict format
        return [
            {
                "fact_id": r["fact_id"],
                "content": r["content"],
                "score": r["score"],
                "subject": r["subject"],
                "edge_type": r["edge_type"],
                "object": r["object"],
                "chunk_id": None,
                "chunk_content": None,
                "chunk_header": None,
                "doc_id": None,
                "document_date": None,
            }
            for r in results
        ]

    async def _get_facts_for_node_with_query(
        self, node_name: str, query_embedding: list[float], top_k: int
    ) -> list[dict]:
        """
        Search facts connected to a node using Qdrant vector search.
        Runs synchronously to avoid Qdrant concurrency issues.
        """
        return self._search_facts_for_node_sync(node_name, query_embedding, top_k)

    async def _get_facts_for_node(self, node_name: str, top_k: int) -> list[dict]:
        """
        Get facts connected to a specific node.
        """
        def _query():
            return self.services.neo4j.query(
                """
                MATCH (n {name: $node_name, group_id: $uid})
                WHERE n:EntityNode OR n:TopicNode

                MATCH (n)-[r1]->(c:EpisodicNode {group_id: $uid})-[r2]->(target)
                WHERE r1.fact_id = r2.fact_id AND (target:EntityNode OR target:TopicNode)

                MATCH (f:FactNode {uuid: r1.fact_id, group_id: $uid})

                OPTIONAL MATCH (d:DocumentNode {group_id: $uid})-[:CONTAINS_CHUNK]->(c)

                RETURN DISTINCT f.uuid as fact_id,
                       f.content as content,
                       1.0 as score,
                       n.name as subject,
                       type(r1) as edge_type,
                       target.name as object,
                       c.uuid as chunk_id,
                       c.content as chunk_content,
                       c.header_path as chunk_header,
                       d.name as doc_id,
                       d.document_date as document_date
                LIMIT $top_k
                """,
                {"node_name": node_name, "uid": self.group_id, "top_k": top_k},
            )

        results = await asyncio.to_thread(_query)
        return [dict(r) for r in results]
