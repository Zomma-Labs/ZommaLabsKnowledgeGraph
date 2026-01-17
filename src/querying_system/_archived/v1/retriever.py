"""
Phase 2a: Hybrid Multi-Strategy Retrieval (Optimized).

Key improvements:
- All embeddings batched upfront
- Scoped + global searches run IN PARALLEL
- No sequential awaits in loops
"""

import asyncio
import os
from collections import defaultdict

from src.querying_system.shared.schemas import QueryDecomposition, ScoredFact, EvidencePool
from src.util.services import get_services
from src.util.deterministic_retrieval import extract_keywords

VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"


def log(msg: str):
    if VERBOSE:
        print(f"[HybridRetriever] {msg}")


class HybridRetriever:
    """
    Multi-strategy parallel retriever with entity+topic scoping.

    Optimized for speed:
    1. Batch all embeddings upfront (single API call)
    2. Run ALL searches in parallel (scoped + global + subquery)
    3. No sequential awaits in loops
    """

    def __init__(self, group_id: str = "default"):
        self.group_id = group_id
        self.services = get_services()

    async def retrieve(
        self, decomposition: QueryDecomposition, top_k_per_query: int = 15
    ) -> EvidencePool:
        """
        Execute retrieval with all searches running in parallel.
        """
        entities = decomposition.entity_hints or []
        topics = decomposition.topic_hints or []
        query_texts = [sq.query_text for sq in decomposition.sub_queries]

        log(f"Entities: {len(entities)}, Topics: {len(topics)}, Sub-queries: {len(query_texts)}")

        # Step 1: Batch ALL embeddings upfront (topics + query texts)
        all_texts_to_embed = list(set(topics + query_texts))
        log(f"Embedding {len(all_texts_to_embed)} texts...")

        all_embeddings = await self._batch_embed(all_texts_to_embed)
        embedding_map = {text: emb for text, emb in zip(all_texts_to_embed, all_embeddings)}

        # Step 2: Build ALL tasks (scoped + global + subquery) - no awaits here
        all_tasks: list[tuple[str, any]] = []

        # Scoped tasks: entity × topic combinations
        if entities and topics:
            for entity in entities:
                for topic in topics:
                    coro = self._scoped_entity_topic_search(entity, topic, top_k_per_query)
                    all_tasks.append((f"scoped:{entity}+{topic}", coro))

            # Topic vector search
            for topic in topics:
                if topic in embedding_map:
                    coro = self._topic_vector_search(embedding_map[topic], topic, top_k_per_query)
                    all_tasks.append((f"topic_vec:{topic}", coro))

        elif entities:
            for entity in entities:
                coro = self._entity_graph_search([entity], top_k_per_query)
                all_tasks.append((f"entity:{entity}", coro))

        elif topics:
            for topic in topics:
                if topic in embedding_map:
                    coro = self._topic_vector_search(embedding_map[topic], topic, top_k_per_query)
                    all_tasks.append((f"topic:{topic}", coro))

        # Global tasks: vector + keyword for each sub-query
        for sq in decomposition.sub_queries:
            if sq.query_text in embedding_map:
                coro = self._vector_search(embedding_map[sq.query_text], top_k_per_query)
                all_tasks.append((f"global_vec:{sq.query_text[:30]}", coro))

            coro = self._keyword_search(sq.query_text, top_k_per_query)
            all_tasks.append((f"global_kw:{sq.query_text[:30]}", coro))

            # Entity hints from sub-query
            if sq.entity_hints:
                coro = self._entity_graph_search(sq.entity_hints, top_k_per_query // 2)
                all_tasks.append((f"subq_entity:{sq.query_text[:20]}", coro))

        # Step 3: Run ALL searches in parallel
        log(f"Running {len(all_tasks)} searches IN PARALLEL...")
        all_results = await asyncio.gather(*[t[1] for t in all_tasks], return_exceptions=True)

        # Step 4: Process results
        all_facts: dict[str, ScoredFact] = {}
        query_to_facts: dict[str, list[str]] = defaultdict(list)

        for (query_name, _), results in zip(all_tasks, all_results):
            # Skip exceptions
            if isinstance(results, Exception):
                log(f"Search {query_name} failed: {results}")
                continue

            for r in results:
                # Handle both dict and ScoredFact
                if isinstance(r, ScoredFact):
                    fact_id = r.fact_id
                    fact = r
                else:
                    fact_id = r.get("fact_id")
                    if not fact_id:
                        continue
                    fact = self._dict_to_scored_fact(r)

                if fact_id not in all_facts:
                    all_facts[fact_id] = fact

                if query_name not in all_facts[fact_id].found_by_queries:
                    all_facts[fact_id].found_by_queries.append(query_name)
                query_to_facts[query_name].append(fact_id)

        # Step 5: Calculate cross-query boost
        num_queries = len(all_tasks)
        for fact in all_facts.values():
            queries_found = len(fact.found_by_queries)
            if num_queries > 0:
                fact.cross_query_boost = min(1.0, (queries_found - 1) * 0.3)
            fact.rrf_score = fact.vector_score

        # Build coverage map
        coverage_map = {}
        for info in decomposition.required_info:
            covering_facts = []
            info_lower = info.lower()
            info_words = set(info_lower.split()[:5])
            for fact_id, fact in all_facts.items():
                content_lower = fact.content.lower()
                if any(word in content_lower for word in info_words if len(word) > 3):
                    covering_facts.append(fact_id)
            coverage_map[info] = covering_facts

        log(
            f"Retrieved {len(all_facts)} unique facts, "
            f"multi-query hits: {sum(1 for f in all_facts.values() if len(f.found_by_queries) > 1)}"
        )

        return EvidencePool(
            scored_facts=list(all_facts.values()),
            coverage_map=coverage_map,
            entities_found=list(
                set(f.subject for f in all_facts.values() if f.subject)
                | set(f.object for f in all_facts.values() if f.object)
            ),
            expansion_performed=False,
        )

    def _dict_to_scored_fact(self, r: dict) -> ScoredFact:
        """Convert a dict result to ScoredFact."""
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
        """Embed multiple texts in one API call."""
        if not texts:
            return []
        return await asyncio.to_thread(
            self.services.embeddings.embed_documents, texts
        )

    async def _scoped_entity_topic_search(
        self, entity: str, topic: str, top_k: int
    ) -> list[dict]:
        """Search facts connected to an entity AND related to a topic."""
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
        """Vector search filtered by topic keyword."""
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
        """Global vector search on fact embeddings."""
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
        """BM25 keyword search on fact content."""
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
        """Direct graph traversal from known entities."""
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
