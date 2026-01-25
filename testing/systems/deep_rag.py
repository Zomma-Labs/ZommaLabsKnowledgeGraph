"""
MODULE: DeepRAG
DESCRIPTION: Advanced RAG system with multi-query decomposition and RRF fusion.
             Uses query decomposition to generate sub-queries, parallel vector
             search, and Reciprocal Rank Fusion to combine results.
"""

import asyncio
from collections import defaultdict

import numpy as np
import faiss
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from testing.common.utils import load_chunks, EmbeddingCache, EMBEDDING_DIMENSION

load_dotenv()


class DeepRAG:
    """Advanced RAG system with query decomposition and RRF fusion.

    This system improves upon SimpleRAG by:
    - Decomposing complex questions into multiple sub-queries
    - Performing parallel vector search for each sub-query
    - Fusing results using Reciprocal Rank Fusion (RRF)
    - Synthesizing answers from the top fused results

    RRF provides more robust ranking than single-query approaches by
    combining evidence from multiple retrieval perspectives.
    """

    def __init__(
        self,
        chunk_file: str,
        cache_file: str | None = "eval/deep_rag_embeddings.pkl",
    ):
        """Initialize DeepRAG with chunks from a JSONL file.

        Args:
            chunk_file: Path to the JSONL file containing chunks.
            cache_file: Optional path for embedding cache persistence.
        """
        self.chunk_file = chunk_file
        self.chunks: list[dict] = []
        self.embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")
        self.llm = ChatOpenAI(model="gpt-5.1", temperature=0)
        self.cache = EmbeddingCache(cache_file)
        self.index: faiss.IndexFlatIP | None = None
        self.chunk_embeddings: np.ndarray | None = None

        # Load and index
        self._load_chunks()
        self._build_index()

    def _load_chunks(self) -> None:
        """Load chunks from the JSONL file."""
        self.chunks = load_chunks(self.chunk_file)
        print(f"[DeepRAG] Loaded {len(self.chunks)} chunks from {self.chunk_file}")

    def _build_index(self) -> None:
        """Embed all chunks and build FAISS index."""
        if not self.chunks:
            raise ValueError("No chunks loaded - cannot build index")

        # Get chunk texts for embedding
        texts = [chunk["body"] for chunk in self.chunks]

        # Get or compute embeddings with caching
        print(f"[DeepRAG] Computing embeddings for {len(texts)} chunks...")
        self.chunk_embeddings = self.cache.get_or_compute(
            texts,
            self.embeddings_model.embed_documents,
        )

        # Normalize for cosine similarity (IndexFlatIP computes inner product)
        embeddings_normalized = self.chunk_embeddings.copy()
        faiss.normalize_L2(embeddings_normalized)

        # Build FAISS index
        self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        self.index.add(embeddings_normalized)

        print(f"[DeepRAG] Built FAISS index with {self.index.ntotal} vectors")

    async def _decompose_query(self, question: str) -> list[str]:
        """Break a complex question into 2-4 simpler sub-queries.

        Args:
            question: The original question to decompose.

        Returns:
            List of sub-query strings (2-4 queries).
        """
        prompt = f"""You are a query decomposition expert. Break down the following question into 2-4 simpler sub-queries that can be used for parallel retrieval.

Guidelines:
- Each sub-query should focus on a specific aspect or entity from the original question
- Sub-queries should be self-contained and searchable
- Keep sub-queries concise (under 15 words each)
- If the question mentions multiple entities (cities, companies, topics), create separate sub-queries for each
- If the question asks about multiple aspects (trends, changes, comparisons), create separate sub-queries for each

Question: {question}

Return ONLY the sub-queries, one per line. No numbering, no explanations, no empty lines.
Example output:
Boston employment trends
Boston wage changes
New York employment trends
New York wage changes"""

        response = self.llm.invoke(prompt)
        content = response.content.strip()

        # Parse the response into individual sub-queries
        sub_queries = [q.strip() for q in content.split("\n") if q.strip()]

        # Ensure we have at least the original question if decomposition fails
        if not sub_queries:
            sub_queries = [question]

        # Limit to 4 sub-queries max
        sub_queries = sub_queries[:4]

        print(f"[DeepRAG] Decomposed into {len(sub_queries)} sub-queries:")
        for sq in sub_queries:
            print(f"  - {sq}")

        return sub_queries

    def _vector_search(
        self, query: str, top_k: int = 20
    ) -> list[tuple[int, float]]:
        """Perform vector search for a single query.

        Args:
            query: The search query string.
            top_k: Number of results to retrieve.

        Returns:
            List of (chunk_index, score) tuples sorted by score descending.
        """
        if self.index is None:
            raise RuntimeError("Index not built - call _build_index() first")

        # Embed the query
        q_embedding = np.array(
            [self.embeddings_model.embed_query(query)],
            dtype=np.float32,
        )
        faiss.normalize_L2(q_embedding)

        # Search
        scores, indices = self.index.search(q_embedding, top_k)

        # Build results list
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:  # Valid index (-1 means no result)
                results.append((int(idx), float(score)))

        return results

    def _rrf_fusion(
        self,
        results_per_query: list[list[tuple[int, float]]],
        sub_queries: list[str],
        k: int = 60,
    ) -> list[tuple[int, float, list[str]]]:
        """Fuse results using Reciprocal Rank Fusion.

        RRF formula: score = sum(1 / (k + rank_i)) for each query where chunk appears

        Args:
            results_per_query: List of [(chunk_index, score), ...] per query.
            sub_queries: List of sub-query strings (for tracking contributing queries).
            k: RRF constant (default 60, as recommended in the literature).

        Returns:
            Fused list of (chunk_index, rrf_score, contributing_queries) sorted by score descending.
        """
        # Track RRF scores and contributing queries per chunk
        rrf_scores: dict[int, float] = defaultdict(float)
        contributing_queries: dict[int, list[str]] = defaultdict(list)

        for query_idx, results in enumerate(results_per_query):
            query_str = sub_queries[query_idx] if query_idx < len(sub_queries) else ""
            for rank, (chunk_idx, _score) in enumerate(results):
                # RRF: score contribution = 1 / (k + rank + 1)
                # rank is 0-indexed, so we add 1 to make it 1-indexed
                rrf_contribution = 1.0 / (k + rank + 1)
                rrf_scores[chunk_idx] += rrf_contribution
                if query_str and query_str not in contributing_queries[chunk_idx]:
                    contributing_queries[chunk_idx].append(query_str)

        # Sort by RRF score descending
        fused_results = [
            (chunk_idx, score, contributing_queries[chunk_idx])
            for chunk_idx, score in rrf_scores.items()
        ]
        fused_results.sort(key=lambda x: x[1], reverse=True)

        return fused_results

    def _format_context(self, retrieved_chunks: list[dict]) -> str:
        """Format retrieved chunks into context string for LLM.

        Args:
            retrieved_chunks: List of chunk dicts with body, header_path, rrf_score.

        Returns:
            Formatted context string.
        """
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            header = chunk.get("header_path", "")
            body = chunk.get("body", "")
            context_parts.append(f"[{i}] {header}\n{body}")
        return "\n\n".join(context_parts)

    async def query(
        self,
        question: str,
        top_k: int = 15,
        per_query_k: int = 20,
    ) -> tuple[str, list[dict]]:
        """Query the RAG system with query decomposition and RRF fusion.

        Args:
            question: The question to answer.
            top_k: Number of fused chunks to use for synthesis.
            per_query_k: Number of chunks to retrieve per sub-query.

        Returns:
            Tuple of (answer, retrieved_chunks).
            Each retrieved chunk has: chunk_id, body, header_path, rrf_score, contributing_queries.
        """
        if self.index is None:
            raise RuntimeError("Index not built - call _build_index() first")

        # Step 1: Decompose the question into sub-queries
        sub_queries = await self._decompose_query(question)

        # Step 2: Vector search for each sub-query (in parallel using asyncio)
        # Note: FAISS operations are CPU-bound and synchronous, but we structure
        # this to allow for future async embedding APIs
        results_per_query = []
        for sq in sub_queries:
            results = self._vector_search(sq, top_k=per_query_k)
            results_per_query.append(results)

        # Step 3: RRF fusion across all sub-query results
        fused_results = self._rrf_fusion(results_per_query, sub_queries)

        # Take top_k fused results
        top_fused = fused_results[:top_k]

        # Step 4: Build retrieved chunks list with RRF metadata
        retrieved_chunks = []
        for chunk_idx, rrf_score, contrib_queries in top_fused:
            chunk = self.chunks[chunk_idx]
            retrieved_chunks.append({
                "chunk_id": chunk.get("chunk_id", f"chunk_{chunk_idx}"),
                "body": chunk.get("body", ""),
                "header_path": chunk.get("header_path", ""),
                "rrf_score": rrf_score,
                "contributing_queries": contrib_queries,
            })

        if not retrieved_chunks:
            return (
                "I cannot find information about this in the provided context.",
                [],
            )

        # Step 5: Format context for LLM
        formatted_context = self._format_context(retrieved_chunks)

        # Step 6: Synthesis prompt
        prompt = f"""You are a helpful assistant. Answer the question based ONLY on the provided context.
If the context doesn't contain relevant information, say "I cannot find information about this in the provided context."

Question: {question}

Context:
{formatted_context}

Answer:"""

        # Step 7: LLM synthesis
        response = self.llm.invoke(prompt)
        answer = response.content

        return (answer, retrieved_chunks)

    def query_sync(
        self,
        question: str,
        top_k: int = 15,
        per_query_k: int = 20,
    ) -> tuple[str, list[dict]]:
        """Synchronous version of query for non-async contexts.

        Args:
            question: The question to answer.
            top_k: Number of fused chunks to use for synthesis.
            per_query_k: Number of chunks to retrieve per sub-query.

        Returns:
            Tuple of (answer, retrieved_chunks).
        """
        return asyncio.get_event_loop().run_until_complete(
            self.query(question, top_k, per_query_k)
        )
