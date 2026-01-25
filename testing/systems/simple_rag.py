"""
MODULE: SimpleRAG
DESCRIPTION: Pure vector similarity RAG system for baseline evaluation.
             Uses FAISS for vector search and GPT-5.1 for synthesis.
"""

import numpy as np
import faiss
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from testing.common.utils import load_chunks, EmbeddingCache, EMBEDDING_DIMENSION

load_dotenv()


class SimpleRAG:
    """Simple RAG system using pure vector similarity search.

    This is a baseline RAG implementation that:
    - Embeds all chunks with OpenAI text-embedding-3-large
    - Builds a FAISS index for fast similarity search
    - Retrieves top-k chunks for each query
    - Synthesizes an answer using GPT-5.1

    No RRF, no header boosting, no complexity - just pure vector RAG.
    """

    def __init__(
        self,
        chunk_file: str,
        cache_file: str | None = "eval/simple_rag_embeddings.pkl",
    ):
        """Initialize SimpleRAG with chunks from a JSONL file.

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
        print(f"[SimpleRAG] Loaded {len(self.chunks)} chunks from {self.chunk_file}")

    def _build_index(self) -> None:
        """Embed all chunks and build FAISS index."""
        if not self.chunks:
            raise ValueError("No chunks loaded - cannot build index")

        # Get chunk texts for embedding
        texts = [chunk["body"] for chunk in self.chunks]

        # Get or compute embeddings with caching
        print(f"[SimpleRAG] Computing embeddings for {len(texts)} chunks...")
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

        print(f"[SimpleRAG] Built FAISS index with {self.index.ntotal} vectors")

    def _format_context(self, retrieved_chunks: list[dict]) -> str:
        """Format retrieved chunks into context string for LLM.

        Args:
            retrieved_chunks: List of chunk dicts with body, header_path, score.

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
    ) -> tuple[str, list[dict]]:
        """Query the RAG system.

        Args:
            question: The question to answer.
            top_k: Number of chunks to retrieve.

        Returns:
            Tuple of (answer, retrieved_chunks).
            Each retrieved chunk has: chunk_id, body, header_path, score.
        """
        if self.index is None:
            raise RuntimeError("Index not built - call _build_index() first")

        # 1. Embed the question
        q_embedding = np.array(
            [self.embeddings_model.embed_query(question)],
            dtype=np.float32,
        )
        faiss.normalize_L2(q_embedding)

        # 2. FAISS similarity search
        scores, indices = self.index.search(q_embedding, top_k)

        # 3. Build retrieved chunks list
        retrieved_chunks = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:  # Valid index (-1 means no result)
                chunk = self.chunks[idx]
                retrieved_chunks.append({
                    "chunk_id": chunk.get("chunk_id", f"chunk_{idx}"),
                    "body": chunk.get("body", ""),
                    "header_path": chunk.get("header_path", ""),
                    "score": float(score),
                })

        if not retrieved_chunks:
            return (
                "I cannot find information about this in the provided context.",
                [],
            )

        # 4. Format context for LLM
        formatted_context = self._format_context(retrieved_chunks)

        # 5. Synthesis prompt
        prompt = f"""You are a helpful assistant. Answer the question based ONLY on the provided context.
If the context doesn't contain relevant information, say "I cannot find information about this in the provided context."

Question: {question}

Context:
{formatted_context}

Answer:"""

        # 6. LLM synthesis (sync call, but we're in async context)
        response = self.llm.invoke(prompt)
        answer = response.content

        return (answer, retrieved_chunks)

    def query_sync(
        self,
        question: str,
        top_k: int = 15,
    ) -> tuple[str, list[dict]]:
        """Synchronous version of query for non-async contexts.

        Args:
            question: The question to answer.
            top_k: Number of chunks to retrieve.

        Returns:
            Tuple of (answer, retrieved_chunks).
        """
        import asyncio

        return asyncio.get_event_loop().run_until_complete(
            self.query(question, top_k)
        )
