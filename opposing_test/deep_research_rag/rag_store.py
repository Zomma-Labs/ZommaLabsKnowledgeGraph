"""
Simple RAG vector store for chunk retrieval.
"""

import os
import json
import numpy as np
from typing import List
from langchain_voyageai import VoyageAIEmbeddings


class RAGStore:
    """Simple vector store over chunks."""

    _instance = None

    def __new__(cls, chunk_file: str = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, chunk_file: str = None):
        if self._initialized:
            return

        if chunk_file is None:
            chunk_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "src/chunker/SAVED/beigebook_20251015.jsonl"
            )

        print("RAGStore: Loading chunks...")
        self.chunks = self._load_chunks(chunk_file)
        print(f"RAGStore: Loaded {len(self.chunks)} chunks")

        print("RAGStore: Embedding chunks...")
        self.embeddings = VoyageAIEmbeddings(model="voyage-finance-2")
        texts = [c["text"] for c in self.chunks]
        self.chunk_embeddings = np.array(self.embeddings.embed_documents(texts))
        print(f"RAGStore: Ready")

        self._initialized = True

    def _load_chunks(self, filepath: str) -> List[dict]:
        """Load chunks from JSONL."""
        chunks = []
        with open(filepath, 'r') as f:
            for line in f:
                data = json.loads(line)
                text = data.get('body') or data.get('chunk_text') or data.get('text', '')
                header = data.get('header_path', '')

                if text.strip():
                    # Prepend header for attribution
                    full_text = f"[{header}]\n{text}" if header else text
                    chunks.append({
                        "text": full_text,
                        "header": header,
                        "uuid": data.get('uuid', '')
                    })
        return chunks

    def search(self, query: str, k: int = 10, query_embedding: np.ndarray = None) -> List[dict]:
        """Search for relevant chunks."""
        if query_embedding is None:
            query_embedding = np.array(self.embeddings.embed_query(query))

        # Cosine similarity
        similarities = np.dot(self.chunk_embeddings, query_embedding) / (
            np.linalg.norm(self.chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        top_indices = np.argsort(similarities)[-k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                "text": self.chunks[idx]["text"],
                "header": self.chunks[idx]["header"],
                "score": float(similarities[idx])
            })
        return results
