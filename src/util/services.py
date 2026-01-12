"""
MODULE: Services
DESCRIPTION: Singleton container for shared infrastructure services.
             Provides centralized, lazy-initialized access to LLM, embeddings, and database clients.
"""

from typing import Optional
from src.util.llm_client import get_llm, get_claude_llm, get_embeddings, get_dedup_embeddings
from src.util.neo4j_client import Neo4jClient


class Services:
    """
    Singleton container for shared infrastructure services.
    Ensures all agents use the same clients instead of creating duplicates.
    """
    _instance: Optional["Services"] = None

    def __init__(self):
        # Lazy initialization flags
        self._llm = None
        self._claude_llm = None
        self._embeddings = None
        self._dedup_embeddings = None
        self._neo4j = None

    @classmethod
    def get(cls) -> "Services":
        """Returns the singleton Services instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def llm(self):
        """Lazy-initialized LLM client (Gemini by default)."""
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    @property
    def claude_llm(self):
        """Lazy-initialized Claude Sonnet LLM (for entity deduplication)."""
        if self._claude_llm is None:
            self._claude_llm = get_claude_llm()
        return self._claude_llm

    @property
    def embeddings(self):
        """Lazy-initialized embeddings client (voyage-finance-2)."""
        if self._embeddings is None:
            self._embeddings = get_embeddings()
        return self._embeddings

    @property
    def dedup_embeddings(self):
        """Lazy-initialized embeddings for entity deduplication (voyage-3-large)."""
        if self._dedup_embeddings is None:
            self._dedup_embeddings = get_dedup_embeddings()
        return self._dedup_embeddings

    @property
    def neo4j(self) -> Neo4jClient:
        """Lazy-initialized Neo4j client."""
        if self._neo4j is None:
            self._neo4j = Neo4jClient()
        return self._neo4j

    def close(self):
        """Closes all active connections."""
        if self._neo4j is not None:
            self._neo4j.close()


# Convenience function
def get_services() -> Services:
    """Helper function to get the singleton Services instance."""
    return Services.get()
