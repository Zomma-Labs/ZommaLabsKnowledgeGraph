"""
MODULE: Services
DESCRIPTION: Singleton container for shared infrastructure services.
             Provides centralized, lazy-initialized access to LLM, embeddings, and database clients.
"""

from typing import Optional
from src.util.llm_client import get_llm, get_embeddings
from src.tools.neo4j_client import Neo4jClient
from qdrant_client import QdrantClient

# Qdrant configuration (reused from FIBO_librarian)
QDRANT_FIBO_PATH = "./qdrant_fibo"
QDRANT_RELATIONSHIPS_PATH = "./qdrant_relationships"


class Services:
    """
    Singleton container for shared infrastructure services.
    Ensures all agents use the same clients instead of creating duplicates.
    """
    _instance: Optional["Services"] = None

    def __init__(self):
        # Lazy initialization flags
        self._llm = None
        self._embeddings = None
        self._neo4j = None
        self._qdrant_fibo = None
        self._qdrant_relationships = None

    @classmethod
    def get(cls) -> "Services":
        """Returns the singleton Services instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def llm(self):
        """Lazy-initialized LLM client."""
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    @property
    def embeddings(self):
        """Lazy-initialized embeddings client."""
        if self._embeddings is None:
            self._embeddings = get_embeddings()
        return self._embeddings

    @property
    def neo4j(self) -> Neo4jClient:
        """Lazy-initialized Neo4j client."""
        if self._neo4j is None:
            self._neo4j = Neo4jClient()
        return self._neo4j

    @property
    def qdrant_fibo(self) -> QdrantClient:
        """Lazy-initialized Qdrant client for FIBO entities."""
        if self._qdrant_fibo is None:
            self._qdrant_fibo = QdrantClient(path=QDRANT_FIBO_PATH)
        return self._qdrant_fibo

    @property
    def qdrant_relationships(self) -> QdrantClient:
        """Lazy-initialized Qdrant client for relationship definitions."""
        if self._qdrant_relationships is None:
            self._qdrant_relationships = QdrantClient(path=QDRANT_RELATIONSHIPS_PATH)
        return self._qdrant_relationships

    def close(self):
        """Closes all active connections."""
        if self._neo4j is not None:
            self._neo4j.close()
        # Qdrant clients don't need explicit closing for local storage


# Convenience function
def get_services() -> Services:
    """Helper function to get the singleton Services instance."""
    return Services.get()
