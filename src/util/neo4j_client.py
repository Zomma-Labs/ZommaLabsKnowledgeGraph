import os
import random
import time
import warnings
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError
from dotenv import load_dotenv

load_dotenv()

# Suppress Neo4j "property does not exist" warnings (harmless on empty DBs)
warnings.filterwarnings("ignore", message=".*property.*does not exist.*")

class Neo4jClient:
    def __init__(self):
        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")

        if not all([uri, username, password]):
            raise ValueError("Missing Neo4j credentials in .env")

        self._uri = uri
        self._auth = (username, password)
        self._max_retries = int(os.getenv("NEO4J_MAX_RETRIES", "5"))
        self._retry_base_seconds = float(os.getenv("NEO4J_RETRY_BASE_SECONDS", "2.0"))
        self._retry_max_seconds = float(os.getenv("NEO4J_RETRY_MAX_SECONDS", "30"))

        self.driver = GraphDatabase.driver(self._uri, auth=self._auth)

    def close(self):
        self.driver.close()

    def _reconnect(self):
        """Force a new driver/connection pool."""
        try:
            self.driver.close()
        finally:
            self.driver = GraphDatabase.driver(self._uri, auth=self._auth)

    def query(self, cypher: str, params: dict = None):
        """Executes a Cypher query and returns the result."""
        params = params or {}
        max_retries = max(self._max_retries, 0)

        for attempt in range(max_retries + 1):
            try:
                with self.driver.session() as session:
                    result = session.run(cypher, params)
                    return [record.data() for record in result]
            except (SessionExpired, ServiceUnavailable, TransientError) as e:
                if attempt >= max_retries:
                    raise
                # Reconnect on dropped/defunct connections.
                if isinstance(e, (SessionExpired, ServiceUnavailable)):
                    self._reconnect()
                sleep_for = min(self._retry_base_seconds * (2 ** attempt), self._retry_max_seconds)
                # Add jitter to avoid thundering herd retries.
                sleep_for *= 0.5 + (random.random() * 0.5)
                time.sleep(sleep_for)

    def verify_connectivity(self):
        try:
            self.driver.verify_connectivity()
            print("✅ Connected to Neo4j")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            return False

    def warmup(self, max_attempts: int = 10, wait_seconds: float = 5.0) -> bool:
        """
        Wake up Neo4j Aura instance and ensure connection is ready.
        Aura free tier can take 30-60s to wake from sleep.
        """
        for attempt in range(max_attempts):
            try:
                self._reconnect()
                # Simple query to verify connection is truly alive
                self.query("RETURN 1 AS ping")
                return True
            except (ServiceUnavailable, SessionExpired, TransientError) as e:
                if attempt < max_attempts - 1:
                    print(f"    Neo4j warmup attempt {attempt + 1}/{max_attempts} failed, waiting {wait_seconds}s...")
                    time.sleep(wait_seconds)
                else:
                    print(f"    Neo4j warmup failed after {max_attempts} attempts: {e}")
                    raise
        return False

    def vector_search(self, index_name: str, query_vector: list, top_k: int = 5, filters: dict = None) -> list:
        """
        Performs a vector search on the specified index with optional property filtering.
        Returns a list of records with 'node' and 'score'.
        """
        # Build WHERE clause dynamically if filters are provided
        where_clause = ""
        if filters:
            conditions = []
            for key in filters:
                conditions.append(f"node.{key} = ${key}")
            where_clause = "WHERE " + " AND ".join(conditions)

        cypher = f"""
        CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
        YIELD node, score
        {where_clause}
        RETURN node, score
        """
        params = {
            "index_name": index_name,
            "query_vector": query_vector,
            "top_k": top_k
        }
        if filters:
            params.update(filters)
            
        return self.query(cypher, params)
