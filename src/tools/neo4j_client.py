import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

class Neo4jClient:
    def __init__(self):
        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        
        if not all([uri, username, password]):
            raise ValueError("Missing Neo4j credentials in .env")
            
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        self.driver.close()

    def query(self, cypher: str, params: dict = None):
        """Executes a Cypher query and returns the result."""
        with self.driver.session() as session:
            result = session.run(cypher, params or {})
            return [record.data() for record in result]

    def verify_connectivity(self):
        try:
            self.driver.verify_connectivity()
            print("✅ Connected to Neo4j")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
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
