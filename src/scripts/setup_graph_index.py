import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.util.neo4j_client import Neo4jClient

def setup_index():
    client = Neo4jClient()
    
    # Check if index exists
    # Note: This query might vary based on Neo4j version. 
    # For Neo4j 5.x, we can use SHOW INDEXES.
    
    # Entity name_embedding index (used by entity resolution and MCP queries)
    print("Checking/Creating Vector Index 'entity_name_embeddings'...")
    cypher_entity_name = """
    CREATE VECTOR INDEX entity_name_embeddings IF NOT EXISTS
    FOR (n:EntityNode)
    ON (n.name_embedding)
    OPTIONS {indexConfig: {
      `vector.dimensions`: 3072,
      `vector.similarity_function`: 'cosine'
    }}
    """
    try:
        client.query(cypher_entity_name)
        print("✅ Vector Index 'entity_name_embeddings' created/verified.")
    except Exception as e:
        print(f"❌ Failed to create entity_name_embeddings index: {e}")

    # Entity name_only_embedding index (for direct name lookup without context)
    print("Checking/Creating Vector Index 'entity_name_only_embeddings'...")
    cypher_entity_name_only = """
    CREATE VECTOR INDEX entity_name_only_embeddings IF NOT EXISTS
    FOR (n:EntityNode)
    ON (n.name_only_embedding)
    OPTIONS {indexConfig: {
      `vector.dimensions`: 3072,
      `vector.similarity_function`: 'cosine'
    }}
    """
    try:
        client.query(cypher_entity_name_only)
        print("✅ Vector Index 'entity_name_only_embeddings' created/verified.")
    except Exception as e:
        print(f"❌ Failed to create entity_name_only_embeddings index: {e}")

    print("Checking/Creating Vector Index 'topic_embeddings'...")
    cypher_topic = """
    CREATE VECTOR INDEX topic_embeddings IF NOT EXISTS
    FOR (n:TopicNode)
    ON (n.embedding)
    OPTIONS {indexConfig: {
      `vector.dimensions`: 3072,
      `vector.similarity_function`: 'cosine'
    }}
    """
    try:
        client.query(cypher_topic)
        print("✅ Vector Index 'topic_embeddings' created/verified.")
    except Exception as e:
        print(f"❌ Failed to create topic index: {e}")

    # Fact embedding index (for relationship/fact search)
    print("Checking/Creating Vector Index 'fact_embeddings'...")
    cypher_fact = """
    CREATE VECTOR INDEX fact_embeddings IF NOT EXISTS
    FOR (n:FactNode)
    ON (n.embedding)
    OPTIONS {indexConfig: {
      `vector.dimensions`: 3072,
      `vector.similarity_function`: 'cosine'
    }}
    """
    try:
        client.query(cypher_fact)
        print("✅ Vector Index 'fact_embeddings' created/verified.")
    except Exception as e:
        print(f"❌ Failed to create fact index: {e}")

    client.close()

if __name__ == "__main__":
    setup_index()
