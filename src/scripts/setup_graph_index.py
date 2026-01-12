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
      `vector.dimensions`: 1024,
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
      `vector.dimensions`: 1024,
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
      `vector.dimensions`: 1024,
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
      `vector.dimensions`: 1024,
      `vector.similarity_function`: 'cosine'
    }}
    """
    try:
        client.query(cypher_fact)
        print("✅ Vector Index 'fact_embeddings' created/verified.")
    except Exception as e:
        print(f"❌ Failed to create fact index: {e}")

    # === FULL-TEXT INDEXES FOR DETERMINISTIC RETRIEVAL ===

    # Full-text index on FactNode.content for BM25/keyword search
    print("Checking/Creating Full-Text Index 'fact_fulltext'...")
    cypher_fact_fulltext = """
    CREATE FULLTEXT INDEX fact_fulltext IF NOT EXISTS
    FOR (n:FactNode)
    ON EACH [n.content]
    """
    try:
        client.query(cypher_fact_fulltext)
        print("✅ Full-Text Index 'fact_fulltext' created/verified.")
    except Exception as e:
        print(f"❌ Failed to create fact_fulltext index: {e}")

    # Full-text index on EpisodicNode.content for chunk text search
    print("Checking/Creating Full-Text Index 'chunk_fulltext'...")
    cypher_chunk_fulltext = """
    CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS
    FOR (n:EpisodicNode)
    ON EACH [n.content]
    """
    try:
        client.query(cypher_chunk_fulltext)
        print("✅ Full-Text Index 'chunk_fulltext' created/verified.")
    except Exception as e:
        print(f"❌ Failed to create chunk_fulltext index: {e}")

    # Full-text index on EntityNode for keyword entity search
    print("Checking/Creating Full-Text Index 'entity_fulltext'...")
    cypher_entity_fulltext = """
    CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS
    FOR (n:EntityNode)
    ON EACH [n.name, n.summary]
    """
    try:
        client.query(cypher_entity_fulltext)
        print("✅ Full-Text Index 'entity_fulltext' created/verified.")
    except Exception as e:
        print(f"❌ Failed to create entity_fulltext index: {e}")

    client.close()

if __name__ == "__main__":
    setup_index()
