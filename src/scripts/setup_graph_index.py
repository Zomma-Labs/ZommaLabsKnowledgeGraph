import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.tools.neo4j_client import Neo4jClient

def setup_index():
    client = Neo4jClient()
    
    # Check if index exists
    # Note: This query might vary based on Neo4j version. 
    # For Neo4j 5.x, we can use SHOW INDEXES.
    
    print("Checking/Creating Vector Index 'entity_embeddings'...")
    
    # Drop if exists (optional, but good for clean setup during dev)
    # client.query("DROP INDEX entity_embeddings IF EXISTS")
    
    # Create Vector Index
    # Dimension: 1024 (voyage-finance-2) or 1536 (openai). 
    # We need to know the dimension of voyage-finance-2. 
    # Voyage-finance-2 is 1024 dimensions.
    
    cypher = """
    CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
    FOR (n:Entity)
    ON (n.embedding)
    OPTIONS {indexConfig: {
      `vector.dimensions`: 1024,
      `vector.similarity_function`: 'cosine'
    }}
    """
    
    try:
        client.query(cypher)
        print("✅ Vector Index 'entity_embeddings' created/verified.")
    except Exception as e:
        print(f"❌ Failed to create index: {e}")

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
        print(f"❌ Failed to create index: {e}")

    print("Checking/Creating Vector Index 'section_embeddings'...")
    cypher_section = """
    CREATE VECTOR INDEX section_embeddings IF NOT EXISTS
    FOR (n:SectionNode)
    ON (n.embedding)
    OPTIONS {indexConfig: {
      `vector.dimensions`: 1024,
      `vector.similarity_function`: 'cosine'
    }}
    """
    try:
        client.query(cypher_section)
        print("✅ Vector Index 'section_embeddings' created/verified.")
    except Exception as e:
        print(f"❌ Failed to create index: {e}")
        
    client.close()

if __name__ == "__main__":
    setup_index()
