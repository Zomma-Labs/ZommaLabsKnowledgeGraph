import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.util.services import get_services

def check_embeddings():
    print("Testing Embedding Storage in Neo4j...")
    
    services = get_services()
    neo4j = services.neo4j
    
    try:
        # Check EntityNodes
        cypher_entity = """
        MATCH (n:EntityNode)
        RETURN count(n) as total, count(n.embedding) as with_embedding
        """
        res_entity = neo4j.query(cypher_entity)
        total_ent = res_entity[0]['total']
        embed_ent = res_entity[0]['with_embedding']
        
        print(f"EntityNodes: {embed_ent}/{total_ent} have embeddings.")
        
        if total_ent > 0 and embed_ent == 0:
            print("❌ FAILURE: EntityNodes exist but passed NO embeddings.")
        elif total_ent > 0 and embed_ent < total_ent:
            print("⚠️ WARNING: Some EntityNodes are missing embeddings.")
        
        # Check TopicNodes
        cypher_topic = """
        MATCH (n:TopicNode)
        RETURN count(n) as total, count(n.embedding) as with_embedding
        """
        res_topic = neo4j.query(cypher_topic)
        total_top = res_topic[0]['total']
        embed_top = res_topic[0]['with_embedding']
        
        print(f"TopicNodes:  {embed_top}/{total_top} have embeddings.")
        
        if total_top > 0 and embed_top == 0:
            print("❌ FAILURE: TopicNodes exist but passed NO embeddings.")
        
        # Check one valid embedding dimension (sanity check)
        cypher_sample = """
        MATCH (n) WHERE n.embedding IS NOT NULL 
        RETURN size(n.embedding) as dim LIMIT 1
        """
        res_sample = neo4j.query(cypher_sample)
        if res_sample:
            print(f"✅ Sanity Check: Embedding dimension is {res_sample[0]['dim']}")
        else:
            print("⚠️ No embeddings found to check dimension.")

    finally:
        services.close()

if __name__ == "__main__":
    check_embeddings()
