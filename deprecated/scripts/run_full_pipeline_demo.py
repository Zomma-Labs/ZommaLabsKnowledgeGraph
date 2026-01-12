import sys
import os
import asyncio
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.workflows.main_pipeline import app
from src.tools.neo4j_client import Neo4jClient

async def run_demo():
    print("--- 🚀 Running Full Pipeline Demo ---")
    
    # Complex Input Text
    chunk_text = """
    Current economic conditions have deteriorated significantly. 
    Inflation rose by 3% last quarter, causing the Federal Reserve to raise rates.
    """
    metadata = {
        "doc_date": "2023-11-01", 
        "chunk_id": "demo_causal", 
        "section_header": "Incident Report",
        "headings": ["2. Incident Report", "2.1 System Failure"], # Added headings
        "group_id": "demo_tenant"
    }
    
    print(f"\n📄 Input Text:\n{chunk_text}\n")
    
    # Run Pipeline
    inputs = {
        "chunk_text": chunk_text,
        "metadata": metadata
    }
    
    try:
        result = await app.ainvoke(inputs)
        print("\n✅ Pipeline Finished.")
        
        # Inspect Neo4j
        print("\n--- 🔍 Inspecting Graph (Neo4j) ---")
        client = Neo4jClient()
        


        # 1. Query Entity-Chunk Relationships (The New V2 Way)
        cypher_facts = """
        MATCH (s:EntityNode)-[r1]->(chunk:EpisodicNode)
        OPTIONAL MATCH (chunk)-[r2]->(o:EntityNode)
        WHERE r1.fact_id = r2.fact_id OR r2 is NULL
        RETURN s.name as Subject, type(r1) as Action, chunk.content as Evidence, type(r2) as PassiveAction, o.name as Object
        """
        facts = client.query(cypher_facts)
        print(f"\n📊 Found {len(facts)} Entity-Chunk-Entity Paths:")
        for record in facts:
            subj = record['Subject']
            action = record['Action']
            obj = record['Object'] or "NONE"
            passive = record['PassiveAction'] or "NONE"
            print(f"   - [{subj}] --{action}--> (CHUNK) --{passive}--> [{obj}]")
            
        # 1.5 Query Topic Links
        cypher_topics = """
        MATCH (t:TopicNode)-[:ABOUT]->(chunk:EpisodicNode)
        RETURN t.name as Topic, chunk.content as Chunk
        """
        topics = client.query(cypher_topics)
        print(f"\n📊 Found {len(topics)} Topic-Chunk Links:")
        for record in topics:
            print(f"   - (Topic: {record['Topic']}) --ABOUT--> (CHUNK)")
            
        # 2. Query Causal Links
        cypher_causal = """
        MATCH (f1:FactNode)-[r:CAUSES]->(f2:FactNode)
        RETURN f1.content as Cause, r.reasoning as Reasoning, f2.content as Effect
        """
        links = client.query(cypher_causal)
        print(f"\n🔗 Created {len(links)} Causal Links:")
        for record in links:
            print(f"   - CAUSE: {record['Cause']}")
            print(f"     EFFECT: {record['Effect']}")
            print(f"     REASON: {record['Reasoning']}\n")
            
        client.close()
            
    except Exception as e:
        print(f"\n❌ Demo Failed: {e}")

if __name__ == "__main__":
    asyncio.run(run_demo())
