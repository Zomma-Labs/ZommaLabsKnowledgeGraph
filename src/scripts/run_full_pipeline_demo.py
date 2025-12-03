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
    chunk_text = (
        "The software update caused the system to crash. "
        "Consequently, the company's stock price dropped by 5%. "
        "The outage was effected by a configuration error."
    )
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
        
        # 0. Query Dimensional Star (Hubs)
        print("\n🏗️ Dimensional Star Topology:")
        cypher_star = """
        MATCH (hub:SectionNode)
        OPTIONAL MATCH (hub)-[:DISCUSSES]->(t:TopicNode)
        OPTIONAL MATCH (hub)-[:REPRESENTS]->(e:EntityNode)
        RETURN hub.header_path as Hub, collect(distinct t.name) as Topics, collect(distinct e.name) as Entities
        """
        stars = client.query(cypher_star)
        for record in stars:
            print(f"   ⭐ Hub: [{record['Hub']}]")
            if record['Topics']:
                print(f"      - Topics: {record['Topics']}")
            if record['Entities']:
                print(f"      - Entities: {record['Entities']}")

        # 1. Query Facts (Broad)
        cypher_facts = """
        MATCH (f:FactNode)
        OPTIONAL MATCH (s)-[:PERFORMED]->(f)
        OPTIONAL MATCH (f)-[:TARGET]->(o)
        RETURN s.name as Subject, f.content as Fact, f.fact_type as Type, o.name as Object, labels(s) as S_Labels, labels(o) as O_Labels
        """
        facts = client.query(cypher_facts)
        print(f"\n📊 Found {len(facts)} Fact Nodes:")
        for record in facts:
            subj = record['Subject'] or "UNKNOWN"
            obj = record['Object'] or "NONE"
            print(f"   - [{subj}] --PERFORMED--> ({record['Fact']}) [Type: {record['Type']}] --TARGET--> [{obj}]")
            print(f"     Labels: S={record['S_Labels']}, O={record['O_Labels']}")
            
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
