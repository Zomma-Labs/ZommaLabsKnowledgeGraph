import sys
import os
import asyncio
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.workflows.main_pipeline import app
from src.tools.neo4j_client import Neo4jClient
from src.scripts.setup_graph_index import setup_index

async def run_large_test():
    print("--- 🚀 Running Large Scale Pipeline Test ---")
    
    # Ensure Vector Indices Exist
    print("\n⚙️ Setting up Vector Indices...")
    setup_index()
    print("✅ Indices Ready.\n")
    
    # Complex Financial Text
    chunk_text = (
        "On Wednesday, the Federal Reserve raised its benchmark interest rate by 0.25%, signaling a continued effort to combat inflation. "
        "Chairman Jerome Powell stated that while price pressures have eased, the labor market remains tight. "
        "This announcement caused the S&P 500 to drop by 1.5% as investors feared a potential recession. "
        "Meanwhile, Apple Inc. reported Q4 earnings of $89.5 billion, beating analyst estimates. "
        "The strong performance was driven by robust sales of the iPhone 15 in China. "
        "However, the tech giant warned that supply chain disruptions could impact holiday quarter revenue. "
        "Consequently, Apple's stock slid 2% in after-hours trading."
    )
    metadata = {"doc_date": "2023-11-02", "chunk_id": "large_test_1", "section_header": "Market Wrap"}
    
    print(f"\n📄 Input Text:\n{chunk_text}\n")
    
    # Run Pipeline
    inputs = {
        "chunk_text": chunk_text,
        "metadata": metadata
    }
    
    try:
        result = await app.ainvoke(inputs)
        print("\n✅ Pipeline Finished.")
        
        # Debug: Print atomized facts
        atomized_facts = result.get("atomic_facts", [])
        print(f"\n🧩 Atomized Facts: {len(atomized_facts)}")
        for i, f in enumerate(atomized_facts):
            print(f"   {i}. {f.fact}")

        # Debug: Print raw links
        raw_links = result.get("causal_links", [])
        print(f"\n🐛 Raw Causal Links from Pipeline: {len(raw_links)}")
        for l in raw_links:
            print(f"   - {l.cause_index} -> {l.effect_index}: {l.reasoning}")
        
        # Inspect Neo4j
        print("\n--- 🔍 Inspecting Graph (Neo4j) ---")
        client = Neo4jClient()
        
        # 1. Query Fact Nodes (No Time Filter)
        cypher_facts = """
        MATCH (f:FactNode)
        OPTIONAL MATCH (s)-[:PERFORMED]->(f)
        OPTIONAL MATCH (f)-[:TARGET]->(o)
        RETURN s.name as Subject, f.content as Fact, f.fact_type as Type, o.name as Object
        """
        facts = client.query(cypher_facts)
        print(f"\n📊 Generated Facts ({len(facts)}):")
        for record in facts:
            print(f"   - [{record['Subject']}] --({record['Type']})--> [{record['Object']}]")
            # print(f"     Fact: {record['Fact']}") # Comment out to save space
            
        # 2. Query Causal Links
        cypher_causal = """
        MATCH (f1:FactNode)-[r:CAUSES]->(f2:FactNode)
        WHERE r.created_at > datetime() - duration('PT5M')
        RETURN f1.content as Cause, r.reasoning as Reasoning, f2.content as Effect
        """
        links = client.query(cypher_causal)
        print(f"\n🔗 Causal Links ({len(links)}):")
        for record in links:
            print(f"   - CAUSE: {record['Cause']}")
            print(f"     EFFECT: {record['Effect']}")
            print(f"     REASON: {record['Reasoning']}\n")
            
        # 3. Count Episodic Nodes
        count = client.query("MATCH (e:EpisodicNode) RETURN count(e) as count")[0]['count']
        print(f"\n📚 Total EpisodicNodes in DB: {count}")
            
        client.close()
            
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")

    # Write debug output to file
    with open("debug_output.txt", "w") as f:
        f.write(f"Atomized Facts: {len(result.get('atomic_facts', []))}\n")
        for i, fact in enumerate(result.get('atomic_facts', [])):
            f.write(f"{i}. {fact.fact}\n")
        
        f.write(f"\nRaw Causal Links: {len(result.get('causal_links', []))}\n")
        
        f.write(f"\nPipeline Errors: {result.get('errors', [])}\n")
        
        f.write("\nResolved Entities:\n")
        for i, res in enumerate(result.get('resolved_entities', [])):
            f.write(f"{i}. S: {res.get('subject_uri')} ({res.get('subject_label')}) -> O: {res.get('object_uri')} ({res.get('object_label')})\n")

        f.write("\nNeo4j Facts:\n")
        client = Neo4jClient()
        facts = client.query("MATCH (f:FactNode) RETURN f.content as Fact, f.fact_type as Type")
        for record in facts:
            f.write(f"[{record['Type']}] {record['Fact']}\n")
        client.close()

if __name__ == "__main__":
    asyncio.run(run_large_test())
