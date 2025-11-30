import unittest
import uuid
from src.tools.neo4j_client import Neo4jClient
from src.agents.graph_assembler import GraphAssembler
from src.schemas.atomic_fact import AtomicFact

class TestNamespacing(unittest.TestCase):
    def setUp(self):
        self.neo4j = Neo4jClient()
        self.assembler = GraphAssembler()
        self.tenant_a = f"test_tenant_A_{uuid.uuid4()}"
        self.tenant_b = f"test_tenant_B_{uuid.uuid4()}"

    def tearDown(self):
        # Clean up
        self.neo4j.query("MATCH (n {group_id: $group_id}) DETACH DELETE n", {"group_id": self.tenant_a})
        self.neo4j.query("MATCH (n {group_id: $group_id}) DETACH DELETE n", {"group_id": self.tenant_b})
        self.assembler.close()
        self.neo4j.close()

    def test_data_isolation(self):
        print(f"\nTesting isolation between {self.tenant_a} and {self.tenant_b}")
        
        # 1. Create Fact for Tenant A
        fact_a = AtomicFact(subject="Apple", fact="released", object="iPhone 15")
        episode_uuid_a = str(uuid.uuid4())
        
        # Create Episode A
        self.neo4j.query(
            "MERGE (e:EpisodicNode {uuid: $uuid, group_id: $group_id})", 
            {"uuid": episode_uuid_a, "group_id": self.tenant_a}
        )
        
        uuid_a = self.assembler.assemble_fact_node(
            fact_obj=fact_a,
            subject_uri="urn:test:apple",
            subject_label="Apple",
            object_uri="urn:test:iphone15",
            object_label="iPhone 15",
            episode_uuid=episode_uuid_a,
            group_id=self.tenant_a
        )
        print(f"Created Fact A: {uuid_a}")

        # 2. Create Fact for Tenant B (Same content)
        fact_b = AtomicFact(subject="Apple", fact="released", object="iPhone 15")
        episode_uuid_b = str(uuid.uuid4())
        
        # Create Episode B
        self.neo4j.query(
            "MERGE (e:EpisodicNode {uuid: $uuid, group_id: $group_id})", 
            {"uuid": episode_uuid_b, "group_id": self.tenant_b}
        )
        
        uuid_b = self.assembler.assemble_fact_node(
            fact_obj=fact_b,
            subject_uri="urn:test:apple",
            subject_label="Apple",
            object_uri="urn:test:iphone15",
            object_label="iPhone 15",
            episode_uuid=episode_uuid_b,
            group_id=self.tenant_b
        )
        print(f"Created Fact B: {uuid_b}")

        # 3. Verify Isolation
        # Query for Tenant A should only return Fact A
        results_a = self.neo4j.query(
            "MATCH (f:FactNode {group_id: $group_id}) RETURN f.uuid as uuid", 
            {"group_id": self.tenant_a}
        )
        uuids_a = [r['uuid'] for r in results_a]
        self.assertIn(uuid_a, uuids_a)
        self.assertNotIn(uuid_b, uuids_a)
        
        # Query for Tenant B should only return Fact B
        results_b = self.neo4j.query(
            "MATCH (f:FactNode {group_id: $group_id}) RETURN f.uuid as uuid", 
            {"group_id": self.tenant_b}
        )
        uuids_b = [r['uuid'] for r in results_b]
        self.assertIn(uuid_b, uuids_b)
        self.assertNotIn(uuid_a, uuids_b)
        
        print("✅ Data Isolation Verified")

    def test_vector_search_isolation(self):
        print(f"\nTesting vector search isolation")
        
        # 1. Create Fact for Tenant A
        fact_a = AtomicFact(subject="Tesla", fact="released", object="Cybertruck")
        episode_uuid_a = str(uuid.uuid4())
        self.neo4j.query(
            "MERGE (e:EpisodicNode {uuid: $uuid, group_id: $group_id})", 
            {"uuid": episode_uuid_a, "group_id": self.tenant_a}
        )
        self.assembler.assemble_fact_node(
            fact_obj=fact_a,
            subject_uri="urn:test:tesla",
            subject_label="Tesla",
            object_uri="urn:test:cybertruck",
            object_label="Cybertruck",
            episode_uuid=episode_uuid_a,
            group_id=self.tenant_a
        )
        
        # 2. Vector Search as Tenant B (should find nothing)
        # We need to generate embedding for "Tesla released Cybertruck"
        # Since we can't easily access the embedding here without recreating it, 
        # we'll rely on the assembler's internal embedding generation or just use a dummy vector if we could,
        # but vector_search requires a vector.
        # Let's use the assembler's embedding client.
        
        vector = self.assembler.embeddings.embed_query("Tesla released Cybertruck")
        
        results = self.neo4j.vector_search(
            "fact_embeddings", 
            vector, 
            top_k=5, 
            filters={"group_id": self.tenant_b}
        )
        
        print(f"Vector Search Results for Tenant B: {len(results)}")
        self.assertEqual(len(results), 0)
        
        # 3. Vector Search as Tenant A (should find it)
        results_a = self.neo4j.vector_search(
            "fact_embeddings", 
            vector, 
            top_k=5, 
            filters={"group_id": self.tenant_a}
        )
        print(f"Vector Search Results for Tenant A: {len(results_a)}")
        self.assertGreater(len(results_a), 0)
        
        print("✅ Vector Search Isolation Verified")

if __name__ == '__main__':
    unittest.main()
