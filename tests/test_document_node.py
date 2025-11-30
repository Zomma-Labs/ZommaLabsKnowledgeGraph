import unittest
import uuid
from src.tools.neo4j_client import Neo4jClient
from src.workflows.main_pipeline import initialize_episode

class TestDocumentNode(unittest.TestCase):
    def setUp(self):
        self.neo4j = Neo4jClient()
        self.group_id = f"test_group_{uuid.uuid4()}"

    def tearDown(self):
        self.neo4j.query("MATCH (n {group_id: $group_id}) DETACH DELETE n", {"group_id": self.group_id})
        self.neo4j.close()

    def test_document_node_creation(self):
        print(f"\nTesting DocumentNode creation for {self.group_id}")
        
        chunk_text = "This is a test chunk."
        metadata = {
            "filename": "test_doc.pdf",
            "file_type": "pdf",
            "source_id": "doc_123",
            "group_id": self.group_id
        }
        
        state = {
            "chunk_text": chunk_text,
            "metadata": metadata,
            "group_id": self.group_id
        }
        
        # Run initialization
        result = initialize_episode(state)
        episode_uuid = result["episodic_uuid"]
        
        # Verify DocumentNode
        doc_res = self.neo4j.query(
            "MATCH (d:DocumentNode {group_id: $group_id}) RETURN d",
            {"group_id": self.group_id}
        )
        self.assertEqual(len(doc_res), 1)
        doc = doc_res[0]['d']
        self.assertEqual(doc['name'], "test_doc.pdf")
        self.assertEqual(doc['file_type'], "pdf")
        # Check for document_date (might be returned as Neo4j DateTime object or string depending on driver)
        # We just check key existence for now
        self.assertTrue('document_date' in doc or hasattr(doc, 'document_date'))
        
        # Verify Link
        link_res = self.neo4j.query(
            """
            MATCH (e:EpisodicNode {uuid: $episode_uuid, group_id: $group_id})
            MATCH (d:DocumentNode {group_id: $group_id})
            MATCH (e)-[r:BELONGS_TO]->(d)
            RETURN r
            """,
            {"episode_uuid": episode_uuid, "group_id": self.group_id}
        )
        self.assertEqual(len(link_res), 1)
        print("✅ DocumentNode created and linked successfully")

if __name__ == '__main__':
    unittest.main()
