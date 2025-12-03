import unittest
from unittest.mock import MagicMock, patch
from src.workflows.main_pipeline import initialize_episode, GraphState
from src.agents.header_analyzer import Dimension, DimensionType

class TestHierarchyBridge(unittest.TestCase):

    @patch('src.workflows.main_pipeline.neo4j_client')
    @patch('src.workflows.main_pipeline.header_analyzer')
    @patch('src.workflows.main_pipeline.librarian')
    @patch('src.workflows.main_pipeline.embeddings')
    def test_initialize_episode_hierarchy(self, mock_embeddings, mock_librarian, mock_analyzer, mock_neo4j):
        # Setup Mocks
        mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        
        # Mock Analyzer: 
        # Path: ["Regional Reports", "New York", "Labor Markets"]
        # Dimensions: "New York" (ENTITY), "Labor Markets" (TOPIC)
        def analyze_path_side_effect(headers):
            return [
                Dimension(value="New York", type=DimensionType.ENTITY, original_header="New York"),
                Dimension(value="Labor Markets", type=DimensionType.TOPIC, original_header="Labor Markets")
            ]
        
        mock_analyzer.analyze_path.side_effect = analyze_path_side_effect
        
        # Mock Librarian:
        # "New York" -> No Match (Simulate Graph Search/Create)
        # "Labor Markets" -> Match (Score 0.95)
        def resolve_side_effect(text):
            if "Labor Markets" in text:
                return {"label": "Labor Market Indicator", "uri": "fibo:LaborMarket", "score": 0.95}
            return None
        
        mock_librarian.resolve.side_effect = resolve_side_effect
        
        # Mock Neo4j
        def neo4j_query_side_effect(cypher, params=None):
            print(f"DEBUG: Cypher: {cypher[:50]}...")
            print(f"DEBUG: Params: {params}")
            if "DocumentNode" in cypher:
                return [{"doc_uuid": "doc-123"}]
            if "SectionNode" in cypher:
                # Hub creation
                return [{"hub_uuid": f"hub-{params.get('header_path', 'UNKNOWN')}"}]
            if "TopicNode" in cypher or "EntityNode" in cypher:
                return [{"node_uuid": f"node-{params.get('name', 'UNKNOWN')}"}]
            return []
            
        mock_neo4j.query.side_effect = neo4j_query_side_effect

        # Input State
        state = {
            "chunk_text": "Some content",
            "metadata": {
                "doc_id": "TestDoc",
                "headings": ["Regional Reports", "New York", "Labor Markets"],
                "group_id": "test_group"
            }
        }

        # Run Function
        result = initialize_episode(state)

        # Assertions
        
        # 1. Check Analyzer Calls
        self.assertEqual(mock_analyzer.analyze_path.call_count, 1)
        
        # 2. Check Librarian Calls
        self.assertEqual(mock_librarian.resolve.call_count, 2)
        
        # 3. Check Neo4j Calls
        calls = mock_neo4j.query.call_args_list
        
        # Check for Hub Creation (SectionNode with header_path)
        hub_calls = [c for c in calls if "SectionNode" in c[0][0] and "header_path" in c[0][0]]
        self.assertTrue(len(hub_calls) >= 1)
        
        # Check for Entity Creation (New York)
        entity_calls = [c for c in calls if "EntityNode" in c[0][0] and "New York" in c[0][1].get('name', '')]
        self.assertTrue(len(entity_calls) >= 1)
        
        # Check for Topic Creation (Labor Markets -> FIBO)
        topic_calls = [c for c in calls if "TopicNode" in c[0][0] and "fibo_uri" in c[0][0]]
        self.assertTrue(len(topic_calls) >= 1)
        
        # Check for Links (DISCUSSES, REPRESENTS)
        link_calls = [c for c in calls if "DISCUSSES" in c[0][0] or "REPRESENTS" in c[0][0]]
        self.assertEqual(len(link_calls), 2) # One query per dimension (creates both links)

if __name__ == '__main__':
    unittest.main()
