import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from src.tools.neo4j_client import Neo4jClient
from src.util.llm_client import get_llm, get_embeddings

class ExtractionResult(BaseModel):
    entities: List[str] = Field(description="List of entity names extracted from the query.")
    topics: List[str] = Field(description="List of topics extracted from the query.")

class CandidateSelection(BaseModel):
    selected_uuids: List[str] = Field(description="List of UUIDs for the candidates that match the user's intent.")
    reasoning: str = Field(description="Reasoning for the selection.")

class IntersectionQueryAgent:
    def __init__(self):
        self.neo4j = Neo4jClient()
        self.llm = get_llm()
        self.embeddings = get_embeddings()
        self.entity_index = "entity_embeddings"
        self.topic_index = "topic_embeddings"

    def ensure_indexes(self):
        """
        Ensures that vector indexes exist for EntityNode and TopicNode.
        """
        print("   🔍 Checking Vector Indexes...")
        
        # Entity Index
        try:
            self.neo4j.query(f"""
            CREATE VECTOR INDEX {self.entity_index} IF NOT EXISTS
            FOR (n:EntityNode)
            ON (n.embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: 1024,
                `vector.similarity_function`: 'cosine'
            }}}}
            """)
            print(f"   ✅ Index '{self.entity_index}' ready.")
        except Exception as e:
            print(f"   ⚠️ Error creating entity index: {e}")

        # Topic Index
        try:
            self.neo4j.query(f"""
            CREATE VECTOR INDEX {self.topic_index} IF NOT EXISTS
            FOR (n:TopicNode)
            ON (n.embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: 1024,
                `vector.similarity_function`: 'cosine'
            }}}}
            """)
            print(f"   ✅ Index '{self.topic_index}' ready.")
        except Exception as e:
            print(f"   ⚠️ Error creating topic index: {e}")

    def extract_query_components(self, query: str) -> ExtractionResult:
        """
        Uses LLM to extract entities and topics from the user query.
        """
        print(f"   🧠 Extracting components from: '{query}'")
        prompt = (
            "Extract the key Entities (specific organizations, places, people) "
            "and Topics (concepts, economic indicators, themes) from the user's question.\n"
            f"Question: {query}"
        )
        structured_llm = self.llm.with_structured_output(ExtractionResult)
        return structured_llm.invoke(prompt)

    def resolve_components(self, extraction: ExtractionResult, group_id: str) -> Dict[str, List[str]]:
        """
        Resolves extracted names to UUIDs using Vector Search + LLM Selection.
        """
        entity_uuids = []
        topic_uuids = []
        
        print("   🔎 Resolving Components...")

        def resolve_item(item: str, index_name: str, item_type: str) -> List[str]:
            print(f"     - Resolving {item_type}: '{item}'")
            try:
                vec = self.embeddings.embed_query(item)
                # Relaxed threshold: top_k=5, no strict score cutoff here, let LLM decide
                results = self.neo4j.vector_search(
                    index_name, 
                    vec, 
                    top_k=5, 
                    filters={"group_id": group_id}
                )
                
                if not results:
                    print(f"       ❌ No candidates found in vector search.")
                    return []
                
                # Prepare candidates for LLM
                candidates_str = ""
                candidate_map = {}
                for i, res in enumerate(results):
                    node = res['node']
                    score = res['score']
                    candidates_str += f"{i+1}. Name: {node.get('name')} (Score: {score:.2f}, UUID: {node.get('uuid')})\n"
                    candidate_map[node.get('uuid')] = node.get('name')
                
                print(f"       🤔 LLM Selecting from {len(results)} candidates...")
                
                selection_prompt = (
                    f"User Query Term: '{item}'\n"
                    f"Candidates found in Graph:\n{candidates_str}\n"
                    "Select ALL candidates that represent the user's query term. "
                    "If multiple candidates are relevant (e.g., synonyms or related concepts), select all of them. "
                    "If none are relevant, return an empty list.\n"
                )
                
                selector = self.llm.with_structured_output(CandidateSelection)
                decision = selector.invoke(selection_prompt)
                
                selected = []
                for uuid in decision.selected_uuids:
                    # Verify UUID exists in our candidates (security/hallucination check)
                    if uuid in candidate_map:
                        selected.append(uuid)
                        print(f"       ✅ Selected: {candidate_map[uuid]} ({uuid})")
                    else:
                        print(f"       ⚠️ LLM returned invalid UUID: {uuid}")
                        
                return selected

            except Exception as e:
                print(f"       ⚠️ Error resolving {item_type}: {e}")
                return []

        # Resolve Entities
        for entity in extraction.entities:
            uuids = resolve_item(entity, self.entity_index, "Entity")
            entity_uuids.extend(uuids)

        # Resolve Topics
        for topic in extraction.topics:
            uuids = resolve_item(topic, self.topic_index, "Topic")
            topic_uuids.extend(uuids)
                
        return {"entity_uuids": list(set(entity_uuids)), "topic_uuids": list(set(topic_uuids))}

    def execute_intersection_query(self, entity_uuids: List[str], topic_uuids: List[str], group_id: str) -> List[Dict]:
        """
        Executes the Intersection Query to find the 'Hub' (SectionNode).
        """
        print("   🕸️ Running Intersection Query...")
        
        if not entity_uuids and not topic_uuids:
            return []

        # Logic:
        # Find Hubs that connect to ANY of the Entity UUIDs AND ANY of the Topic UUIDs.
        # If one list is empty, just match the other.
        
        cypher = """
        MATCH (hub:SectionNode)
        WHERE hub.group_id = $group_id
        
        // Check intersection with Entities
        OPTIONAL MATCH (hub)-[:REPRESENTS]->(e:EntityNode)
        WHERE e.uuid IN $entity_uuids
        WITH hub, count(e) as entity_matches
        
        // Check intersection with Topics
        OPTIONAL MATCH (hub)-[:DISCUSSES]->(t:TopicNode)
        WHERE t.uuid IN $topic_uuids
        WITH hub, entity_matches, count(t) as topic_matches
        
        // Filter: Must match something
        WHERE (size($entity_uuids) > 0 AND entity_matches > 0) 
           OR (size($topic_uuids) > 0 AND topic_matches > 0)
           
        // Enforce Intersection if both are provided
        AND (size($entity_uuids) = 0 OR entity_matches > 0)
        AND (size($topic_uuids) = 0 OR topic_matches > 0)
        
        // Harvest
        MATCH (hub)-[:CONTAINS]->(chunk:EpisodicNode)
        OPTIONAL MATCH (fact:FactNode)-[:MENTIONED_IN]->(chunk)
        
        RETURN 
            hub.header_path as SectionContext,
            chunk.content as RawText,
            collect(distinct fact.content) as KeyFacts
        LIMIT 20
        """
        
        results = self.neo4j.query(cypher, {
            "entity_uuids": entity_uuids,
            "topic_uuids": topic_uuids,
            "group_id": group_id
        })
        
        print(f"   📊 Found {len(results)} relevant contexts.")
        return results

    def generate_response(self, query: str, context: List[Dict]) -> str:
        """
        Generates the final answer using the retrieved context.
        """
        print("   ✍️ Generating Answer...")
        
        if not context:
            return "I could not find any relevant information in the Knowledge Graph to answer your question."
            
        context_str = ""
        for i, item in enumerate(context):
            context_str += f"\n--- Source {i+1} ({item['SectionContext']}) ---\n"
            context_str += f"Text: {item['RawText']}\n"
            context_str += f"Facts: {'; '.join(item['KeyFacts'])}\n"
            
        prompt = (
            "You are a helpful assistant answering questions based ONLY on the provided Knowledge Graph context.\n"
            "Cite your sources by referring to the Section Context (e.g., 'According to [Section Name]...').\n"
            "If the information is not in the context, state that you don't know.\n\n"
            f"User Question: {query}\n\n"
            f"Context:\n{context_str}\n\n"
            "Answer:"
        )
        
        response = self.llm.invoke(prompt)
        return response.content

    def run(self, query: str, group_id: str = "default_tenant"):
        print(f"\n🚀 Starting Intersection Query for: '{query}'")
        
        # 0. Ensure Indexes
        self.ensure_indexes()
        
        # 1. Extract
        extraction = self.extract_query_components(query)
        print(f"   📋 Extracted: Entities={extraction.entities}, Topics={extraction.topics}")
        
        # 2. Resolve
        resolved = self.resolve_components(extraction, group_id)
        
        # 3. Intersect
        results = self.execute_intersection_query(
            resolved["entity_uuids"], 
            resolved["topic_uuids"], 
            group_id
        )
        
        # 4. Answer
        answer = self.generate_response(query, results)
        
        return answer
