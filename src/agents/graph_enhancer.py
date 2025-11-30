"""
MODULE: Graph Enhancer
DESCRIPTION: 
    Enhances the graph extraction process by:
    1. Reflexion: Checking for missed facts and promoting Concepts to Entities.
    2. Deduplication: Resolving entities against the existing graph (fallback to FIBO).
    3. Enrichment: Extracting attributes and summaries.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from src.util.llm_client import get_llm, get_embeddings
from src.tools.neo4j_client import Neo4jClient

class MissedFacts(BaseModel):
    missed_facts: List[str] = Field(description="List of facts that were missed in the initial extraction.")
    reasoning: str = Field(description="Why these facts are important and should be included.")

class EntityResolution(BaseModel):
    decision: str = Field(description="One of: 'MERGE', 'CREATE_NEW'")
    target_uuid: Optional[str] = Field(description="UUID of the existing node to merge with, if MERGE.")
    reasoning: str = Field(description="Reason for the decision.")

class GraphEnhancer:
    def __init__(self, neo4j_client: Optional[Neo4jClient] = None):
        self.llm = get_llm()
        self.embeddings = get_embeddings()
        self.neo4j = neo4j_client or Neo4jClient()

    def reflexion_check(self, chunk_text: str, existing_facts: List[Any]) -> List[str]:
        """
        Asks the LLM if any important facts were missed, specifically looking to promote Concepts to Entities.
        """
        structured_llm = self.llm.with_structured_output(MissedFacts)
        
        # Convert existing facts to string for context
        facts_str = "\n".join([str(f) for f in existing_facts])
        
        system_prompt = (
            "You are a Quality Assurance Auditor for a Knowledge Graph.\n"
            "Your goal is to review the Extracted Facts against the Source Text and identify MISSING information.\n\n"
            "CRITICAL GOAL: PROMOTE CONCEPTS TO ENTITIES.\n"
            "If the text says 'the tech giant' and we extracted it as a generic Concept, but the text implies it is 'Google', "
            "you MUST flag this as a missed fact: 'The tech giant is Google'.\n\n"
            "Rules:\n"
            "1. Only report SUBSTANTIAL missing facts that change the meaning or add specific entities.\n"
            "2. Ignore minor wording differences.\n"
            "3. Focus on specific Names, Dates, and Financial Metrics."
        )
        
        prompt = f"SOURCE TEXT:\n{chunk_text}\n\nEXTRACTED FACTS:\n{facts_str}"
        
        try:
            response = structured_llm.invoke([
                ("system", system_prompt),
                ("human", prompt)
            ])
            return response.missed_facts
        except Exception as e:
            print(f"Reflexion failed: {e}")
            return []

    def find_graph_candidates(self, entity_name: str, group_id: str, top_k: int = 5) -> List[Dict]:
        """
        Performs a vector search on the Neo4j graph to find similar entities.
        """
        try:
            vector = self.embeddings.embed_query(entity_name)
            results = self.neo4j.vector_search("entity_embeddings", vector, top_k, filters={"group_id": group_id})
            
            candidates = []
            for record in results:
                node = record['node']
                score = record['score']
                candidates.append({
                    "uuid": node.get("uri"), # Assuming URI is used as UUID/ID
                    "name": node.get("name"),
                    "labels": node.get("labels", []), # Neo4j python driver might return labels differently, but node object usually has them
                    "score": score
                })
            return candidates
        except Exception as e:
            print(f"Graph candidate search failed: {e}")
            return []

    def resolve_entity_against_graph(self, entity_name: str, candidates: List[Dict]) -> Dict[str, Any]:
        """
        Uses LLM to decide whether to merge with a candidate or create a new node.
        """
        if not candidates:
            return {"decision": "CREATE_NEW", "target_uuid": None}
            
        structured_llm = self.llm.with_structured_output(EntityResolution)
        
        candidates_str = "\n".join([
            f"- ID: {c['uuid']}, Name: {c['name']}, Score: {c['score']:.2f}" 
            for c in candidates
        ])
        
        system_prompt = (
            "You are an Entity Resolution Expert.\n"
            "Decide if the New Entity matches any of the Existing Graph Candidates.\n\n"
            "Rules:\n"
            "1. MERGE only if you are confident it is the SAME real-world entity.\n"
            "2. If the name is similar but represents a different thing (e.g. 'Apple' vs 'Applebee's'), CREATE_NEW.\n"
            "3. If the candidate list is empty or irrelevant, CREATE_NEW."
        )
        
        prompt = f"NEW ENTITY: {entity_name}\n\nEXISTING CANDIDATES:\n{candidates_str}"
        
        try:
            response = structured_llm.invoke([
                ("system", system_prompt),
                ("human", prompt)
            ])
            return {
                "decision": response.decision,
                "target_uuid": response.target_uuid,
                "reasoning": response.reasoning
            }
        except Exception as e:
            print(f"Resolution failed: {e}")
            # Default to new if LLM fails
            return {"decision": "CREATE_NEW", "target_uuid": None}

    def extract_attributes(self, entity_name: str, context_text: str) -> Dict[str, Any]:
        """
        Extracts attributes and summary for an entity from the text.
        """
        # Placeholder for attribute extraction logic
        # For now, we can just return a simple summary
        return {"summary": f"Entity extracted from: {context_text[:50]}..."}

if __name__ == "__main__":
    # Test
    enhancer = GraphEnhancer()
    print("GraphEnhancer initialized.")
