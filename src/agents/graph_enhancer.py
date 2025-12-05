"""
MODULE: Graph Enhancer
DESCRIPTION: 
    Enhances the graph extraction process by:
    1. Reflexion: Checking for missed facts and promoting Concepts to Entities.
    2. Deduplication: Resolving entities against the existing graph (fallback to FIBO).
    3. Enrichment: Extracting attributes and summaries.
"""

from typing import List, Dict, Any, Optional, TYPE_CHECKING
from pydantic import BaseModel, Field
from src.tools.neo4j_client import Neo4jClient

if TYPE_CHECKING:
    from src.util.services import Services

class MissedFacts(BaseModel):
    missed_facts: List[str] = Field(description="List of facts that were missed in the initial extraction.")
    reasoning: str = Field(description="Why these facts are important and should be included.")

class EntityResolution(BaseModel):
    decision: str = Field(description="One of: 'MERGE', 'CREATE_NEW'")
    target_uuid: Optional[str] = Field(description="UUID of the existing node to merge with, if MERGE.")
    reasoning: str = Field(description="Reason for the decision.")

class GraphEnhancer:
    def __init__(self, services: Optional["Services"] = None):
        if services is None:
            from src.util.services import get_services
            services = get_services()
        self.llm = services.llm
        self.embeddings = services.embeddings
        self.neo4j = services.neo4j

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

    def extract_entity_description(self, entity_name: str, context_text: str) -> str:
        """
        Generates a brief 1-sentence description of what the entity IS in this context.
        """
        prompt = (
            f"Based on the text below, provide a brief, 1-sentence description of what '{entity_name}' IS.\n"
            f"Example: 'A multinational technology company.' or 'A type of fruit.'\n"
            f"TEXT: {context_text}\n"
            f"DESCRIPTION:"
        )
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            print(f"Description extraction failed: {e}")
            return "Entity"

    def find_graph_candidates(self, entity_name: str, entity_description: str, group_id: str, top_k: int = 5) -> List[Dict]:
        """
        Finds candidates using Exact Match (Step 1) and Vector Search (Step 2).
        """
        candidates = []
        seen_uuids = set()

        # 1. Exact Name Match
        cypher_exact = """
        MATCH (n:Entity {name: $name, group_id: $group_id})
        RETURN n.uuid as uuid, n.name as name, n.summary as summary, labels(n) as labels
        LIMIT 1
        """
        try:
            exact_matches = self.neo4j.query(cypher_exact, {"name": entity_name, "group_id": group_id})
            for match in exact_matches:
                match['score'] = 1.0
                candidates.append(match)
                seen_uuids.add(match['uuid'])
        except Exception as e:
            print(f"Exact match query failed: {e}")

        # 2. Vector Search (Fallback/Supplementary)
        # Embed "Name: Description"
        query_text = f"{entity_name}: {entity_description}"
        try:
            vector = self.embeddings.embed_query(query_text)
            # Note: We are searching 'entity_embeddings' index. Ensure it exists.
            results = self.neo4j.vector_search("entity_embeddings", vector, top_k, filters={"group_id": group_id})
            
            for record in results:
                node = record['node']
                uuid = node.get("uuid") or node.get("uri") # Handle potential schema variations
                
                if uuid not in seen_uuids:
                    candidates.append({
                        "uuid": uuid,
                        "name": node.get("name"),
                        "summary": node.get("summary", ""),
                        "labels": node.get("labels", []),
                        "score": record['score']
                    })
                    seen_uuids.add(uuid)
                    
            return candidates
        except Exception as e:
            print(f"Graph candidate search failed: {e}")
            return candidates # Return whatever we found in exact match

    def resolve_entity_against_graph(self, entity_name: str, entity_description: str, candidates: List[Dict]) -> Dict[str, Any]:
        """
        Uses LLM to decide whether to merge with a candidate or create a new node.
        """
        if not candidates:
            return {"decision": "CREATE_NEW", "target_uuid": None}
            
        # Optimization: If we have an exact match (score=1.0), check if descriptions align
        # But for safety, we still ask LLM unless it's overwhelmingly obvious?
        # Actually, let's let the LLM decide even for exact matches to handle "Apple" (Fruit) vs "Apple" (Corp) collision if names are identical.
        # Wait, if names are identical, exact match returns it. We need to know if it's the SAME entity.
        
        structured_llm = self.llm.with_structured_output(EntityResolution)
        
        candidates_str = "\n".join([
            f"- ID: {c['uuid']}, Name: {c['name']}, Description: {c.get('summary', 'N/A')}, Score: {c['score']:.2f}" 
            for c in candidates
        ])
        
        system_prompt = (
            "You are an Entity Resolution Expert.\n"
            "Decide if the New Entity matches any of the Existing Graph Candidates.\n\n"
            "Rules:\n"
            "1. MERGE only if you are confident it is the SAME real-world entity.\n"
            "2. Pay close attention to DESCRIPTIONS. 'Apple' (Tech Company) != 'Apple' (Fruit).\n"
            "3. If the name is the same but the description implies a different entity, CREATE_NEW.\n"
            "4. If the candidate list is empty or irrelevant, CREATE_NEW."
        )
        
        prompt = (
            f"NEW ENTITY: {entity_name}\n"
            f"DESCRIPTION: {entity_description}\n\n"
            f"EXISTING CANDIDATES:\n{candidates_str}"
        )
        
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
