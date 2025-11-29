from typing import List, Optional
from src.schemas.atomic_fact import AtomicFact
from src.schemas.relationship import RelationshipClassification
from src.tools.neo4j_client import Neo4jClient

class GraphAssembler:
    def __init__(self):
        self.neo4j = Neo4jClient()

    def assemble_and_write(self, 
                           fact_obj: AtomicFact, 
                           subject_uri: str, 
                           subject_label: Optional[str],
                           object_uri: Optional[str], 
                           object_label: Optional[str],
                           relationship: RelationshipClassification):
        """
        Merges the Fact, Subject, Object (Entity or Concept), and Relationship into the Graph.
        """
        
        # 1. Prepare Subject Logic
        subj_id = subject_uri if subject_uri else fact_obj.subject
        # Use FIBO label if available, else raw text
        subj_name = subject_label if subject_label else fact_obj.subject
        
        # 2. Prepare Object Logic
        if object_uri:
            obj_id = object_uri
            obj_type_label = "Entity"
            # Use FIBO label if available, else raw text
            obj_name = object_label if object_label else fact_obj.object
        else:
            obj_id = fact_obj.object 
            obj_type_label = "Concept"
            obj_name = fact_obj.object

        # 3. Cypher Query
        rel_type = relationship.relationship.value
        
        cypher = f"""
        MERGE (s:Entity {{uri: $subj_id}})
        ON CREATE SET s.name = $subj_name
        
        MERGE (o:{obj_type_label} {{uri: $obj_id}})
        ON CREATE SET o.name = $obj_name
        
        MERGE (s)-[r:{rel_type}]->(o)
        SET r.fact = $fact_text,
            r.date = $date_context,
            r.confidence = $confidence
        """
        
        params = {
            "subj_id": subj_id,
            "subj_name": subj_name,
            "obj_id": obj_id,
            "obj_name": obj_name,
            "fact_text": fact_obj.fact,
            "date_context": fact_obj.date_context,
            "confidence": relationship.confidence
        }
        
        try:
            self.neo4j.query(cypher, params)
            print(f"   ✅ Graph Write: ({subj_name}) -[{rel_type}]-> ({obj_name})")
        except Exception as e:
            print(f"   ❌ Graph Write Failed: {e}")

    def close(self):
        self.neo4j.close()
