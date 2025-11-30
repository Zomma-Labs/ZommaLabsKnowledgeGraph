from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from src.schemas.atomic_fact import AtomicFact
from src.schemas.nodes import FactNode, EntityNode, EpisodicNode
from src.schemas.relationship import RelationshipClassification
from src.tools.neo4j_client import Neo4jClient
from src.util.llm_client import get_embeddings, get_llm

class MergeDecision(BaseModel):
    should_merge: bool = Field(description="True if the new fact is semantically identical to the existing fact.")
    reasoning: str = Field(description="Reason for the decision.")

class GraphAssembler:
    def __init__(self):
        self.neo4j = Neo4jClient()
        self.embeddings = get_embeddings()
        self.llm = get_llm()

    def assemble_fact_node(self, 
                           fact_obj: AtomicFact, 
                           subject_uri: str, 
                           subject_label: str,
                           object_uri: Optional[str], 
                           object_label: Optional[str],
                           episode_uuid: str,
                           relationship_classification: Optional[RelationshipClassification] = None) -> str:
        """
        Creates a FactNode and links it to Subject, Object, and Episode.
        Returns the UUID of the FactNode (either new or merged).
        """
        
        # Debug Logging
        with open("assembler_debug.log", "a") as log:
            log.write(f"\n--- Assembling Fact: {fact_obj.fact[:50]}... ---\n")
            log.write(f"S: {subject_uri} ({subject_label}) -> O: {object_uri} ({object_label})\n")
        
        # 1. Generate Embedding
        try:
            fact_embedding = self.embeddings.embed_query(fact_obj.fact)
            with open("assembler_debug.log", "a") as log:
                log.write("✅ Embedding generated.\n")
        except Exception as e:
            with open("assembler_debug.log", "a") as log:
                log.write(f"❌ Embedding failed: {e}\n")
            raise e
        
        # Determine Fact Type from Classification
        fact_type = "statement"
        if relationship_classification:
            fact_type = relationship_classification.relationship.value
        
        # 2. Semantic Deduplication (Vector Search)
        # We search for existing FactNodes with similar content
        candidates = self.neo4j.vector_search("fact_embeddings", fact_embedding, top_k=1)
        
        fact_uuid = None
        
        if candidates:
            best_candidate = candidates[0]
            score = best_candidate['score']
            existing_node = best_candidate['node']
            
            # If score is high, verify with LLM
            if score > 0.90:
                decision = self._verify_merge(fact_obj.fact, existing_node.get('content', ''))
                if decision.should_merge:
                    print(f"   🔄 Merging with existing FactNode ({existing_node.get('uuid')})")
                    fact_uuid = existing_node.get('uuid')
        
        # 3. Create New Node if not merged
        if not fact_uuid:
            import uuid
            fact_uuid = str(uuid.uuid4())
            
            cypher_create = """
            MERGE (f:FactNode {uuid: $uuid})
            ON CREATE SET 
                f.content = $content,
                f.embedding = $embedding,
                f.created_at = datetime(),
                f.confidence = 1.0,
                f.fact_type = $fact_type
            """
            try:
                self.neo4j.query(cypher_create, {
                    "uuid": fact_uuid,
                    "content": fact_obj.fact,
                    "embedding": fact_embedding,
                    "fact_type": fact_type
                })
                print(f"   ✨ Created new FactNode ({fact_uuid}) [Type: {fact_type}]")
                with open("assembler_debug.log", "a") as log:
                    log.write(f"✅ Node created: {fact_uuid}\n")
            except Exception as e:
                with open("assembler_debug.log", "a") as log:
                    log.write(f"❌ Node creation failed: {e}\n")
                raise e
        
        # 4. Link Structure (Subject -> Fact -> Object)
        
        cypher_link = """
        MERGE (s:Entity {uri: $subj_uri})
        ON CREATE SET s.name = $subj_label
        ON MATCH SET s.name = $subj_label // Update name if it changed or was missing
        WITH s
        MATCH (f:FactNode {uuid: $fact_uuid})
        MERGE (s)-[:PERFORMED]->(f)
        """
        try:
            self.neo4j.query(cypher_link, {
                "subj_uri": subject_uri, 
                "subj_label": subject_label,
                "fact_uuid": fact_uuid
            })
            with open("assembler_debug.log", "a") as log:
                log.write(f"✅ Linked Subject: {subject_uri}\n")
        except Exception as e:
            with open("assembler_debug.log", "a") as log:
                log.write(f"❌ Link Subject failed: {e}\n")
        
        if object_uri:
            cypher_link_obj = """
            MERGE (o:Entity {uri: $obj_uri})
            ON CREATE SET o.name = $obj_label
            ON MATCH SET o.name = $obj_label
            WITH o
            MATCH (f:FactNode {uuid: $fact_uuid})
            MERGE (f)-[:TARGET]->(o)
            """
            try:
                self.neo4j.query(cypher_link_obj, {
                    "obj_uri": object_uri, 
                    "obj_label": object_label,
                    "fact_uuid": fact_uuid
                })
                with open("assembler_debug.log", "a") as log:
                    log.write(f"✅ Linked Object: {object_uri}\n")
            except Exception as e:
                with open("assembler_debug.log", "a") as log:
                    log.write(f"❌ Link Object failed: {e}\n")

        # 5. Link Provenance (Fact -> Episode)
        cypher_prov = """
        MATCH (e:EpisodicNode {uuid: $episode_uuid})
        MATCH (f:FactNode {uuid: $fact_uuid})
        MERGE (f)-[:MENTIONED_IN]->(e)
        """
        try:
            self.neo4j.query(cypher_prov, {"episode_uuid": episode_uuid, "fact_uuid": fact_uuid})
            with open("assembler_debug.log", "a") as log:
                log.write(f"✅ Linked Provenance\n")
        except Exception as e:
            with open("assembler_debug.log", "a") as log:
                log.write(f"❌ Link Provenance failed: {e}\n")
        
        return fact_uuid

    def link_causality(self, cause_uuid: str, effect_uuid: str, reasoning: str):
        """
        Creates a CAUSES edge between two FactNodes.
        """
        cypher = """
        MATCH (c:FactNode {uuid: $cause_uuid})
        MATCH (e:FactNode {uuid: $effect_uuid})
        MERGE (c)-[r:CAUSES]->(e)
        SET r.reasoning = $reasoning, r.created_at = datetime()
        """
        self.neo4j.query(cypher, {
            "cause_uuid": cause_uuid, 
            "effect_uuid": effect_uuid,
            "reasoning": reasoning
        })
        print(f"   🔗 Linked Causality: {cause_uuid} -> {effect_uuid}")

    def _verify_merge(self, new_fact: str, existing_fact: str) -> MergeDecision:
        structured_llm = self.llm.with_structured_output(MergeDecision)
        
        system_prompt = (
            "You are a Data Deduplication Expert.\n"
            "Determine if the NEW FACT describes the EXACT SAME EVENT as the EXISTING FACT.\n"
            "Ignore minor wording differences.\n"
            "If they are the same event, return True."
        )
        
        prompt = f"NEW FACT: {new_fact}\nEXISTING FACT: {existing_fact}"
        
        try:
            return structured_llm.invoke([
                ("system", system_prompt),
                ("human", prompt)
            ])
        except Exception:
            return MergeDecision(should_merge=False, reasoning="LLM Error")

    def close(self):
        self.neo4j.close()
