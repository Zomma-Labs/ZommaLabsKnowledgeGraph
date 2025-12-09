from typing import List, Optional, Dict, Any, TYPE_CHECKING
from pydantic import BaseModel, Field
from src.schemas.atomic_fact import AtomicFact
from src.schemas.nodes import FactNode, EntityNode, EpisodicNode
from src.schemas.relationship import RelationshipClassification, ACTIVE_TO_PASSIVE
from src.tools.neo4j_client import Neo4jClient

if TYPE_CHECKING:
    from src.util.services import Services

class MergeDecision(BaseModel):
    should_merge: bool = Field(description="True if the new fact is semantically identical to the existing fact.")
    reasoning: str = Field(description="Reason for the decision.")

class GraphAssembler:
    def __init__(self, services: Optional["Services"] = None):
        if services is None:
            from src.util.services import get_services
            services = get_services()
        self.neo4j = services.neo4j
        self.embeddings = services.embeddings
        self.llm = services.llm

    def assemble_fact_node(self, 
                           fact_obj: AtomicFact, 
                           subject_uuid: str, 
                           subject_label: str,
                           object_uuid: Optional[str], 
                           object_label: Optional[str],
                           episode_uuid: str,
                           group_id: str,
                           relationship_classification: Optional[RelationshipClassification] = None,
                           subject_summary: str = "",
                           object_summary: str = "",
                           subject_type: str = "Entity",
                           object_type: str = "Entity") -> str:
        """
        Creates a FactNode and links it to Subject, Object, and Episode.
        Returns the UUID of the FactNode (either new or merged).
        """
        
        # Debug Logging
        with open("assembler_debug.log", "a") as log:
            log.write(f"\n--- Assembling Fact: {fact_obj.fact[:50]}... ---\n")
            log.write(f"S: {subject_uuid} ({subject_label}) [{subject_type}] -> O: {object_uuid} ({object_label}) [{object_type}]\n")
        
        # 1. Generate Embedding for Fact
        try:
            fact_embedding = self.embeddings.embed_query(fact_obj.fact)
            with open("assembler_debug.log", "a") as log:
                log.write("✅ Fact Embedding generated.\n")
        except Exception as e:
            with open("assembler_debug.log", "a") as log:
                log.write(f"❌ Fact Embedding failed: {e}\n")
            raise e
        
        # Determine Fact Type from Classification
        fact_type = "statement"
        if relationship_classification:
            fact_type = relationship_classification.relationship.value
        
        # 2. Semantic Deduplication (Vector Search)
        # We search for existing FactNodes with similar content
        candidates = self.neo4j.vector_search("fact_embeddings", fact_embedding, top_k=1, filters={"group_id": group_id})
        
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
            MERGE (f:FactNode {uuid: $uuid, group_id: $group_id})
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
                    "group_id": group_id,
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
        
        # 4. Link Provenance (Fact -> Episode)
        # This keeps the FactNode grounded, even if not linked to entities directly
        cypher_prov = """
        MATCH (e:EpisodicNode {uuid: $episode_uuid, group_id: $group_id})
        MATCH (f:FactNode {uuid: $fact_uuid, group_id: $group_id})
        MERGE (f)-[:MENTIONED_IN]->(e)
        """
        try:
            self.neo4j.query(cypher_prov, {"episode_uuid": episode_uuid, "fact_uuid": fact_uuid, "group_id": group_id})
            with open("assembler_debug.log", "a") as log:
                log.write(f"✅ Linked Provenance\n")
        except Exception as e:
            with open("assembler_debug.log", "a") as log:
                log.write(f"❌ Link Provenance failed: {e}\n")
        
        # 5. Link Structure (Subject -> Episode -> Object) with Semantic Edges
        # Determine Edge Types
        active_edge = "RELATED_TO"
        passive_edge = "RELATED_TO_BY"
        
        if relationship_classification:
            rel_type = relationship_classification.relationship
            active_edge = rel_type.value
            passive_edge = ACTIVE_TO_PASSIVE.get(rel_type, f"GOT_{active_edge}")
            
        
        # Generate Subject Embedding
        subj_embedding = None
        if subject_label:
            try:
                subj_text = f"{subject_label}: {subject_summary}" if subject_summary else subject_label
                subj_embedding = self.embeddings.embed_query(subj_text)
            except Exception:
                pass

        # Link Subject -> [Active Edge] -> Episode
        # Determine Node Label based on subject_type
        subj_node_label = "TopicNode" if subject_type == "Topic" else "EntityNode"
        
        # NOTE: Node should already exist from Resolution step. matching on UUID.
        # We use MERGE for safety in case of async delays or failures, but strictly on UUID.
        # But we don't want to create duplicates if it doesn't exist? Ideally MATCH.
        # But main_pipeline ensures creation. Let's use MATCH.
        # Actually, MERGE on UUID is safer if correct. MATCH might fail if race condition.
        # Let's use MERGE (n {uuid: ...})
        
        cypher_link_subj = f"""
        MERGE (s:{subj_node_label} {{uuid: $subj_uuid, group_id: $group_id}})
        ON CREATE SET 
            s.name = $subj_label,
            s.summary = $subj_desc,
            s.embedding = $subj_embedding,
            s.created_at = datetime()
        ON MATCH SET 
            s.name = $subj_label,
            s.summary = CASE WHEN s.summary IS NULL OR s.summary = "" THEN $subj_desc ELSE s.summary END
        WITH s
        MATCH (e:EpisodicNode {{uuid: $episode_uuid, group_id: $group_id}})
        MERGE (s)-[r:{active_edge}]->(e)
        SET r.fact_id = $fact_uuid, r.confidence = $confidence
        """
        try:
            self.neo4j.query(cypher_link_subj, {
                "subj_uuid": subject_uuid, 
                "subj_label": subject_label,
                "subj_desc": subject_summary,
                "subj_embedding": subj_embedding,
                "episode_uuid": episode_uuid,
                "fact_uuid": fact_uuid,
                "confidence": relationship_classification.confidence if relationship_classification else 1.0,
                "group_id": group_id
            })
            with open("assembler_debug.log", "a") as log:
                log.write(f"✅ Linked Subject ({subj_node_label}, {active_edge}): {subject_uri} -> Episode\n")
        except Exception as e:
            with open("assembler_debug.log", "a") as log:
                log.write(f"❌ Link Subject failed: {e}\n")
        
        # Link Episode -> [Passive Edge] -> Object
        if object_uuid:
            # Generate Object Embedding
            obj_embedding = None
            if object_label:
                try:
                    obj_text = f"{object_label}: {object_summary}" if object_summary else object_label
                    obj_embedding = self.embeddings.embed_query(obj_text)
                except Exception:
                    pass

            # Determine Node Label based on object_type
            obj_node_label = "TopicNode" if object_type == "Topic" else "EntityNode"
            
            cypher_link_obj = f"""
            MERGE (o:{obj_node_label} {{uuid: $obj_uuid, group_id: $group_id}})
            ON CREATE SET 
                o.name = $obj_label,
                o.summary = $obj_desc,
                o.embedding = $obj_embedding,
                o.created_at = datetime()
            ON MATCH SET 
                o.name = $obj_label,
                o.summary = CASE WHEN o.summary IS NULL OR o.summary = "" THEN $obj_desc ELSE o.summary END
            WITH o
            MATCH (e:EpisodicNode {{uuid: $episode_uuid, group_id: $group_id}})
            MERGE (e)-[r:{passive_edge}]->(o)
            SET r.fact_id = $fact_uuid, r.confidence = $confidence
            """
            try:
                self.neo4j.query(cypher_link_obj, {
                    "obj_uuid": object_uuid, 
                    "obj_label": object_label,
                    "obj_desc": object_summary,
                    "obj_embedding": obj_embedding,
                    "episode_uuid": episode_uuid,
                    "fact_uuid": fact_uuid,
                    "confidence": relationship_classification.confidence if relationship_classification else 1.0,
                    "group_id": group_id
                })
                with open("assembler_debug.log", "a") as log:
                    log.write(f"✅ Linked Object ({obj_node_label}, {passive_edge}): Episode -> {object_uuid}\n")
            except Exception as e:
                with open("assembler_debug.log", "a") as log:
                    log.write(f"❌ Link Object failed: {e}\n")

        return fact_uuid

    def link_causality(self, cause_uuid: str, effect_uuid: str, reasoning: str, group_id: str):
        """
        Creates a CAUSES edge between two FactNodes.
        """
        cypher = """
        MATCH (c:FactNode {uuid: $cause_uuid, group_id: $group_id})
        MATCH (e:FactNode {uuid: $effect_uuid, group_id: $group_id})
        MERGE (c)-[r:CAUSES]->(e)
        SET r.reasoning = $reasoning, r.created_at = datetime()
        """
        self.neo4j.query(cypher, {
            "cause_uuid": cause_uuid, 
            "effect_uuid": effect_uuid,
            "reasoning": reasoning,
            "group_id": group_id
        })
        print(f"   🔗 Linked Causality: {cause_uuid} -> {effect_uuid}")

    def link_topic_to_episode(self, topic_uuid: str, episode_uuid: str, group_id: str):
        """
        Links a TopicNode to an EpisodicNode via [:ABOUT].
        """
        cypher = """
        MATCH (t:TopicNode {uuid: $topic_uuid, group_id: $group_id})
        MATCH (e:EpisodicNode {uuid: $episode_uuid, group_id: $group_id})
        MERGE (t)-[:ABOUT]->(e)
        """
        try:
            self.neo4j.query(cypher, {
                "topic_uuid": topic_uuid,
                "episode_uuid": episode_uuid,
                "group_id": group_id
            })
            with open("assembler_debug.log", "a") as log:
                log.write(f"✅ Linked Topic {topic_uuid} -> Episode\n")
        except Exception as e:
            with open("assembler_debug.log", "a") as log:
                log.write(f"❌ Link Topic failed: {e}\n")

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
