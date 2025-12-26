import os
import logging
import time
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from pydantic import BaseModel, Field
from src.schemas.atomic_fact import AtomicFact
from src.schemas.nodes import FactNode, EntityNode, EpisodicNode
from src.schemas.relationship import RelationshipClassification, ACTIVE_TO_PASSIVE
from src.tools.neo4j_client import Neo4jClient

if TYPE_CHECKING:
    from src.util.services import Services

# Configure logging for assembler
logger = logging.getLogger(__name__)
DEBUG_ASSEMBLER = os.getenv("DEBUG_ASSEMBLER", "false").lower() == "true"
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"

def log(msg: str):
    """Print only if VERBOSE mode is enabled."""
    if VERBOSE:
        print(msg)

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
        
        # Debug Logging (controlled by DEBUG_ASSEMBLER env var)
        if DEBUG_ASSEMBLER:
            logger.debug(f"Assembling Fact: {fact_obj.fact[:50]}...")
            logger.debug(f"S: {subject_uuid} ({subject_label}) [{subject_type}] -> O: {object_uuid} ({object_label}) [{object_type}]")
        
        # 1. Generate Embedding for Fact
        try:
            fact_embedding = self.embeddings.embed_query(fact_obj.fact)
            if DEBUG_ASSEMBLER:
                logger.debug("✅ Fact Embedding generated.")
        except Exception as e:
            logger.error(f"❌ Fact Embedding failed: {e}")
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
                    log(f"   🔄 Merging with existing FactNode ({existing_node.get('uuid')})")
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
                log(f"   ✨ Created new FactNode ({fact_uuid}) [Type: {fact_type}]")
                if DEBUG_ASSEMBLER:
                    logger.debug(f"✅ Node created: {fact_uuid}")
            except Exception as e:
                logger.error(f"❌ Node creation failed: {e}")
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
            if DEBUG_ASSEMBLER:
                logger.debug("✅ Linked Provenance")
        except Exception as e:
            logger.error(f"❌ Link Provenance failed: {e}")
        
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
        
        # Strategy 3: Lock Episode FIRST to prevent deadlocks
        cypher_link_subj = f"""
        MATCH (e:EpisodicNode {{uuid: $episode_uuid, group_id: $group_id}})
        WITH e
        MERGE (s:{subj_node_label} {{uuid: $subj_uuid, group_id: $group_id}})
        ON CREATE SET
            s.name = $subj_label,
            s.summary = $subj_desc,
            s.embedding = $subj_embedding,
            s.created_at = datetime()
        ON MATCH SET
            s.name = $subj_label,
            s.summary = CASE WHEN s.summary IS NULL OR s.summary = "" THEN $subj_desc ELSE s.summary END
        WITH s, e
        MERGE (s)-[r:{active_edge} {{fact_id: $fact_uuid}}]->(e)
        SET r.confidence = $confidence
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
            if DEBUG_ASSEMBLER:
                logger.debug(f"✅ Linked Subject ({subj_node_label}, {active_edge}): {subject_uuid} -> Episode")
        except Exception as e:
            logger.error(f"❌ Link Subject failed: {e}")
        
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
            
            # Strategy 3: Lock Episode FIRST to prevent deadlocks
            cypher_link_obj = f"""
            MATCH (e:EpisodicNode {{uuid: $episode_uuid, group_id: $group_id}})
            WITH e
            MERGE (o:{obj_node_label} {{uuid: $obj_uuid, group_id: $group_id}})
            ON CREATE SET
                o.name = $obj_label,
                o.summary = $obj_desc,
                o.embedding = $obj_embedding,
                o.created_at = datetime()
            ON MATCH SET
                o.name = $obj_label,
                o.summary = CASE WHEN o.summary IS NULL OR o.summary = "" THEN $obj_desc ELSE o.summary END
            WITH e, o
            MERGE (e)-[r:{passive_edge} {{fact_id: $fact_uuid}}]->(o)
            SET r.confidence = $confidence
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
                if DEBUG_ASSEMBLER:
                    logger.debug(f"✅ Linked Object ({obj_node_label}, {passive_edge}): Episode -> {object_uuid}")
            except Exception as e:
                logger.error(f"❌ Link Object failed: {e}")

        return fact_uuid

    def assemble_facts_batch(self,
                             facts_data: List[Dict[str, Any]],
                             episode_uuid: str,
                             group_id: str) -> List[Optional[str]]:
        """
        Assembles multiple facts: precomputes embeddings, then executes transaction.
        For deadlock prevention, use precompute_facts_batch + execute_facts_batch separately.
        """
        precomputed = self.precompute_facts_batch(facts_data, group_id)
        return self.execute_facts_batch(precomputed, facts_data, episode_uuid, group_id)

    def precompute_facts_batch(self,
                               facts_data: List[Dict[str, Any]],
                               group_id: str) -> List[Dict[str, Any]]:
        """
        PHASE 1 & 2: Precompute all embeddings, vector searches, and LLM merge decisions.
        This is safe to run in parallel across chunks.
        
        Returns:
            List of precomputed dicts with embeddings and resolved UUIDs.
        """
        import uuid as uuid_module
        
        log(f"   📊 Precomputing embeddings for {len(facts_data)} facts...")
        
        # Step 1a: Collect ALL texts to embed with their destinations
        texts_to_embed = []  # List of (text, fact_idx, field_name, topic_uuid_or_none)
        
        precomputed = []  # List of dicts with all precomputed data per fact
        
        for i, fact_data in enumerate(facts_data):
            fact_obj = fact_data["fact_obj"]
            subject_label = fact_data["subject_label"]
            subject_summary = fact_data.get("subject_summary", "")
            object_label = fact_data.get("object_label")
            object_summary = fact_data.get("object_summary", "")
            topics = fact_data.get("topics", [])
            
            precomputed_fact = {
                "fact_data": fact_data,
                "fact_embedding": None,
                "subj_embedding": None,
                "obj_embedding": None,
                "topic_embeddings": {},  # topic_uuid -> embedding
                "resolved_fact_uuid": None,  # If merged with existing
                "new_fact_uuid": None,  # If creating new
            }
            precomputed.append(precomputed_fact)
            
            # Collect fact text
            texts_to_embed.append((fact_obj.fact, i, "fact", None))
            
            # Collect subject text
            if subject_label:
                subj_text = f"{subject_label}: {subject_summary}" if subject_summary else subject_label
                texts_to_embed.append((subj_text, i, "subject", None))
            
            # Collect object text
            if object_label:
                obj_text = f"{object_label}: {object_summary}" if object_summary else object_label
                texts_to_embed.append((obj_text, i, "object", None))
            
            # Collect topic texts
            for topic in topics:
                if topic.get("uuid"):
                    t_text = f"{topic['label']}: {topic.get('summary', '')}" if topic.get('summary') else topic['label']
                    texts_to_embed.append((t_text, i, "topic", topic["uuid"]))
                    # Initialize the topic slot
                    precomputed_fact["topic_embeddings"][topic["uuid"]] = None
        
        # Step 1b: Batch embed all texts (128 per API call to stay under limits)
        BATCH_SIZE = 128
        MAX_RETRIES = 3
        all_embeddings = []
        
        log(f"   📦 Batching {len(texts_to_embed)} texts into {(len(texts_to_embed) + BATCH_SIZE - 1) // BATCH_SIZE} API calls...")
        
        for batch_start in range(0, len(texts_to_embed), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(texts_to_embed))
            batch_texts = [t[0] for t in texts_to_embed[batch_start:batch_end]]
            
            # Retry with exponential backoff
            for attempt in range(MAX_RETRIES):
                try:
                    batch_embeddings = self.embeddings.embed_documents(batch_texts)
                    all_embeddings.extend(batch_embeddings)
                    break  # Success!
                except Exception as e:
                    error_str = str(e).lower()
                    if "rate limit" in error_str or "429" in error_str or "requests per minute" in error_str:
                        if attempt < MAX_RETRIES - 1:
                            wait_time = (2 ** attempt)  # 1s, 2s, 4s
                            logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 2}/{MAX_RETRIES}...")
                            time.sleep(wait_time)
                            continue
                    logger.error(f"Batch embedding failed for batch starting at {batch_start}: {e}")
                    # Fill with None for this batch
                    all_embeddings.extend([None] * len(batch_texts))
                    break
        
        # Step 1c: Map embeddings back to precomputed structure
        for idx, (text, fact_idx, field_name, topic_uuid) in enumerate(texts_to_embed):
            embedding = all_embeddings[idx] if idx < len(all_embeddings) else None
            
            if field_name == "fact":
                precomputed[fact_idx]["fact_embedding"] = embedding
            elif field_name == "subject":
                precomputed[fact_idx]["subj_embedding"] = embedding
            elif field_name == "object":
                precomputed[fact_idx]["obj_embedding"] = embedding
            elif field_name == "topic" and topic_uuid:
                precomputed[fact_idx]["topic_embeddings"][topic_uuid] = embedding
        
        log(f"   ✅ Embedded {len(all_embeddings)} texts successfully")
        
        # =====================================================================
        # PHASE 2: VECTOR SEARCH & DEDUPLICATION
        # =====================================================================
        log(f"   🔍 Running deduplication checks...")
        
        for i, pc in enumerate(precomputed):
            fact_embedding = pc["fact_embedding"]
            fact_obj = pc["fact_data"]["fact_obj"]
            
            if fact_embedding:
                # Vector search for existing similar facts
                candidates = self.neo4j.vector_search(
                    "fact_embeddings", fact_embedding, top_k=1, filters={"group_id": group_id}
                )
                
                if candidates:
                    best = candidates[0]
                    if best['score'] > 0.90:
                        # LLM verification
                        decision = self._verify_merge(fact_obj.fact, best['node'].get('content', ''))
                        if decision.should_merge:
                            pc["resolved_fact_uuid"] = best['node'].get('uuid')
                            if DEBUG_ASSEMBLER:
                                logger.debug(f"Fact {i} will merge with existing: {pc['resolved_fact_uuid']}")
            
            # Generate new UUID if not merging
            if not pc["resolved_fact_uuid"]:
                pc["new_fact_uuid"] = str(uuid_module.uuid4())
        
        return precomputed

    def execute_facts_batch(self,
                            precomputed: List[Dict[str, Any]],
                            facts_data: List[Dict[str, Any]],
                            episode_uuid: str,
                            group_id: str) -> List[Optional[str]]:
        """
        PHASE 3: Execute Neo4j transaction with precomputed data.
        This should be serialized (one at a time) to prevent deadlocks.
        
        Args:
            precomputed: Output from precompute_facts_batch
            facts_data: Original facts data
            episode_uuid: UUID of the EpisodicNode
            group_id: Tenant identifier
        
        Returns:
            List of FactNode UUIDs (None if fact failed)
        """
        log(f"   ⚡ Executing Neo4j transaction for {len(precomputed)} facts...")
        
        fact_uuids = []
        
        with self.neo4j.driver.session() as session:
            with session.begin_transaction() as tx:
                for i, pc in enumerate(precomputed):
                    try:
                        fact_data = pc["fact_data"]
                        fact_obj = fact_data["fact_obj"]
                        relationship_classification = fact_data.get("relationship_classification")
                        
                        # Use precomputed UUID (either merged or new)
                        fact_uuid = pc["resolved_fact_uuid"] or pc["new_fact_uuid"]
                        
                        # Assemble fact using precomputed embeddings
                        self._assemble_fact_in_transaction_precomputed(
                            tx=tx,
                            fact_uuid=fact_uuid,
                            fact_obj=fact_obj,
                            fact_embedding=pc["fact_embedding"],
                            is_new=(pc["resolved_fact_uuid"] is None),
                            subject_uuid=fact_data["subject_uuid"],
                            subject_label=fact_data["subject_label"],
                            subject_summary=fact_data.get("subject_summary", ""),
                            subject_type=fact_data.get("subject_type", "Entity"),
                            subj_embedding=pc["subj_embedding"],
                            object_uuid=fact_data.get("object_uuid"),
                            object_label=fact_data.get("object_label"),
                            object_summary=fact_data.get("object_summary", ""),
                            object_type=fact_data.get("object_type", "Entity"),
                            obj_embedding=pc["obj_embedding"],
                            episode_uuid=episode_uuid,
                            group_id=group_id,
                            relationship_classification=relationship_classification
                        )
                        fact_uuids.append(fact_uuid)
                        
                        # Link Topics with precomputed embeddings
                        topics = fact_data.get("topics", [])
                        for topic in topics:
                            if topic.get("uuid"):
                                self._link_topic_in_transaction_precomputed(
                                    tx=tx,
                                    topic_uuid=topic["uuid"],
                                    topic_label=topic["label"],
                                    topic_summary=topic.get("summary", ""),
                                    topic_embedding=pc["topic_embeddings"].get(topic["uuid"]),
                                    episode_uuid=episode_uuid,
                                    group_id=group_id
                                )
                    
                    except Exception as e:
                        print(f"   ⚠️ Fact {i} failed in batch: {e}")
                        fact_uuids.append(None)
                
                # Commit all at once
                tx.commit()
                log(f"   ✅ Batch committed {len([u for u in fact_uuids if u])} facts successfully")
        
        return fact_uuids

    def _assemble_fact_in_transaction(self, tx, fact_obj, subject_uuid, subject_label,
                                       object_uuid, object_label, episode_uuid, group_id,
                                       relationship_classification, subject_summary, object_summary,
                                       subject_type, object_type) -> str:
        """
        Helper method to assemble a single fact within an existing transaction.
        Similar to assemble_fact_node but uses tx.run() instead of self.neo4j.query()
        """

        # 1. Generate Embedding for Fact
        fact_embedding = self.embeddings.embed_query(fact_obj.fact)

        # Determine Fact Type
        fact_type = "statement"
        if relationship_classification:
            fact_type = relationship_classification.relationship.value

        # 2. Semantic Deduplication (Vector Search)
        candidates = self.neo4j.vector_search("fact_embeddings", fact_embedding, top_k=1, filters={"group_id": group_id})

        fact_uuid = None

        if candidates:
            best_candidate = candidates[0]
            score = best_candidate['score']
            existing_node = best_candidate['node']

            if score > 0.90:
                decision = self._verify_merge(fact_obj.fact, existing_node.get('content', ''))
                if decision.should_merge:
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
            tx.run(cypher_create, {
                "uuid": fact_uuid,
                "group_id": group_id,
                "content": fact_obj.fact,
                "embedding": fact_embedding,
                "fact_type": fact_type
            })

        # 4. Link Provenance
        cypher_prov = """
        MATCH (e:EpisodicNode {uuid: $episode_uuid, group_id: $group_id})
        MATCH (f:FactNode {uuid: $fact_uuid, group_id: $group_id})
        MERGE (f)-[:MENTIONED_IN]->(e)
        """
        tx.run(cypher_prov, {"episode_uuid": episode_uuid, "fact_uuid": fact_uuid, "group_id": group_id})

        # 5. Link Subject and Object
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

        # Link Subject
        subj_node_label = "TopicNode" if subject_type == "Topic" else "EntityNode"

        cypher_link_subj = f"""
        MATCH (e:EpisodicNode {{uuid: $episode_uuid, group_id: $group_id}})
        WITH e
        MERGE (s:{subj_node_label} {{uuid: $subj_uuid, group_id: $group_id}})
        ON CREATE SET
            s.name = $subj_label,
            s.summary = $subj_desc,
            s.embedding = $subj_embedding,
            s.created_at = datetime()
        ON MATCH SET
            s.name = $subj_label,
            s.summary = CASE WHEN s.summary IS NULL OR s.summary = "" THEN $subj_desc ELSE s.summary END
        WITH s, e
        MERGE (s)-[r:{active_edge} {{fact_id: $fact_uuid}}]->(e)
        SET r.confidence = $confidence
        """
        tx.run(cypher_link_subj, {
            "subj_uuid": subject_uuid,
            "subj_label": subject_label,
            "subj_desc": subject_summary,
            "subj_embedding": subj_embedding,
            "episode_uuid": episode_uuid,
            "fact_uuid": fact_uuid,
            "confidence": relationship_classification.confidence if relationship_classification else 1.0,
            "group_id": group_id
        })

        # Link Object
        if object_uuid:
            obj_embedding = None
            if object_label:
                try:
                    obj_text = f"{object_label}: {object_summary}" if object_summary else object_label
                    obj_embedding = self.embeddings.embed_query(obj_text)
                except Exception:
                    pass

            obj_node_label = "TopicNode" if object_type == "Topic" else "EntityNode"

            cypher_link_obj = f"""
            MATCH (e:EpisodicNode {{uuid: $episode_uuid, group_id: $group_id}})
            WITH e
            MERGE (o:{obj_node_label} {{uuid: $obj_uuid, group_id: $group_id}})
            ON CREATE SET
                o.name = $obj_label,
                o.summary = $obj_desc,
                o.embedding = $obj_embedding,
                o.created_at = datetime()
            ON MATCH SET
                o.name = $obj_label,
                o.summary = CASE WHEN o.summary IS NULL OR o.summary = "" THEN $obj_desc ELSE o.summary END
            WITH e, o
            MERGE (e)-[r:{passive_edge} {{fact_id: $fact_uuid}}]->(o)
            SET r.confidence = $confidence
            """
            tx.run(cypher_link_obj, {
                "obj_uuid": object_uuid,
                "obj_label": object_label,
                "obj_desc": object_summary,
                "obj_embedding": obj_embedding,
                "episode_uuid": episode_uuid,
                "fact_uuid": fact_uuid,
                "confidence": relationship_classification.confidence if relationship_classification else 1.0,
                "group_id": group_id
            })

        return fact_uuid

    def _link_topic_in_transaction(self, tx, topic_uuid: str, topic_label: str,
                                   topic_summary: str, episode_uuid: str, group_id: str):
        """
        Links a TopicNode to an EpisodicNode within an existing transaction.
        """
        # Generate Topic Embedding
        topic_embedding = None
        try:
            t_text = f"{topic_label}: {topic_summary}" if topic_summary else topic_label
            topic_embedding = self.embeddings.embed_query(t_text)
        except Exception:
            pass

        cypher_topic_merge = """
        MERGE (t:TopicNode {uuid: $topic_uuid, group_id: $group_id})
        ON CREATE SET
            t.name = $name,
            t.summary = $summary,
            t.embedding = $embedding,
            t.is_fibo = false,
            t.created_at = datetime()
        ON MATCH SET
            t.embedding = CASE WHEN t.embedding IS NULL THEN $embedding ELSE t.embedding END,
            t.summary = CASE WHEN t.summary IS NULL OR t.summary = "" THEN $summary ELSE t.summary END
        """
        tx.run(cypher_topic_merge, {
            "topic_uuid": topic_uuid,
            "group_id": group_id,
            "name": topic_label,
            "summary": topic_summary,
            "embedding": topic_embedding
        })

        # Link to Episode
        cypher_link = """
        MATCH (t:TopicNode {uuid: $topic_uuid, group_id: $group_id})
        MATCH (e:EpisodicNode {uuid: $episode_uuid, group_id: $group_id})
        MERGE (t)-[:ABOUT]->(e)
        """
        tx.run(cypher_link, {
            "topic_uuid": topic_uuid,
            "episode_uuid": episode_uuid,
            "group_id": group_id
        })

    def _assemble_fact_in_transaction_precomputed(
        self, tx, fact_uuid: str, fact_obj, fact_embedding, is_new: bool,
        subject_uuid: str, subject_label: str, subject_summary: str, subject_type: str, subj_embedding,
        object_uuid, object_label, object_summary: str, object_type: str, obj_embedding,
        episode_uuid: str, group_id: str, relationship_classification
    ) -> None:
        """
        Assembles a single fact within an existing transaction using PRECOMPUTED embeddings.
        No network I/O happens inside this method - all embeddings are passed in.
        """
        # Determine Fact Type
        fact_type = "statement"
        if relationship_classification:
            fact_type = relationship_classification.relationship.value

        # Create FactNode only if it's new (not merged with existing)
        if is_new and fact_embedding:
            cypher_create = """
            MERGE (f:FactNode {uuid: $uuid, group_id: $group_id})
            ON CREATE SET
                f.content = $content,
                f.embedding = $embedding,
                f.created_at = datetime(),
                f.confidence = 1.0,
                f.fact_type = $fact_type
            """
            tx.run(cypher_create, {
                "uuid": fact_uuid,
                "group_id": group_id,
                "content": fact_obj.fact,
                "embedding": fact_embedding,
                "fact_type": fact_type
            })

        # Link Provenance (Fact -> Episode)
        cypher_prov = """
        MATCH (e:EpisodicNode {uuid: $episode_uuid, group_id: $group_id})
        MATCH (f:FactNode {uuid: $fact_uuid, group_id: $group_id})
        MERGE (f)-[:MENTIONED_IN]->(e)
        """
        tx.run(cypher_prov, {
            "episode_uuid": episode_uuid,
            "fact_uuid": fact_uuid,
            "group_id": group_id
        })

        # Determine Edge Types
        active_edge = "RELATED_TO"
        passive_edge = "RELATED_TO_BY"

        if relationship_classification:
            rel_type = relationship_classification.relationship
            active_edge = rel_type.value
            passive_edge = ACTIVE_TO_PASSIVE.get(rel_type, f"GOT_{active_edge}")

        # Link Subject -> Episode (using precomputed embedding)
        subj_node_label = "TopicNode" if subject_type == "Topic" else "EntityNode"

        cypher_link_subj = f"""
        MATCH (e:EpisodicNode {{uuid: $episode_uuid, group_id: $group_id}})
        WITH e
        MERGE (s:{subj_node_label} {{uuid: $subj_uuid, group_id: $group_id}})
        ON CREATE SET
            s.name = $subj_label,
            s.summary = $subj_desc,
            s.embedding = $subj_embedding,
            s.created_at = datetime()
        ON MATCH SET
            s.name = $subj_label,
            s.summary = CASE WHEN s.summary IS NULL OR s.summary = "" THEN $subj_desc ELSE s.summary END
        WITH s, e
        MERGE (s)-[r:{active_edge} {{fact_id: $fact_uuid}}]->(e)
        SET r.confidence = $confidence
        """
        tx.run(cypher_link_subj, {
            "subj_uuid": subject_uuid,
            "subj_label": subject_label,
            "subj_desc": subject_summary,
            "subj_embedding": subj_embedding,
            "episode_uuid": episode_uuid,
            "fact_uuid": fact_uuid,
            "confidence": relationship_classification.confidence if relationship_classification else 1.0,
            "group_id": group_id
        })

        # Link Episode -> Object (using precomputed embedding)
        if object_uuid:
            obj_node_label = "TopicNode" if object_type == "Topic" else "EntityNode"

            cypher_link_obj = f"""
            MATCH (e:EpisodicNode {{uuid: $episode_uuid, group_id: $group_id}})
            WITH e
            MERGE (o:{obj_node_label} {{uuid: $obj_uuid, group_id: $group_id}})
            ON CREATE SET
                o.name = $obj_label,
                o.summary = $obj_desc,
                o.embedding = $obj_embedding,
                o.created_at = datetime()
            ON MATCH SET
                o.name = $obj_label,
                o.summary = CASE WHEN o.summary IS NULL OR o.summary = "" THEN $obj_desc ELSE o.summary END
            WITH e, o
            MERGE (e)-[r:{passive_edge} {{fact_id: $fact_uuid}}]->(o)
            SET r.confidence = $confidence
            """
            tx.run(cypher_link_obj, {
                "obj_uuid": object_uuid,
                "obj_label": object_label,
                "obj_desc": object_summary,
                "obj_embedding": obj_embedding,
                "episode_uuid": episode_uuid,
                "fact_uuid": fact_uuid,
                "confidence": relationship_classification.confidence if relationship_classification else 1.0,
                "group_id": group_id
            })

    def _link_topic_in_transaction_precomputed(
        self, tx, topic_uuid: str, topic_label: str, topic_summary: str,
        topic_embedding, episode_uuid: str, group_id: str
    ) -> None:
        """
        Links a TopicNode to an EpisodicNode using a PRECOMPUTED embedding.
        No network I/O happens inside this method.
        """
        cypher_topic_merge = """
        MERGE (t:TopicNode {uuid: $topic_uuid, group_id: $group_id})
        ON CREATE SET
            t.name = $name,
            t.summary = $summary,
            t.embedding = $embedding,
            t.is_fibo = false,
            t.created_at = datetime()
        ON MATCH SET
            t.embedding = CASE WHEN t.embedding IS NULL THEN $embedding ELSE t.embedding END,
            t.summary = CASE WHEN t.summary IS NULL OR t.summary = "" THEN $summary ELSE t.summary END
        """
        tx.run(cypher_topic_merge, {
            "topic_uuid": topic_uuid,
            "group_id": group_id,
            "name": topic_label,
            "summary": topic_summary,
            "embedding": topic_embedding
        })

        # Link to Episode
        cypher_link = """
        MATCH (t:TopicNode {uuid: $topic_uuid, group_id: $group_id})
        MATCH (e:EpisodicNode {uuid: $episode_uuid, group_id: $group_id})
        MERGE (t)-[:ABOUT]->(e)
        """
        tx.run(cypher_link, {
            "topic_uuid": topic_uuid,
            "episode_uuid": episode_uuid,
            "group_id": group_id
        })

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
        log(f"   🔗 Linked Causality: {cause_uuid} -> {effect_uuid}")

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
            if DEBUG_ASSEMBLER:
                logger.debug(f"✅ Linked Topic {topic_uuid} -> Episode")
        except Exception as e:
            logger.error(f"❌ Link Topic failed: {e}")

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
