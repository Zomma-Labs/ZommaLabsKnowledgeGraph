import asyncio
import logging
import os
import time
import uuid
from datetime import datetime
from typing import List, Dict, Any, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.types import RetryPolicy

from src.agents.atomizer import atomizer_with_reflexion
from src.agents.entity_extractor import EntityExtractor
from src.agents.topic_librarian import TopicLibrarian
from src.agents.analyst import AnalystAgent
from src.agents.graph_assembler import GraphAssembler
from src.agents.graph_enhancer import GraphEnhancer
# from src.agents.causal_linker import CausalLinker  # Disabled for cost optimization
from src.agents.header_analyzer import HeaderAnalyzer, DimensionType
from src.agents.temporal_extractor import TemporalExtractor
from src.schemas.document_types import Chunk
from src.schemas.atomic_fact import AtomicFact
from src.schemas.relationship import RelationshipClassification
from src.schemas.nodes import TopicNode
from src.util.services import get_services
from src.util.similarity_lock import SimilarityLockManager
from src.util.deferred_dedup import get_dedup_manager, is_deferred_mode

# Configure logging - set VERBOSE=true to enable detailed output
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"
logger = logging.getLogger(__name__)

# Deferred deduplication mode: skip locks, merge duplicates after ingestion
DEFER_DEDUPLICATION = is_deferred_mode()

def log(msg: str):
    """Print only if VERBOSE mode is enabled."""
    if VERBOSE:
        print(msg)

# Define the State
class GraphState(TypedDict):
    chunk_text: str
    metadata: Dict[str, Any]
    group_id: str # Tenant ID
    header_path: str # Breadcrumbs for context
    episodic_uuid: str # New field for Provenance
    propositions: List[str] # Intermediate step
    atomic_facts: List[AtomicFact]
    resolved_entities: List[Dict[str, Any]]
    resolved_topics: List[Dict[str, Any]] # New field for topics
    classified_relationships: List[RelationshipClassification] # Kept for backward compatibility/analyst usage
    causal_links: List[Any] # New field for Causal Linking
    errors: List[str]

# Init shared services (singleton)
services = get_services()

# Init agents with shared services
topic_librarian = TopicLibrarian(services=services)
analyst = AnalystAgent(services=services)
assembler = GraphAssembler(services=services)
enhancer = GraphEnhancer(services=services)
# causal_linker = CausalLinker(services=services)  # Disabled for cost optimization
header_analyzer = HeaderAnalyzer(services=services)
entity_extractor = EntityExtractor() # New agent

# Use shared clients from services
neo4j_client = services.neo4j
embeddings = services.embeddings

# Global lock for Neo4j transactions to prevent deadlocks across chunks
# Precomputation (embeddings, LLM) runs in parallel; only the final TX is serialized

neo4j_tx_lock = asyncio.Lock()

llm_semaphore = asyncio.Semaphore(int(os.getenv("LLM_CONCURRENCY", "100")))

def initialize_episode(state: GraphState) -> Dict[str, Any]:
    log("---INITIALIZE EPISODE (Dimensional Star)---")
    chunk_text = state["chunk_text"]
    
    # Create EpisodicNode in Neo4j
    episode_uuid = str(uuid.uuid4())
    group_id = state.get("group_id")
    if not group_id:
        group_id = state["metadata"].get("group_id", "default_tenant")
        print(f"⚠️ No group_id in state, using: {group_id}")
    
    # Extract Document Info
    doc_name = state["metadata"].get("doc_id") or state["metadata"].get("filename") or state["metadata"].get("source_id") or "Unknown Document"
    file_type = state["metadata"].get("file_type", "text")
    
    # 1. Create/Merge DocumentNode
    
    # Resolve Document Date
    doc_date_str = state["metadata"].get("doc_date")
    document_date = datetime.now() # Default to now
    if doc_date_str:
        try:
            # Parse YYYY-MM-DD
            document_date = datetime.strptime(doc_date_str, "%Y-%m-%d")
        except ValueError:
            print(f"⚠️ Could not parse doc_date: {doc_date_str}, using now()")

    cypher_doc = """
    MERGE (d:DocumentNode {name: $doc_name, group_id: $group_id})
    ON CREATE SET 
        d.uuid = $doc_uuid,
        d.created_at = datetime(),
        d.document_date = $document_date,
        d.file_type = $file_type,
        d.metadata = $metadata
    RETURN d.uuid as doc_uuid
    """
    
    doc_uuid_candidate = str(uuid.uuid4())
    clean_metadata = {k: v for k, v in state["metadata"].items() if isinstance(v, (str, int, float, bool))}
    import json
    metadata_json = json.dumps(clean_metadata)

    doc_result = neo4j_client.query(cypher_doc, {
        "doc_name": doc_name,
        "group_id": group_id,
        "doc_uuid": doc_uuid_candidate,
        "file_type": file_type,
        "metadata": metadata_json,
        "document_date": document_date
    })
    
    doc_uuid = doc_result[0]['doc_uuid']
    
    # 2. Calculate Header Path
    headings = state["metadata"].get("headings", [])
    if not headings:
        headings = state["metadata"].get("breadcrumbs", [])
    
    # Filter out "Body" and strip whitespace
    headings = [h.strip() for h in headings if h.strip().lower() != "body"]
    
    header_path = " > ".join(headings) if headings else "ROOT"
    
    # 3. Create EpisodicNode (The Hub) - NOW we link Document -> Episode
    cypher_episode = """
    MERGE (e:EpisodicNode {uuid: $episode_uuid, group_id: $group_id})
    ON CREATE SET 
        e.content = $content,
        e.source = 'text',
        e.header_path = $header_path,
        e.created_at = datetime()
        
    WITH e
    MATCH (d:DocumentNode {uuid: $doc_uuid})
    MERGE (d)-[:HAS_CHUNK]->(e)
    """

    neo4j_client.query(cypher_episode, {
        "episode_uuid": episode_uuid,
        "group_id": group_id,
        "content": chunk_text,
        "header_path": header_path,
        "doc_uuid": doc_uuid
    })
    
    log(f"   Created EpisodicNode: {episode_uuid} (Header: {header_path})")

    # 4. Analyze Dimensions & Link to Episode (instead of Hub/SectionNode)
    if headings:
        # Extract context
        doc_filename = state["metadata"].get("filename", "")
        doc_context = header_analyzer.extract_document_context(chunk_text, doc_filename)
        log(f"   Document Context: {doc_context}")
        
        dimensions = header_analyzer.analyze_path(headings, document_context=doc_context)
        log(f"   Dimensions found: {[d.value for d in dimensions]}")
        
        for dim in dimensions:
            resolved_node_uuid = None
            resolved_node_type = None

            target_label = "TopicNode" if dim.type == DimensionType.TOPIC else "EntityNode"

            # Resolution Logic: Topic Ontology for Topics, Graph Search for Entities
            if dim.type == DimensionType.TOPIC:
                # Use topic ontology for topics
                topic_res = topic_librarian.resolve(dim.value)
                if topic_res and topic_res['score'] > 0.70:
                    log(f"     ✅ Topic Match: {topic_res['label']} ({topic_res['score']:.2f})")

                    cypher_merge_topic = """
                    MERGE (n:TopicNode {name: $name, group_id: $group_id})
                    ON CREATE SET n.uuid = $uuid, n.embedding = $embedding, n.topic_uri = $topic_uri
                    RETURN n.uuid as node_uuid
                    """

                    try:
                        text_to_embed = f"{topic_res['label']}: {topic_res.get('definition', '')}"
                        dim_embedding = embeddings.embed_query(text_to_embed)
                    except Exception as e:
                        log(f"   ⚠️ Topic embedding failed for '{topic_res['label']}': {e}")
                        dim_embedding = None

                    res = neo4j_client.query(cypher_merge_topic, {
                        "name": topic_res['label'],
                        "group_id": group_id,
                        "uuid": str(uuid.uuid4()),
                        "embedding": dim_embedding,
                        "topic_uri": topic_res['uri']
                    })
                    resolved_node_uuid = res[0]['node_uuid']
                    resolved_node_type = target_label

            # For entities OR topics that didn't match ontology: Graph Search / Create New
            if resolved_node_uuid is None:
                
                cypher_find = f"""
                MATCH (n:{target_label} {{group_id: $group_id}})
                WHERE n.name = $name
                RETURN n.uuid as node_uuid
                LIMIT 1
                """
                existing = neo4j_client.query(cypher_find, {"group_id": group_id, "name": dim.value})
                
                if existing:
                    log(f"     🔄 Found existing {{target_label}}: {dim.value}")
                    resolved_node_uuid = existing[0]['node_uuid']
                    resolved_node_type = target_label
                else:
                    log(f"     🆕 Creating new {{target_label}}: {dim.value}")
                    # with open("new_topics_entities.log", "a") as f:
                    #     f.write(f"[{{target_label}}] {dim.value}\n")
                    
                    try:
                        text_to_embed = f"{dim.value}: {dim.description}" if dim.description else dim.value
                        dim_embedding = embeddings.embed_query(text_to_embed)
                    except Exception as e:
                        log(f"   ⚠️ Dimension embedding failed for '{dim.value}': {e}")
                        dim_embedding = None
                        
                    cypher_create_new = f"""
                    MERGE (n:{target_label} {{name: $name, group_id: $group_id}})
                    ON CREATE SET 
                        n.uuid = $uuid, 
                        n.embedding = $embedding, 
                        n.summary = $summary,
                        n.is_fibo = false,
                        n.created_at = datetime()
                    RETURN n.uuid as node_uuid
                    """
                    res = neo4j_client.query(cypher_create_new, {
                        "name": dim.value,
                        "group_id": group_id,
                        "uuid": str(uuid.uuid4()),
                        "embedding": dim_embedding,
                        "summary": dim.description
                    })
                    resolved_node_uuid = res[0]['node_uuid']
                    resolved_node_type = target_label
            
            # Link Dimension -> Episode
            # Topic: ABOUT
            # Entity: MENTIONED_IN (or ABOUT? Doc says TopicNode-[:ABOUT]->Channel)
            if resolved_node_uuid:
                if dim.type == DimensionType.TOPIC:
                    rel_type = "ABOUT"
                else:
                    rel_type = "MENTIONED_IN"
                
                # In V2: TopicNode -[:ABOUT]-> EpisodicNode
                # Check V2 doc: (TopicNode) ─[:ABOUT]─▶ (EpisodicNode)
                
                cypher_link_dim = f"""
                MATCH (ep:EpisodicNode {{uuid: $episode_uuid}})
                MATCH (dim:{resolved_node_type} {{uuid: $dim_uuid}})
                MERGE (dim)-[:{rel_type}]->(ep)
                """
                
                # If it's an entity, maybe we also want a passive edge?
                # But these are 'header' entities, not extraction entities. 
                # Keeping it simple for now as per V2 design which emphasizes TopicNode->Episode.
                
                neo4j_client.query(cypher_link_dim, {
                    "episode_uuid": episode_uuid,
                    "dim_uuid": resolved_node_uuid
                })
    
    return {"episodic_uuid": episode_uuid, "group_id": group_id, "header_path": header_path}

def atomize_node(state: GraphState) -> Dict[str, Any]:
    log("---ATOMIZER (Decomposition Only)---")
    chunk_text = state["chunk_text"]
    metadata = state["metadata"]
    
    try:
        props = atomizer_with_reflexion(chunk_text, metadata)
        log(f"   Decomposed into {len(props)} propositions.")
        return {"propositions": props}
    except Exception as e:
        return {"errors": [f"Atomizer Error: {str(e)}"]}

async def entity_extraction_node(state: GraphState) -> Dict[str, Any]:
    log("---ENTITY EXTRACTION (Context-Aware)---")
    propositions = state["propositions"]
    chunk_text = state["chunk_text"]
    
    
    # Get header path from state (or derive if missing)
    header_path = state.get("header_path", "")
    if not header_path:
        # Fallback to metadata
        headings = state["metadata"].get("headings", [])
        if not headings:
            headings = state["metadata"].get("breadcrumbs", [])
        header_path = " > ".join([h.strip() for h in headings if h.strip().lower() != "body"])
    
    async def extract_async(prop: str) -> List[AtomicFact]:
        # Use semaphore to limit concurrent LLM calls
        async with llm_semaphore:
            # Extract with reflexion loop - validates entities and re-extracts if needed
            relations = await asyncio.to_thread(entity_extractor.extract_with_reflexion, prop, chunk_text, header_path)
        
        # Convert relations back to AtomicFact (preserving the original prop text)
        facts = []
        for rel in relations:
            # Create AtomicFact using the original 'prop' as the fact text
            # and the extracted components from the relation (including summaries)
            facts.append(AtomicFact(
                fact=prop,
                subject=rel.subject,
                subject_type=rel.subject_type,
                subject_summary=rel.subject_summary,  # Now extracted inline
                object=rel.object,
                object_type=rel.object_type,
                object_summary=rel.object_summary,  # Now extracted inline
                relationship_description=rel.relationship_description,
                topics=rel.topics,
                date_context=rel.date_context
            ))
        return facts
    
    try:
        tasks = [extract_async(p) for p in propositions]
        # We get a List[List[AtomicFact]]
        fact_lists = await asyncio.gather(*tasks)
        
        # Flatten
        all_facts = [fact for sublist in fact_lists for fact in sublist]
        
        log(f"   Extracted {len(all_facts)} AtomicFacts from {len(propositions)} propositions.")
        return {"atomic_facts": all_facts}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"errors": [f"Entity Extraction Error: {str(e)}"]}

async def parallel_resolution_node(state: GraphState) -> Dict[str, Any]:
    log("---PARALLEL RESOLUTION---")
    facts = state["atomic_facts"]
    chunk_text = state["chunk_text"]
    group_id = state.get("group_id", "default_tenant")

    # ========================================
    # PHASE 0: BATCH DEFINE ALL TOPICS FOR ENRICHED MATCHING
    # ========================================
    # Collect all topic names from all facts (subjects, objects, topic lists)
    all_topic_names = set()
    for fact in facts:
        if fact.subject_type == "Topic":
            all_topic_names.add(fact.subject)
        if fact.object and fact.object_type == "Topic":
            all_topic_names.add(fact.object)
        if fact.topics:
            all_topic_names.update(fact.topics)

    # Batch define topics in ONE LLM call
    topic_definitions = {}
    if all_topic_names:
        log(f"   📖 Batch defining {len(all_topic_names)} topics...")
        topic_definitions = topic_librarian.batch_define_topics(list(all_topic_names), chunk_text)

    # ========================================
    # PHASE 1: COLLECT ALL UNIQUE ENTITIES (with inline summaries)
    # ========================================
    unique_items_to_resolve = {} # (name, type) -> { "context": first_context, "summary": inline_summary, "original_indices": [...] }
    invalid_fact_indices = set()  # Track facts to filter out (e.g., subject Topic failed validation)

    for i, fact in enumerate(facts):
        # Subject - validate Topic types against ontology
        subj_name = fact.subject
        subj_type = fact.subject_type
        subj_summary = getattr(fact, 'subject_summary', None) or ""  # Get inline summary

        if subj_type == "Topic":
            # Validate subject against topic ontology (use enriched definition for better matching)
            enriched = topic_definitions.get(subj_name, subj_name)
            match = topic_librarian.resolve_with_definition(subj_name, enriched, context=fact.fact)
            if match:
                subj_name = match['label']  # Use canonical name
            else:
                # Not a valid topic - discard this fact entirely (subject is required)
                log(f"   ⚠️ Subject '{fact.subject}' marked as Topic but not in ontology - discarding fact")
                invalid_fact_indices.add(i)
                continue  # Skip this fact, will be filtered out before assembly

        subj_key = (subj_name, subj_type)
        if subj_key not in unique_items_to_resolve:
            unique_items_to_resolve[subj_key] = {"context": fact.fact, "summary": subj_summary, "original_indices": []}
        elif subj_summary and not unique_items_to_resolve[subj_key].get("summary"):
            # Update summary if we have one and the existing entry doesn't
            unique_items_to_resolve[subj_key]["summary"] = subj_summary
        unique_items_to_resolve[subj_key]["original_indices"].append({"fact_idx": i, "role": "subject"})

        # Object - validate Topic types against ontology
        if fact.object:
            obj_name = fact.object
            obj_type = fact.object_type
            obj_summary = getattr(fact, 'object_summary', None) or ""  # Get inline summary
            obj_valid = True  # Track if object should be added

            if obj_type == "Topic":
                # Validate object against topic ontology (use enriched definition)
                enriched = topic_definitions.get(obj_name, obj_name)
                match = topic_librarian.resolve_with_definition(obj_name, enriched, context=fact.fact)
                if match:
                    obj_name = match['label']  # Use canonical name
                else:
                    # Not a valid topic - discard this object (don't create garbage nodes)
                    log(f"   ⚠️ Object '{fact.object}' marked as Topic but not in ontology - discarding")
                    obj_valid = False

            if obj_valid:
                obj_key = (obj_name, obj_type)
                if obj_key not in unique_items_to_resolve:
                    unique_items_to_resolve[obj_key] = {"context": fact.fact, "summary": obj_summary, "original_indices": []}
                elif obj_summary and not unique_items_to_resolve[obj_key].get("summary"):
                    # Update summary if we have one and the existing entry doesn't
                    unique_items_to_resolve[obj_key]["summary"] = obj_summary
                unique_items_to_resolve[obj_key]["original_indices"].append({"fact_idx": i, "role": "object"})

        # Topics - Validate against topic ontology before processing
        if fact.topics:
            # Validate each topic and log rejected ones for manual review
            validated_topics = []
            for raw_topic in fact.topics:
                enriched = topic_definitions.get(raw_topic, raw_topic)
                match = topic_librarian.resolve_with_definition(raw_topic, enriched, context=fact.fact)
                if match:
                    validated_topics.append(match['label'])
                else:
                    # Log rejected topic for manual review (full fact text for context)
                    with open("rejected_topics.log", "a") as f:
                        f.write(f"{raw_topic}\t|\t{fact.fact}\n")

            for topic in validated_topics:
                top_key = (topic, "Topic")
                # Use topic definition from ontology as summary
                topic_def = topic_definitions.get(topic, topic)
                if top_key not in unique_items_to_resolve:
                    unique_items_to_resolve[top_key] = {"context": fact.fact, "summary": topic_def, "original_indices": []}
                unique_items_to_resolve[top_key]["original_indices"].append({"fact_idx": i, "role": "topic_list", "topic_name": topic})

    log(f"   Calculated {len(unique_items_to_resolve)} unique items to resolve from {len(facts)} facts.")

    # ========================================
    # PHASE 2: USE INLINE SUMMARIES (extracted during entity extraction)
    # ========================================
    # Build summaries_dict from inline summaries - NO LLM call needed!
    summaries_dict = {}
    missing_summaries = []

    for (name, node_type), item_data in unique_items_to_resolve.items():
        inline_summary = item_data.get("summary", "")
        if inline_summary:
            summaries_dict[name] = inline_summary
        else:
            missing_summaries.append(name)

    # Only call batch_extract_summaries for entities missing inline summaries
    if missing_summaries:
        log(f"   🔄 Generating summaries for {len(missing_summaries)} entities missing inline definitions...")
        fallback_summaries = await asyncio.to_thread(
            enhancer.batch_extract_summaries,
            missing_summaries,
            chunk_text,
            batch_size=15
        )
        summaries_dict.update(fallback_summaries)
    else:
        log(f"   ✅ Using {len(summaries_dict)} inline summaries (no LLM call needed)")

    # ========================================
    # PHASE 2.5: BATCH EMBED ALL ENTITY NAMES
    # (Avoids rate limits by batching 128 texts per API call)
    # ========================================
    log(f"   📦 Batch embedding {len(unique_items_to_resolve)} entities...")
    
    # Collect all texts to embed
    sorted_keys = list(unique_items_to_resolve.keys())
    embedding_texts = []
    for name, node_type in sorted_keys:
        summary = summaries_dict.get(name, "Entity")
        normalized_name = name.strip().title()
        embedding_texts.append(f"{normalized_name}: {summary}")
    
    # Batch embed (128 per API call) with retry logic
    BATCH_SIZE = 128
    MAX_RETRIES = 3
    all_entity_embeddings = []
    
    for batch_start in range(0, len(embedding_texts), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(embedding_texts))
        batch = embedding_texts[batch_start:batch_end]
        
        # Retry with exponential backoff
        for attempt in range(MAX_RETRIES):
            try:
                batch_embeddings = embeddings.embed_documents(batch)
                all_entity_embeddings.extend(batch_embeddings)
                break  # Success!
            except Exception as e:
                error_str = str(e).lower()
                if "rate limit" in error_str or "429" in error_str or "requests per minute" in error_str:
                    if attempt < MAX_RETRIES - 1:
                        wait_time = (2 ** attempt)  # 1s, 2s, 4s
                        print(f"   ⏳ Rate limit hit, waiting {wait_time}s before retry {attempt + 2}/{MAX_RETRIES}...")
                        time.sleep(wait_time)
                        continue
                print(f"   ⚠️ Batch embedding failed: {e}")
                all_entity_embeddings.extend([None] * len(batch))
                break
    
    # Create lookup: (name, node_type) -> embedding
    entity_embedding_lookup = {}
    for idx, key in enumerate(sorted_keys):
        entity_embedding_lookup[key] = all_entity_embeddings[idx] if idx < len(all_entity_embeddings) else None
    
    log(f"   ✅ Batch embedded {len(all_entity_embeddings)} entities")

    # ========================================
    # PHASE 3: PARALLEL RESOLUTION WITH PRE-COMPUTED EMBEDDINGS
    # ========================================

    async def resolve_item_async(name: str, node_type: str, summary: str, cached_embedding: List[float] | None) -> tuple[str | None, str | None, str]:
        """Resolution with pre-computed summary AND embedding"""
        if not name:
            return None, None, ""

        # Normalize
        name = name.strip().title()

        # In deferred mode, skip locking entirely - we'll deduplicate after ingestion
        if DEFER_DEDUPLICATION:
            return await asyncio.to_thread(resolve_single_item, name, summary, cached_embedding, node_type)

        # Normal mode: use similarity lock to prevent concurrent duplicate creation
        if cached_embedding:
            lock_id = await SimilarityLockManager.acquire_lock(cached_embedding, name)
            try:
                return await asyncio.to_thread(resolve_single_item, name, summary, cached_embedding, node_type)
            finally:
                await SimilarityLockManager.release_lock(lock_id)
        else:
            return await asyncio.to_thread(resolve_single_item, name, summary, None, node_type)

    def resolve_single_item(name: str, summary: str, cached_embedding: List[float] | None, node_type: str) -> tuple[str | None, str | None, str]:
        if not name:
            return None, None, ""

        # In deferred mode: skip LLM resolution, just create entities
        # We'll deduplicate after ingestion using clustering
        if not DEFER_DEDUPLICATION:
            # 1. Graph Deduplication (Semantic) - ONLY in normal mode
            # Pass cached embedding
            candidates = enhancer.find_graph_candidates(name, summary, group_id=group_id, node_type=node_type, cached_embedding=cached_embedding)
            decision = enhancer.resolve_entity_against_graph(name, summary, candidates)

            if decision['decision'] == 'MERGE' and decision['target_uuid']:
                # Merge summaries: combine existing entity's summary with new summary
                existing_summary = ""
                for c in candidates:
                    if c.get('uuid') == decision['target_uuid']:
                        existing_summary = c.get('summary', '')
                        break

                if existing_summary and summary:
                    merged_summary = enhancer.merge_summaries(name, existing_summary, summary)
                    # Update the entity with merged summary
                    label = "TopicNode" if node_type == "Topic" else "EntityNode"
                    try:
                        neo4j_client.query(f"""
                            MATCH (n:{label} {{uuid: $uuid, group_id: $group_id}})
                            SET n.summary = $summary
                        """, {"uuid": decision['target_uuid'], "group_id": group_id, "summary": merged_summary})
                    except Exception as e:
                        log(f"   ⚠️ Summary merge update failed: {e}")
                    return decision['target_uuid'], name, merged_summary

                return decision['target_uuid'], name, summary

        # 2. New Entity/Topic
        node_embedding = cached_embedding
        new_uuid = str(uuid.uuid4())

        # In deferred mode: DON'T write to Neo4j - just register for later
        if DEFER_DEDUPLICATION:
            dedup_manager = get_dedup_manager()
            dedup_manager.register_entity(
                uuid=new_uuid,
                name=name,
                node_type=node_type,
                summary=summary,
                embedding=node_embedding,
                group_id=group_id
            )
            log(f"   📝 Registered pending {node_type}: {name}")
            return new_uuid, name, summary

        # Normal mode: CREATE IMMEDIATELY (Atomic Write)
        log(f"   🆕 Creating New {node_type} (Atomic): {name}")

        label = "TopicNode" if node_type == "Topic" else "EntityNode"

        cypher_atomic_create = f"""
        MERGE (n:{label} {{name: $name, group_id: $group_id}})
        ON CREATE SET
            n.uuid = $uuid,
            n.summary = $summary,
            n.embedding = $embedding,
            n.created_at = datetime()
        ON MATCH SET
            n.embedding = CASE WHEN n.embedding IS NULL THEN $embedding ELSE n.embedding END,
            n.summary = CASE WHEN n.summary IS NULL OR n.summary = "" THEN $summary ELSE n.summary END
        RETURN n.uuid as node_uuid
        """

        try:
            neo4j_client.query(cypher_atomic_create, {
                "name": name,
                "group_id": group_id,
                "uuid": new_uuid,
                "summary": summary,
                "embedding": node_embedding
            })
        except Exception as e:
            print(f"   ⚠️ Atomic Create Failed: {e}")

        return new_uuid, name, summary

    # Wrapper for Analyst processing
    async def classify_fact_relationship(fact: AtomicFact) -> RelationshipClassification:
        # Use relationship_description for vector search (more focused than full fact text)
        # Fallback to full fact if relationship_description is missing
        if fact.relationship_description:
            query_str = fact.relationship_description
        else:
            query_str = f"{fact.subject} {fact.fact} {fact.object if fact.object else ''}"
        return await asyncio.to_thread(analyst.classify_relationship, query_str)

    # ========================================
    # PHASE 4: CREATE RESOLUTION & CLASSIFICATION TASKS
    # ========================================

    resolution_tasks = []
    task_idx_to_key = {}

    for idx, (name, node_type) in enumerate(sorted_keys):
        # Use pre-computed summary AND embedding
        summary = summaries_dict.get(name, "Entity")  # Fallback if missing
        cached_embedding = entity_embedding_lookup.get((name, node_type))
        resolution_tasks.append(resolve_item_async(name, node_type, summary, cached_embedding))
        task_idx_to_key[idx] = (name, node_type)

    classification_tasks = [classify_fact_relationship(fact) for fact in facts]

    # ========================================
    # PHASE 5: RUN EVERYTHING IN PARALLEL
    # ========================================

    log(f"   🚀 Running {len(resolution_tasks)} resolution + {len(classification_tasks)} classification tasks...")
    all_results = await asyncio.gather(*(resolution_tasks + classification_tasks))

    resolution_results = all_results[:len(resolution_tasks)]
    classification_results = all_results[len(resolution_tasks):]

    # ========================================
    # PHASE 6: DISTRIBUTE RESULTS BACK TO FACTS
    # ========================================
    # Init structure
    resolved_entities = [{} for _ in facts] # List of dicts
    resolved_topics_by_fact = [[] for _ in facts] # List of lists
    
    # Map results to a lookup dict
    resolution_lookup = {} # (name, type) -> (uuid, label, summary)
    for idx, res in enumerate(resolution_results):
        key = task_idx_to_key[idx]
        resolution_lookup[key] = res
        
    # Re-assemble
    for (name, node_type), item_data in unique_items_to_resolve.items():
        # unpack new return values: uuid, label, summary
        res_uuid, label, summary = resolution_lookup[(name, node_type)]
        
        for usage in item_data["original_indices"]:
            f_idx = usage["fact_idx"]
            role = usage["role"]
            
            if role == "subject":
                resolved_entities[f_idx]["subject_uuid"] = res_uuid
                resolved_entities[f_idx]["subject_label"] = label
                resolved_entities[f_idx]["subject_summary"] = summary
                resolved_entities[f_idx]["subject_type"] = node_type
                
            elif role == "object":
                resolved_entities[f_idx]["object_uuid"] = res_uuid
                resolved_entities[f_idx]["object_label"] = label
                resolved_entities[f_idx]["object_summary"] = summary
                resolved_entities[f_idx]["object_type"] = node_type # Need to pass this to assembler
                
            elif role == "topic_list":
                # For topics, we append to the list for that fact
                resolved_topics_by_fact[f_idx].append({
                    "uuid": res_uuid,
                    "label": label,
                    "summary": summary
                })

    # ========================================
    # PHASE 7: FILTER OUT INVALID FACTS
    # ========================================
    # Facts with invalid topic subjects are removed before assembly
    if invalid_fact_indices:
        valid_indices = [i for i in range(len(facts)) if i not in invalid_fact_indices]
        facts = [facts[i] for i in valid_indices]
        resolved_entities = [resolved_entities[i] for i in valid_indices]
        resolved_topics_by_fact = [resolved_topics_by_fact[i] for i in valid_indices]
        classification_results = [classification_results[i] for i in valid_indices]
        log(f"   ⚠️ Filtered out {len(invalid_fact_indices)} facts with invalid topic subjects")

    # Ensure Object fields are populated (None if no object)
    for i, fact in enumerate(facts):
        if not fact.object:
            resolved_entities[i]["object_uuid"] = None
            resolved_entities[i]["object_label"] = None
            resolved_entities[i]["object_summary"] = ""
            resolved_entities[i]["object_type"] = "Entity" # Default

    return {
        "atomic_facts": facts,  # Return filtered facts to update state
        "resolved_entities": resolved_entities,
        "resolved_topics": resolved_topics_by_fact,
        "classified_relationships": classification_results
    }

def causal_linking_node(state: GraphState) -> Dict[str, Any]:
    log("---CAUSAL LINKING---")
    facts = state["atomic_facts"]
    chunk_text = state["chunk_text"]
    
    try:
        links = causal_linker.extract_causality(facts, chunk_text)
        log(f"   Found {len(links)} causal links.")
        return {"causal_links": links}
    except Exception as e:
        print(f"Causal linking error: {e}")
        return {"causal_links": []}

def _should_retry_assembler(error: Exception) -> bool:
    """Retry on rate limits and Neo4j deadlocks."""
    error_str = str(error).lower()
    return "requests per minute" in error_str or "deadlockdetected" in error_str


async def assemble_node(state: GraphState) -> Dict[str, Any]:
    log("---ASSEMBLER---")
    facts = state["atomic_facts"]
    resolved_ents = state["resolved_entities"]
    resolved_tops = state.get("resolved_topics", [[] for _ in facts])
    links = state.get("causal_links", [])  # Default to empty list (causal linking disabled)
    episode_uuid = state["episodic_uuid"]
    group_id = state.get("group_id", "default_tenant")
    
    fact_uuids = []

    # 1. Create Fact Nodes
    # Prepare all facts_data for batch processing
    facts_data = []
    for i in range(len(facts)):
        fact = facts[i]
        res_ent = resolved_ents[i]
        res_top = resolved_tops[i]
        rel = state["classified_relationships"][i] if "classified_relationships" in state and i < len(state["classified_relationships"]) else None

        facts_data.append({
            "fact_obj": fact,
            "subject_uuid": res_ent["subject_uuid"],
            "subject_label": res_ent["subject_label"],
            "object_uuid": res_ent.get("object_uuid"),
            "object_label": res_ent.get("object_label"),
            "relationship_classification": rel,
            "subject_summary": res_ent.get("subject_summary", ""),
            "object_summary": res_ent.get("object_summary", ""),
            "subject_type": res_ent.get("subject_type", "Entity"),
            "object_type": res_ent.get("object_type", "Entity"),
            "topics": res_top
        })

    # PHASE 1 & 2: Precompute embeddings + deduplication (runs in parallel across chunks)
    precomputed = await asyncio.to_thread(
        assembler.precompute_facts_batch,
        facts_data,
        group_id
    )

    # PHASE 3: Execute Neo4j transaction OR register for deferred write
    try:
        if DEFER_DEDUPLICATION:
            # Deferred mode: DON'T write to Neo4j - register facts for later
            dedup_manager = get_dedup_manager()
            for i, (fact_data, pc) in enumerate(zip(facts_data, precomputed)):
                dedup_manager.register_fact(
                    fact_data=fact_data,
                    precomputed=pc,
                    episode_uuid=episode_uuid,
                    group_id=group_id
                )
            log(f"   📝 Registered {len(facts_data)} pending facts (will write after clustering)")
            fact_uuids = [pc.get("new_fact_uuid") or pc.get("resolved_fact_uuid") for pc in precomputed]
        else:
            # Normal mode: serialize Neo4j writes to prevent deadlocks
            async with neo4j_tx_lock:
                fact_uuids = await asyncio.to_thread(
                    assembler.execute_facts_batch,
                    precomputed,
                    facts_data,
                    episode_uuid,
                    group_id
                )
            log(f"   ✅ Successfully assembled {len([u for u in fact_uuids if u])}/{len(facts)} facts")

    except Exception as e:
        # Check if this is a transient error that RetryPolicy should handle
        if _should_retry_assembler(e):
            raise  # Let RetryPolicy handle it

        # Non-transient error: handle gracefully
        error_msg = f"Error in batch assembly: {e}"
        print(error_msg)
        return {"errors": state.get("errors", []) + [error_msg]}
        
    # 2. Link Causality
    for link in links:
        cause_idx = link.cause_index
        effect_idx = link.effect_index
        
        if (0 <= cause_idx < len(fact_uuids) and 
            0 <= effect_idx < len(fact_uuids) and
            fact_uuids[cause_idx] and 
            fact_uuids[effect_idx]):
            
            try:
                assembler.link_causality(
                    cause_uuid=fact_uuids[cause_idx],
                    effect_uuid=fact_uuids[effect_idx],
                    reasoning=link.reasoning,
                    group_id=group_id
                )
            except Exception as e:
                print(f"Error linking causality: {e}")
            
    return {}

# Build the Graph
workflow = StateGraph(GraphState)

workflow.add_node("initialize_episode", initialize_episode)
workflow.add_node("atomizer", atomize_node)
workflow.add_node("entity_extraction", entity_extraction_node)
workflow.add_node("parallel_resolution", parallel_resolution_node)
# workflow.add_node("causal_linking", causal_linking_node)  # Disabled for cost optimization
workflow.add_node(
    "assembler",
    assemble_node,
    retry_policy=RetryPolicy(
        max_attempts=3,
        initial_interval=1.0,
        backoff_factor=2.0,
        retry_on=_should_retry_assembler
    )
)

workflow.set_entry_point("initialize_episode")

workflow.add_edge("initialize_episode", "atomizer")
workflow.add_edge("atomizer", "entity_extraction")
workflow.add_edge("entity_extraction", "parallel_resolution")
workflow.add_edge("parallel_resolution", "assembler")  # Direct edge, skipping causal_linking
workflow.add_edge("assembler", END)

app = workflow.compile()


async def ingest_document(chunks: List[Chunk], group_id: str) -> None:
    """
    High-level entry point to ingest an entire document.
    
    1. Runs Temporal Extractor on the full document context.
    2. Runs the GraphRAG pipeline for each chunk in parallel (or batched).
    """
    print(f"--- 📥 Starting Ingestion for Document: {chunks[0].doc_id if chunks else 'Unknown'} ({len(chunks)} chunks) ---")
    
    if not chunks:
        print("⚠️ No chunks to ingest.")
        return

    # 1. Temporal Extraction (Document Level)
    log("⏳ Running Temporal Extraction...")
    extractor = TemporalExtractor()
    # Assume title is the filename or doc_id
    title = chunks[0].metadata.get("origin_filename") or chunks[0].doc_id
    
    enriched_chunks = await asyncio.to_thread(extractor.enrich_chunks, chunks, title)
    doc_date = enriched_chunks[0].metadata.get("doc_date") if enriched_chunks else None
    log(f"   📅 Document Date Extracted: {doc_date}")
    
    # 2. Pipeline Execution (Chunk Level)
    log("🚀 Dispatching chunks to pipeline...")
    
    async def process_chunk(chunk: Chunk):
        inputs = {
            "chunk_text": chunk.body,
            "metadata": {
                "doc_id": chunk.doc_id,
                "filename": chunk.metadata.get("origin_filename") or f"{chunk.doc_id}.pdf",
                "chunk_id": chunk.chunk_id,
                "headings": list(chunk.breadcrumbs),
                "breadcrumbs": list(chunk.breadcrumbs),
                "group_id": group_id,
                "doc_date": chunk.metadata.get("doc_date") # Passed from enrichment
            }
        }
        try:
            # Run the compiled graph
            result = await app.ainvoke(inputs)
            # Check for errors in result?
            if result.get("errors"):
                 print(f"❌ Error processing chunk {chunk.chunk_id}: {result['errors']}")
            else:
                 pass # Success silent
        except Exception as e:
            print(f"❌ Critical pipeline failure for chunk {chunk.chunk_id}: {e}")

    # Run in parallel with semaphore if needed, for now all at once or batched?
    # Python asyncio.gather might overwhelm if too many chunks.
    # Let's use a semaphore.
    # No chunk-level semaphore - let LLM_CONCURRENCY handle rate limiting
    # All chunks process in parallel, individual LLM calls are throttled by llm_semaphore
    tasks = [process_chunk(c) for c in enriched_chunks]
    
    # Run all
    await asyncio.gather(*tasks)

    # 3. Post-Ingestion: Cluster + Write (if deferred mode enabled)
    if DEFER_DEDUPLICATION:
        log("🔄 Finalizing: Clustering entities and writing to Neo4j...")
        dedup_manager = get_dedup_manager()

        # Finalize: cluster, write canonical entities, write facts with remapped UUIDs
        stats = dedup_manager.finalize(
            neo4j_client=neo4j_client,
            assembler=assembler,
            group_id=group_id,
            similarity_threshold=0.85
        )

        log(f"   📊 Final stats: {stats['entities_written']} entities, {stats['facts_written']} facts")
        log(f"   🔍 Deduplicated: {stats['duplicates_found']} duplicates merged into {stats['clusters_found']} clusters")

    log(f"✅ Ingestion Complete for {chunks[0].doc_id}.")
