import asyncio
import uuid
from typing import List, Dict, Any, TypedDict
from langgraph.graph import StateGraph, END

from src.agents.atomizer import atomizer
from src.agents.entity_extractor import EntityExtractor
from src.agents.FIBO_librarian import FIBOLibrarian
from src.agents.analyst import AnalystAgent
from src.agents.graph_assembler import GraphAssembler
from src.agents.graph_enhancer import GraphEnhancer
from src.agents.causal_linker import CausalLinker
from src.agents.header_analyzer import HeaderAnalyzer, DimensionType
from src.schemas.atomic_fact import AtomicFact
from src.schemas.relationship import RelationshipClassification
from src.schemas.nodes import TopicNode
from src.util.services import get_services
from src.util.similarity_lock import SimilarityLockManager

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
librarian = FIBOLibrarian(services=services)
analyst = AnalystAgent(services=services)
assembler = GraphAssembler(services=services)
enhancer = GraphEnhancer(services=services)
causal_linker = CausalLinker(services=services)
header_analyzer = HeaderAnalyzer(services=services)
entity_extractor = EntityExtractor() # New agent

# Use shared clients from services
neo4j_client = services.neo4j
embeddings = services.embeddings

def initialize_episode(state: GraphState) -> Dict[str, Any]:
    print("---INITIALIZE EPISODE (Dimensional Star)---")
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
    cypher_doc = """
    MERGE (d:DocumentNode {name: $doc_name, group_id: $group_id})
    ON CREATE SET 
        d.uuid = $doc_uuid,
        d.created_at = datetime(),
        d.document_date = datetime(),
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
        "metadata": metadata_json
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
    
    print(f"   Created EpisodicNode: {episode_uuid} (Header: {header_path})")

    # 4. Analyze Dimensions & Link to Episode (instead of Hub/SectionNode)
    if headings:
        # Extract context
        doc_filename = state["metadata"].get("filename", "")
        doc_context = header_analyzer.extract_document_context(chunk_text, doc_filename)
        print(f"   Document Context: {doc_context}")
        
        dimensions = header_analyzer.analyze_path(headings, document_context=doc_context)
        print(f"   Dimensions found: {[d.value for d in dimensions]}")
        
        for dim in dimensions:
            resolved_node_uuid = None
            resolved_node_type = None
            
            # Resolution Logic (Standard Paradigm: FIBO -> Graph -> Create)
            # 1. FIBO Resolution
            resolve_query = f"{dim.value} {dim.description}" if dim.description else dim.value
            fibo_res = librarian.resolve(resolve_query)
            
            if fibo_res and fibo_res['score'] > 0.9:
                print(f"     ✅ FIBO Match: {fibo_res['label']} ({fibo_res['score']:.2f})")
                
                target_label = "TopicNode" if dim.type == DimensionType.TOPIC else "EntityNode"
                
                
                cypher_merge_fibo = f"""
                MERGE (n:{target_label} {{name: $name, group_id: $group_id}}) 
                ON CREATE SET n.uuid = $uuid, n.embedding = $embedding, n.fibo_uri = $fibo_uri, n.fibo_id = $fibo_uri
                RETURN n.uuid as node_uuid
                """
                
                # Embed dimension value + description
                try:
                    text_to_embed = f"{dim.value}: {dim.description}" if dim.description else dim.value
                    dim_embedding = embeddings.embed_query(text_to_embed)
                except:
                    dim_embedding = None

                res = neo4j_client.query(cypher_merge_fibo, {
                    "name": fibo_res['label'],
                    "group_id": group_id,
                    "uuid": str(uuid.uuid4()),
                    "embedding": dim_embedding,
                    "fibo_uri": fibo_res['uri']
                })
                resolved_node_uuid = res[0]['node_uuid']
                resolved_node_type = target_label
                
            else:
                # Graph Search / Create New
                target_label = "TopicNode" if dim.type == DimensionType.TOPIC else "EntityNode"
                
                cypher_find = f"""
                MATCH (n:{target_label} {{group_id: $group_id}})
                WHERE n.name = $name
                RETURN n.uuid as node_uuid
                LIMIT 1
                """
                existing = neo4j_client.query(cypher_find, {"group_id": group_id, "name": dim.value})
                
                if existing:
                    print(f"     🔄 Found existing {{target_label}}: {dim.value}")
                    resolved_node_uuid = existing[0]['node_uuid']
                    resolved_node_type = target_label
                else:
                    print(f"     🆕 Creating new {{target_label}}: {dim.value}")
                    # with open("new_topics_entities.log", "a") as f:
                    #     f.write(f"[{{target_label}}] {dim.value}\n")
                    
                    try:
                        text_to_embed = f"{dim.value}: {dim.description}" if dim.description else dim.value
                        dim_embedding = embeddings.embed_query(text_to_embed)
                    except:
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
    print("---ATOMIZER (Decomposition Only)---")
    chunk_text = state["chunk_text"]
    metadata = state["metadata"]
    
    try:
        props = atomizer(chunk_text, metadata)
        print(f"   Decomposed into {len(props)} propositions.")
        return {"propositions": props}
    except Exception as e:
        return {"errors": [f"Atomizer Error: {str(e)}"]}

async def entity_extraction_node(state: GraphState) -> Dict[str, Any]:
    print("---ENTITY EXTRACTION (Context-Aware)---")
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
        # Extract returns List[FinancialRelation]
        relations = await asyncio.to_thread(entity_extractor.extract, prop, chunk_text, header_path)
        
        # Convert relations back to AtomicFact (preserving the original prop text)
        facts = []
        for rel in relations:
            # Create AtomicFact using the original 'prop' as the fact text
            # and the extracted components from the relation
            facts.append(AtomicFact(
                fact=prop,
                subject=rel.subject,
                subject_type=rel.subject_type,
                object=rel.object,
                object_type=rel.object_type,
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
        
        print(f"   Extracted {len(all_facts)} AtomicFacts from {len(propositions)} propositions.")
        return {"atomic_facts": all_facts}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"errors": [f"Entity Extraction Error: {str(e)}"]}

async def parallel_resolution_node(state: GraphState) -> Dict[str, Any]:
    print("---PARALLEL RESOLUTION---")
    facts = state["atomic_facts"]
    group_id = state.get("group_id", "default_tenant")
    
    # helper for resolving a single entity/topic

    # helper for resolving a single entity/topic
    async def resolve_item_async(name: str, context: str, node_type: str) -> tuple[str | None, str | None, str]:
        if not name:
             return None, None, ""

        # NORMALIZE: Enforce Title Case to prevent "Tariff-related" vs "Tariff-Related"
        name = name.strip().title()

        # 1. Generate Embedding for Lock (and downstream use)
        # We need to run this in thread pool because it might block? usually embed_query is blocking in some libs
        # But wait, services.embeddings is likely synchronos wrapper around API. 
        # Actually it's better to do this once.
        # Note: We use the name + context? No, just the name for locking similarity of nodes.
        # But we want to support "Apple" (fruit) vs "Apple" (company).
        # We should lock on the *Name* vector.
        try:
             name_embedding = await asyncio.to_thread(embeddings.embed_query, name)
        except Exception:
             name_embedding = None

        if name_embedding:
             # Accelerate: Wait for similar ops
             lock_id = await SimilarityLockManager.acquire_lock(name_embedding, name)
             try:
                 # Inside the lock, we do the full resolution (which checks DB updates from other threads)
                 return await asyncio.to_thread(resolve_single_item, name, context, node_type)
             finally:
                 await SimilarityLockManager.release_lock(lock_id)
        else:
             # Fallback if embedding fails
             return await asyncio.to_thread(resolve_single_item, name, context, node_type)

    def resolve_single_item(name: str, context: str, node_type: str) -> tuple[str | None, str | None, str]:
        if not name:
            return None, None, ""
        
        # 0. Extract Summary
        summary = enhancer.extract_entity_summary(name, context)
            
        # 1. FIBO Resolution
        res = librarian.resolve(name)
        if res:
            # Materialize FIBO node to get UUID
            fibo_uri = res['uri']
            fibo_label = res['label']
            
            target_label = "TopicNode" if node_type == "Topic" else "EntityNode"
            
            # We must ensure this node exists and has a UUID
            cypher_fibo = f"""
            MERGE (n:{target_label} {{name: $name, group_id: $group_id}})
            ON CREATE SET 
                n.uuid = $uuid,
                n.fibo_uri = $fibo_uri,
                n.summary = $summary,
                n.is_fibo = true,
                n.created_at = datetime()
            ON MATCH SET
                n.fibo_uri = $fibo_uri,
                n.is_fibo = true
            RETURN n.uuid as node_uuid
            """
            try:
                # Use a deterministic UUID for FIBO if we want? Or just random.
                # If we merge on name only, we might merge with existing local entity.
                # Ideally we merge on `fibo_uri` if it exists?
                # But current schema merges on `name` within `group_id`.
                # Let's stick to MERGE on name for now to avoid duplicates if name matches.
                
                fibo_node_res = neo4j_client.query(cypher_fibo, {
                    "name": fibo_label,
                    "group_id": group_id,
                    "uuid": str(uuid.uuid4()),
                    "fibo_uri": fibo_uri,
                    "summary": summary
                })
                if fibo_node_res:
                    return fibo_node_res[0]['node_uuid'], fibo_label, summary
            except Exception as e:
                print(f"FIBO materialization failed: {e}")
            
        # 2. Graph Deduplication (Semantic)
        candidates = enhancer.find_graph_candidates(name, summary, group_id=group_id, node_type=node_type)
        decision = enhancer.resolve_entity_against_graph(name, summary, candidates)
        
        if decision['decision'] == 'MERGE' and decision['target_uuid']:
            return decision['target_uuid'], name, summary
        
        # 3. New Entity - CREATE IMMEDIATELY (Atomic Write)
        
        # Generator Embedding (Name + Summary)
        embedding_text = f"{name}: {summary}"
        try:
            node_embedding = embeddings.embed_query(embedding_text)
        except Exception as e:
            print(f"   ⚠️ Embedding Generation Failed: {e}")
            node_embedding = None

        new_uuid = str(uuid.uuid4())
        print(f"   🆕 Creating New {node_type} (Atomic): {name}")
        
        label = "TopicNode" if node_type == "Topic" else "EntityNode"
        
        # Create with UUID, NO URI
        cypher_atomic_create = f"""
        MERGE (n:{label} {{name: $name, group_id: $group_id}})
        ON CREATE SET 
            n.uuid = $uuid, 
            n.summary = $summary,
            n.embedding = $embedding,
            n.is_fibo = false,
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
        fact_str = f"{fact.subject} {fact.fact} {fact.object if fact.object else ''}"
        return await asyncio.to_thread(analyst.classify_relationship, fact_str)

    # 1. Deduplicate Items to Resolve
    # Key: (name, type) -> List of fact indices/roles needing this resolution
    # We use (name, type) because "Apple" (Entity) and "Apple" (Topic/Fruit) should typically be distinct if types differ,
    # though our resolution logic might handle them similarly. Safest to respect the type from Atomizer.
    unique_items_to_resolve = {} # (name, type) -> { "context": first_context, "original_indices": [...] }
    
    for i, fact in enumerate(facts):
        # Subject
        subj_key = (fact.subject, fact.subject_type)
        if subj_key not in unique_items_to_resolve:
            unique_items_to_resolve[subj_key] = {"context": fact.fact, "original_indices": []}
        unique_items_to_resolve[subj_key]["original_indices"].append({"fact_idx": i, "role": "subject"})
        
        # Object
        if fact.object:
            obj_key = (fact.object, fact.object_type)
            if obj_key not in unique_items_to_resolve:
                unique_items_to_resolve[obj_key] = {"context": fact.fact, "original_indices": []}
            unique_items_to_resolve[obj_key]["original_indices"].append({"fact_idx": i, "role": "object"})
            
        # Topics (Always Type=Topic)
        if fact.topics:
            for topic in fact.topics:
                top_key = (topic, "Topic")
                if top_key not in unique_items_to_resolve:
                    unique_items_to_resolve[top_key] = {"context": fact.fact, "original_indices": []}
                unique_items_to_resolve[top_key]["original_indices"].append({"fact_idx": i, "role": "topic_list", "topic_name": topic})

    print(f"   Calculated {len(unique_items_to_resolve)} unique items to resolve from {len(facts)} facts.")

    # 2. Create Resolution Tasks
    resolution_tasks = []
    # We need to map task index back to the key to store results
    task_idx_to_key = {}
    
    sorted_keys = list(unique_items_to_resolve.keys())
    for idx, (name, node_type) in enumerate(sorted_keys):
        context = unique_items_to_resolve[(name, node_type)]["context"]
        resolution_tasks.append(resolve_item_async(name, context, node_type))
        task_idx_to_key[idx] = (name, node_type)
        
    # 3. Create Classification Tasks
    classification_tasks = [classify_fact_relationship(fact) for fact in facts]
    
    # 4. Run Everything in Parallel
    print(f"   🚀 Running {len(resolution_tasks)} resolution + {len(classification_tasks)} classification tasks...")
    all_results = await asyncio.gather(*(resolution_tasks + classification_tasks))
    
    resolution_results = all_results[:len(resolution_tasks)]
    classification_results = all_results[len(resolution_tasks):]
    
    # 5. Distribute Resolution Results back to Facts
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

    # Ensure Object fields are populated (None if no object)
    for i, fact in enumerate(facts):
        if not fact.object:
            resolved_entities[i]["object_uuid"] = None
            resolved_entities[i]["object_label"] = None
            resolved_entities[i]["object_summary"] = ""
            resolved_entities[i]["object_type"] = "Entity" # Default

    return {
        "resolved_entities": resolved_entities,
        "resolved_topics": resolved_topics_by_fact,
        "classified_relationships": classification_results
    }

def causal_linking_node(state: GraphState) -> Dict[str, Any]:
    print("---CAUSAL LINKING---")
    facts = state["atomic_facts"]
    chunk_text = state["chunk_text"]
    
    try:
        links = causal_linker.extract_causality(facts, chunk_text)
        print(f"   Found {len(links)} causal links.")
        return {"causal_links": links}
    except Exception as e:
        print(f"Causal linking error: {e}")
        return {"causal_links": []}

def assemble_node(state: GraphState) -> Dict[str, Any]:
    print("---ASSEMBLER---")
    facts = state["atomic_facts"]
    resolved_ents = state["resolved_entities"]
    resolved_tops = state.get("resolved_topics", [[] for _ in facts])
    links = state["causal_links"]
    episode_uuid = state["episodic_uuid"]
    group_id = state.get("group_id", "default_tenant")
    
    fact_uuids = []
    
    # 1. Create Fact Nodes
    for i in range(len(facts)):
        fact = facts[i]
        res_ent = resolved_ents[i]
        res_top = resolved_tops[i]
        rel = state["classified_relationships"][i] if "classified_relationships" in state and i < len(state["classified_relationships"]) else None
        
        # Retry logic for Rate Limiting
        max_retries = 3
        for attempt in range(max_retries):
            try:
                uuid = assembler.assemble_fact_node(
                    fact_obj=fact,
                    subject_uuid=res_ent["subject_uuid"],
                    subject_label=res_ent["subject_label"],
                    object_uuid=res_ent["object_uuid"],
                    object_label=res_ent["object_label"],
                    episode_uuid=episode_uuid,
                    group_id=group_id,
                    relationship_classification=rel,
                    subject_summary=res_ent.get("subject_summary", ""),
                    object_summary=res_ent.get("object_summary", ""),
                    subject_type=res_ent.get("subject_type", "Entity"),
                    object_type=res_ent.get("object_type", "Entity")
                )
                fact_uuids.append(uuid)
                
                # Link Topics
                for topic in res_top:
                    if topic["uuid"]: # Only link if we have a valid UUID
                        # Generate Topic Embedding
                        topic_embedding = None
                        try:
                            t_text = f"{topic['label']}: {topic['summary']}" if topic['summary'] else topic['label']
                            topic_embedding = embeddings.embed_query(t_text)
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
                        RETURN t.uuid
                        """
                        neo4j_client.query(cypher_topic_merge, {
                            "topic_uuid": topic["uuid"],
                            "group_id": group_id,
                            "name": topic["label"],
                            "summary": topic["summary"],
                            "embedding": topic_embedding
                        })
                        
                        # Link to Episode
                        assembler.link_topic_to_episode(topic["uuid"], episode_uuid, group_id)
                
                # If success, break retry loop
                break

            except Exception as e:
                # Check for Rate Limit
                if "requests per minute" in str(e).lower() and attempt < max_retries - 1:
                    print(f"   ⚠️ Rate Limit hit for fact {i} (Attempt {attempt+1}/{max_retries}). Waiting 10s...")
                    
                    ##### TEMP
                    import time
                    time.sleep(10)
                    ##### TEMP
                    
                    continue # Retry
                
                # If other error or max retries exceeded
                error_msg = f"Error assembling fact {i}: {e}"
                print(error_msg)
                if "errors" not in state:
                    state["errors"] = []
                state["errors"].append(error_msg)
                fact_uuids.append(None) # Keep index alignment
                break
        
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
workflow.add_node("causal_linking", causal_linking_node)
workflow.add_node("assembler", assemble_node)

workflow.set_entry_point("initialize_episode")

workflow.add_edge("initialize_episode", "atomizer")
workflow.add_edge("atomizer", "entity_extraction")
workflow.add_edge("entity_extraction", "parallel_resolution")
workflow.add_edge("parallel_resolution", "causal_linking")
workflow.add_edge("causal_linking", "assembler")
workflow.add_edge("assembler", END)

app = workflow.compile()
