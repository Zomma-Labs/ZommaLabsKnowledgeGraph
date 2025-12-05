import asyncio
import uuid
from typing import List, Dict, Any, TypedDict
from langgraph.graph import StateGraph, END

from src.agents.atomizer import atomizer
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

# Define the State
class GraphState(TypedDict):
    chunk_text: str
    metadata: Dict[str, Any]
    group_id: str # Tenant ID
    episodic_uuid: str # New field for Provenance
    atomic_facts: List[AtomicFact]
    resolved_entities: List[Dict[str, Any]]
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
    
    # 2. Dimensional Star: Squash Hierarchy -> Context Hub
    headings = state["metadata"].get("headings", [])
    if not headings:
        headings = state["metadata"].get("breadcrumbs", [])
    
    # Filter out "Body" and strip whitespace
    headings = [h.strip() for h in headings if h.strip().lower() != "body"]
    
    # Define the Hub (SectionNode)
    # If no headings, the Document itself is the hub (or we create a generic one)
    # But typically we want a SectionNode even for top-level content
    header_path = " > ".join(headings) if headings else "ROOT"
    
    # Embed the header path
    try:
        hub_embedding = embeddings.embed_query(header_path)
    except Exception:
        hub_embedding = None
        
    hub_uuid_candidate = str(uuid.uuid4())
    
    # Create the Hub (SectionNode)
    cypher_hub = """
    MERGE (s:SectionNode {header_path: $header_path, doc_id: $doc_id, group_id: $group_id})
    ON CREATE SET
        s.uuid = $hub_uuid,
        s.embedding = $embedding,
        s.created_at = datetime()
    RETURN s.uuid as hub_uuid
    """
    
    hub_result = neo4j_client.query(cypher_hub, {
        "header_path": header_path,
        "doc_id": doc_name,
        "group_id": group_id,
        "hub_uuid": hub_uuid_candidate,
        "embedding": hub_embedding
    })
    
    hub_uuid = hub_result[0]['hub_uuid']
    
    # Link Hub -> Document
    cypher_link_doc = """
    MATCH (s:SectionNode {uuid: $hub_uuid})
    MATCH (d:DocumentNode {uuid: $doc_uuid})
    MERGE (s)-[:PART_OF]->(d)
    MERGE (d)-[:TALKS_ABOUT]->(s)
    """
    neo4j_client.query(cypher_link_doc, {"hub_uuid": hub_uuid, "doc_uuid": doc_uuid})
    
    # 3. Analyze Dimensions & Link to Hub
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
            # Use description to aid resolution
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
                    print(f"     🔄 Found existing {target_label}: {dim.value}")
                    resolved_node_uuid = existing[0]['node_uuid']
                    resolved_node_type = target_label
                else:
                    print(f"     🆕 Creating new {target_label}: {dim.value}")
                    with open("new_topics_entities.log", "a") as f:
                        f.write(f"[{target_label}] {dim.value}\n")
                    
                    try:
                        text_to_embed = f"{dim.value}: {dim.description}" if dim.description else dim.value
                        dim_embedding = embeddings.embed_query(text_to_embed)
                    except:
                        dim_embedding = None
                        
                    cypher_create_new = f"""
                    MERGE (n:{target_label} {{name: $name, group_id: $group_id}})
                    ON CREATE SET n.uuid = $uuid, n.embedding = $embedding, n.created_at = datetime()
                    RETURN n.uuid as node_uuid
                    """
                    res = neo4j_client.query(cypher_create_new, {
                        "name": dim.value,
                        "group_id": group_id,
                        "uuid": str(uuid.uuid4()),
                        "embedding": dim_embedding
                    })
                    resolved_node_uuid = res[0]['node_uuid']
                    resolved_node_type = target_label
            
            # Link Hub <-> Dimension (Bi-directional)
            if resolved_node_uuid:
                rel_to_dim = "DISCUSSES" if dim.type == DimensionType.TOPIC else "REPRESENTS"
                rel_from_dim = "IS_DISCUSSED_IN" if dim.type == DimensionType.TOPIC else "IS_REPRESENTED_IN"
                
                cypher_link_star = f"""
                MATCH (hub:SectionNode {{uuid: $hub_uuid}})
                MATCH (dim:{resolved_node_type} {{uuid: $dim_uuid}})
                MERGE (hub)-[:{rel_to_dim}]->(dim)
                MERGE (dim)-[:{rel_from_dim}]->(hub)
                """
                neo4j_client.query(cypher_link_star, {
                    "hub_uuid": hub_uuid,
                    "dim_uuid": resolved_node_uuid
                })

    # 4. Create EpisodicNode and Link to Hub
    cypher_episode = """
    MERGE (e:EpisodicNode {uuid: $episode_uuid, group_id: $group_id})
    ON CREATE SET 
        e.content = $content,
        e.source = 'text',
        e.created_at = datetime()
        
    WITH e
    MATCH (hub:SectionNode {uuid: $hub_uuid})
    MERGE (hub)-[:CONTAINS]->(e)
    """

    neo4j_client.query(cypher_episode, {
        "episode_uuid": episode_uuid,
        "group_id": group_id,
        "content": chunk_text,
        "hub_uuid": hub_uuid
    })
    
    print(f"   Created EpisodicNode: {episode_uuid} linked to Hub ({header_path})")
    
    return {"episodic_uuid": episode_uuid, "group_id": group_id}

def atomize_node(state: GraphState) -> Dict[str, Any]:
    print("---ATOMIZER---")
    chunk_text = state["chunk_text"]
    metadata = state["metadata"]
    
    try:
        facts = atomizer(chunk_text, metadata)
        return {"atomic_facts": facts}
    except Exception as e:
        return {"errors": [f"Atomizer Error: {str(e)}"]}

async def parallel_resolution_node(state: GraphState) -> Dict[str, Any]:
    print("---PARALLEL RESOLUTION---")
    facts = state["atomic_facts"]
    group_id = state.get("group_id", "default_tenant")
    
    # Single entity resolution function (blocking, will be wrapped in asyncio.to_thread)
    def resolve_single_entity(name: str, context: str) -> tuple[str | None, str | None, str]:
        if not name:
            return None, None, ""
        
        # 0. Extract Description
        description = enhancer.extract_entity_description(name, context)
            
        # 1. FIBO Resolution
        res = librarian.resolve(name)
        if res:
            return res['uri'], res['label'], description
            
        # 2. Graph Deduplication
        candidates = enhancer.find_graph_candidates(name, description, group_id=group_id)
        decision = enhancer.resolve_entity_against_graph(name, description, candidates)
        
        if decision['decision'] == 'MERGE' and decision['target_uuid']:
            return decision['target_uuid'], name, description
        
        # 3. New Entity
        new_uuid = f"urn:uuid:{uuid.uuid4()}"
        print(f"   🆕 Creating New Entity: {name} ({description})")
        return new_uuid, name, description

    # Async wrapper for a single entity
    async def resolve_entity_async(name: str, context: str) -> tuple[str | None, str | None, str]:
        return await asyncio.to_thread(resolve_single_entity, name, context)

    # Wrapper for Analyst processing
    async def classify_fact_relationship(fact: AtomicFact) -> RelationshipClassification:
        fact_str = f"{fact.subject} {fact.fact} {fact.object if fact.object else ''}"
        return await asyncio.to_thread(analyst.classify_relationship, fact_str)

    # Build ALL tasks: subjects, objects, and relationships in one batch
    # This maximizes parallelism by resolving everything at once
    
    entity_resolution_tasks = []
    analyst_tasks = []
    
    for fact in facts:
        # Create tasks for subject and object resolution (parallel within each fact)
        entity_resolution_tasks.append(resolve_entity_async(fact.subject, fact.fact))
        entity_resolution_tasks.append(resolve_entity_async(fact.object, fact.fact))
        # Create task for relationship classification
        analyst_tasks.append(classify_fact_relationship(fact))
    
    # Run ALL tasks in parallel (entity resolutions + relationship classifications)
    all_tasks = entity_resolution_tasks + analyst_tasks
    print(f"   🚀 Running {len(entity_resolution_tasks)} entity resolutions + {len(analyst_tasks)} classifications in parallel...")
    
    results = await asyncio.gather(*all_tasks)
    
    # Split results
    num_entity_tasks = len(entity_resolution_tasks)
    entity_results = results[:num_entity_tasks]
    classified_relationships = results[num_entity_tasks:]
    
    # Reconstruct resolved_entities from paired subject/object results
    resolved_entities = []
    for i in range(len(facts)):
        subj_idx = i * 2
        obj_idx = i * 2 + 1
        
        subj_uri, subj_label, subj_desc = entity_results[subj_idx]
        obj_uri, obj_label, obj_desc = entity_results[obj_idx]
        
        resolved_entities.append({
            "subject_uri": subj_uri, 
            "subject_label": subj_label,
            "subject_description": subj_desc,
            "object_uri": obj_uri,
            "object_label": obj_label,
            "object_description": obj_desc
        })
    
    return {
        "resolved_entities": resolved_entities,
        "classified_relationships": classified_relationships
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
    resolved = state["resolved_entities"]
    links = state["causal_links"]
    episode_uuid = state["episodic_uuid"]
    group_id = state.get("group_id", "default_tenant")
    
    fact_uuids = []
    
    # 1. Create Fact Nodes
    for i in range(len(facts)):
        fact = facts[i]
        res = resolved[i]
        rel = state["classified_relationships"][i] if "classified_relationships" in state and i < len(state["classified_relationships"]) else None
        
        try:
            uuid = assembler.assemble_fact_node(
                fact_obj=fact,
                subject_uri=res["subject_uri"],
                subject_label=res["subject_label"],
                object_uri=res["object_uri"],
                object_label=res["object_label"],
                episode_uuid=episode_uuid,
                group_id=group_id,
                relationship_classification=rel,
                subject_description=res.get("subject_description", ""),
                object_description=res.get("object_description", "")
            )
            fact_uuids.append(uuid)
        except Exception as e:
            error_msg = f"Error assembling fact {i}: {e}"
            print(error_msg)
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append(error_msg)
            fact_uuids.append(None) # Keep index alignment
        
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
workflow.add_node("parallel_resolution", parallel_resolution_node)
workflow.add_node("causal_linking", causal_linking_node)
workflow.add_node("assembler", assemble_node)

workflow.set_entry_point("initialize_episode")

workflow.add_edge("initialize_episode", "atomizer")
workflow.add_edge("atomizer", "parallel_resolution")
workflow.add_edge("parallel_resolution", "causal_linking")
workflow.add_edge("causal_linking", "assembler")
workflow.add_edge("assembler", END)

app = workflow.compile()
