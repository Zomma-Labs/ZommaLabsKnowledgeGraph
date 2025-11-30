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
from src.schemas.atomic_fact import AtomicFact
from src.schemas.relationship import RelationshipClassification
from src.tools.neo4j_client import Neo4jClient

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

# Init agents
librarian = FIBOLibrarian()
analyst = AnalystAgent()
assembler = GraphAssembler()
enhancer = GraphEnhancer()
causal_linker = CausalLinker()
neo4j_client = Neo4jClient()

def initialize_episode(state: GraphState) -> Dict[str, Any]:
    print("---INITIALIZE EPISODE---")
    chunk_text = state["chunk_text"]
    
    # Create EpisodicNode in Neo4j
    episode_uuid = str(uuid.uuid4())
    group_id = state.get("group_id")
    if not group_id:
        # Fallback or Error? For now, we'll try to get it from metadata or raise error
        group_id = state["metadata"].get("group_id", "default_tenant")
        print(f"⚠️ No group_id in state, using: {group_id}")
    
    # Extract Document Info
    doc_name = state["metadata"].get("filename") or state["metadata"].get("source_id") or "Unknown Document"
    file_type = state["metadata"].get("file_type", "text")
    
    # We use a single transaction to ensure atomicity
    cypher = """
    MERGE (d:DocumentNode {name: $doc_name, group_id: $group_id})
    ON CREATE SET 
        d.uuid = $doc_uuid,
        d.created_at = datetime(),
        d.document_date = datetime(),
        d.file_type = $file_type,
        d.metadata = $metadata
        
    MERGE (e:EpisodicNode {uuid: $episode_uuid, group_id: $group_id})
    ON CREATE SET 
        e.content = $content,
        e.source = 'text',
        e.created_at = datetime()
        
    MERGE (e)-[:BELONGS_TO]->(d)
    """
    
    # Generate a UUID for the document just in case it's new (used in ON CREATE)
    doc_uuid_candidate = str(uuid.uuid4())
    
    # Filter metadata to be JSON serializable if needed, or just pass it
    # Neo4j driver handles dicts as maps usually
    clean_metadata = {k: v for k, v in state["metadata"].items() if isinstance(v, (str, int, float, bool))}
    import json
    metadata_json = json.dumps(clean_metadata)

    neo4j_client.query(cypher, {
        "uuid": episode_uuid, # This was $uuid in the original query, but I changed it to $episode_uuid in my new query? No wait, I need to be careful with param names.
        "episode_uuid": episode_uuid,
        "content": chunk_text, 
        "group_id": group_id,
        "doc_name": doc_name,
        "doc_uuid": doc_uuid_candidate,
        "file_type": file_type,
        "metadata": metadata_json
    })
    print(f"   Created EpisodicNode: {episode_uuid} linked to Document: {doc_name}")
    
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
    
    # Wrapper for Librarian processing
    def resolve_fact_entities(fact: AtomicFact) -> Dict[str, Any]:
        
        def resolve_entity(name: str) -> tuple[str | None, str | None]:
            if not name:
                return None, None
                
            # 1. FIBO Resolution
            res = librarian.resolve(name)
            if res:
                return res['uri'], res['label']
                
            # 2. Graph Deduplication
            group_id = state.get("group_id", "default_tenant")
            candidates = enhancer.find_graph_candidates(name, group_id=group_id)
            decision = enhancer.resolve_entity_against_graph(name, candidates)
            
            if decision['decision'] == 'MERGE' and decision['target_uuid']:
                return decision['target_uuid'], name
            
            # 3. New Entity
            new_uuid = f"urn:uuid:{uuid.uuid4()}"
            print(f"   🆕 Creating New Entity: {name}")
            with open("new_entities.log", "a") as f:
                f.write(f"{name}\n")
            return new_uuid, name

        subj_uri, subj_label = resolve_entity(fact.subject)
        obj_uri, obj_label = resolve_entity(fact.object)
            
        return {
            "subject_uri": subj_uri, 
            "subject_label": subj_label,
            "object_uri": obj_uri,
            "object_label": obj_label
        }

    # Wrapper for Analyst processing
    async def classify_fact_relationship(fact: AtomicFact) -> RelationshipClassification:
        # Construct a string representation for the analyst
        fact_str = f"{fact.subject} {fact.fact} {fact.object if fact.object else ''}"
        return await asyncio.to_thread(analyst.classify_relationship, fact_str)

    # Run resolution in parallel
    async def resolve_wrapper(fact):
        return await asyncio.to_thread(resolve_fact_entities, fact)

    # Create tasks
    librarian_tasks = [resolve_wrapper(f) for f in facts]
    analyst_tasks = [classify_fact_relationship(f) for f in facts]
    
    # Gather all results
    results = await asyncio.gather(*librarian_tasks, *analyst_tasks)
    
    # Split results
    num_facts = len(facts)
    resolved_entities = results[:num_facts]
    classified_relationships = results[num_facts:]
    
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
                relationship_classification=rel
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
