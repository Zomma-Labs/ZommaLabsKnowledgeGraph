import asyncio
from typing import List, Dict, Any, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage
import operator

from src.agents.atomizer import atomizer
from src.agents.FIBO_librarian import FIBOLibrarian
from src.agents.analyst import AnalystAgent
from src.agents.graph_assembler import GraphAssembler
from src.schemas.atomic_fact import AtomicFact
from src.schemas.relationship import RelationshipClassification

# Define the State
class GraphState(TypedDict):
    chunk_text: str
    metadata: Dict[str, Any]
    atomic_facts: List[AtomicFact]
    resolved_entities: List[Dict[str, Any]] # Aligned with atomic_facts
    classified_relationships: List[RelationshipClassification] # Aligned with atomic_facts
    errors: List[str]

# Initialize Agents (Global or per-node? Better to init once if they have state/connections)
# NOTE: In a real prod env, we might want to dependency inject these or init inside nodes if they are stateless.
# Librarian has a Qdrant client, Analyst has a VectorStore, Assembler has Neo4j.
# We'll init them lazily or globally. Let's init inside the node or a setup phase to avoid connection issues at import time?
# For now, we'll instantiate them at the module level but be careful about connections.
# Actually, let's instantiate them inside the nodes to be safe, or pass them in. 
# But `atomizer` is a function. The others are classes.

# Let's use a helper to get instances to avoid re-creating heavy clients every time if possible, 
# but for this script, re-creation per run might be okay or we can use singletons.
# The existing code for agents creates clients in __init__.

# Init agents (they will use their own separate Qdrant paths)
librarian = FIBOLibrarian()
analyst = AnalystAgent()
assembler = GraphAssembler()

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
    print("---PARALLEL RESOLUTION (SPLIT BRAIN)---")
    facts = state["atomic_facts"]
    
    # We need to process EACH fact.
    # We will create tasks for all facts for both agents.
    
    # Wrapper for Librarian processing a single fact's subject/object
    def resolve_fact_entities(fact: AtomicFact) -> Dict[str, Any]:
        # Resolve Subject
        subj_res = librarian.resolve(fact.subject)
        subj_uri = subj_res['uri'] if subj_res else None
        subj_label = subj_res['label'] if subj_res else None
        
        # Resolve Object (if it exists and is not just a concept, but Librarian tries anyway)
        obj_uri = None
        obj_label = None
        if fact.object:
            obj_res = librarian.resolve(fact.object)
            obj_uri = obj_res['uri'] if obj_res else None
            obj_label = obj_res['label'] if obj_res else None
            
        return {
            "subject_uri": subj_uri, 
            "subject_label": subj_label,
            "object_uri": obj_uri,
            "object_label": obj_label
        }

    # Wrapper for Analyst processing a single fact
    async def classify_fact_relationship(fact: AtomicFact) -> RelationshipClassification:
        # Note: classify_relationship is synchronous in the current AnalystAgent implementation.
        # We can run it in a thread pool to make it async-compatible if needed, 
        # or just call it if it was async. 
        # Since it calls LLMs (network bound), it *should* be async, but the code provided shows synchronous `invoke`.
        # To get true parallelism with sync code, we need `asyncio.to_thread`.
        
        # Construct a string representation for the analyst
        fact_str = f"{fact.subject} {fact.fact} {fact.object if fact.object else ''}"
        return await asyncio.to_thread(analyst.classify_relationship, fact_str)

    # Prepare tasks
    # We want to run Librarian and Analyst for ALL facts in parallel.
    
    # Librarian Tasks (one per fact)
    # Note: Librarian.resolve is also synchronous in the provided code (uses Qdrant sync client).
    # So we use to_thread there too.
    
    async def resolve_wrapper(fact):
        return await asyncio.to_thread(resolve_fact_entities, fact)

    librarian_tasks = [resolve_wrapper(f) for f in facts]
    analyst_tasks = [classify_fact_relationship(f) for f in facts]
    
    # Run everything concurrently!
    # We gather all librarian results and all analyst results.
    results = await asyncio.gather(*librarian_tasks, *analyst_tasks)
    
    # Split results back
    num_facts = len(facts)
    resolved_entities = results[:num_facts]
    classified_relationships = results[num_facts:]
    
    return {
        "resolved_entities": resolved_entities,
        "classified_relationships": classified_relationships
    }

def assemble_node(state: GraphState) -> Dict[str, Any]:
    print("---ASSEMBLER---")
    facts = state["atomic_facts"]
    resolved = state["resolved_entities"]
    classified = state["classified_relationships"]
    
    success_count = 0
    
    for i in range(len(facts)):
        fact = facts[i]
        res = resolved[i]
        rel = classified[i]
        
        if rel: # Only write if we have a relationship classification
            try:
                assembler.assemble_and_write(
                    fact_obj=fact,
                    subject_uri=res["subject_uri"],
                    subject_label=res["subject_label"],
                    object_uri=res["object_uri"],
                    object_label=res["object_label"],
                    relationship=rel
                )
                success_count += 1
            except Exception as e:
                print(f"Error assembling fact {i}: {e}")
        else:
            print(f"Skipping fact {i}: No relationship classified.")

    return {} # No state update needed, side effect only

# Build the Graph
workflow = StateGraph(GraphState)

workflow.add_node("atomizer", atomize_node)
workflow.add_node("parallel_resolution", parallel_resolution_node)
workflow.add_node("assembler", assemble_node)

workflow.set_entry_point("atomizer")

workflow.add_edge("atomizer", "parallel_resolution")
workflow.add_edge("parallel_resolution", "assembler")
workflow.add_edge("assembler", END)

app = workflow.compile()
