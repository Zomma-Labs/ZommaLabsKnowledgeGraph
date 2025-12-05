"""
MODULE: Atomizer
SYSTEM: Financial-GraphRAG Ingestion Pipeline
AUTHOR: ZommaLabs
VERSION: 1.0.0 (Schema-on-Write Implementation)

DESCRIPTION:
    This module defines the `Atomizer` agent. Its primary responsibility is to 
    decompose large, unstructured text chunks (e.g., 10-K sections, Press Releases) into 
    granular, self-contained "Propositions" (Atomic Facts).

    Unlike simple chunking, this agent performs "De-contextualization":
    1.  Resolves Coreferences: Replaces "he", "it", "the company" with specific names 
        (e.g., "Warren Buffett", "Berkshire Hathaway").
    2.  Temporal Grounding: Calculates specific dates for relative time terms 
        (e.g., "A year later" -> "1963-01-01").
    3.  Entity Spotting: Identifies key nouns for downstream FIBO resolution.

INPUT:
    - `chunk_text`: The raw string from the document.
    - `metadata`: Dict containing `doc_date`, `chunk_id`, and `section_header`.

OUTPUT:
    - `List[AtomicFact]`: A list of structured objects ready for Graph Assembly.

DEPENDENCIES:
    - pydantic (Data Validation)
    - langchain (LLM Integration)
    - langgraph (StateGraph) 
"""

import os
from typing import List
from langchain_core.messages import BaseMessage
from src.schemas.atomic_fact import AtomicFact, AtomicFactList
from src.util.llm_client import get_llm

def atomizer(chunk_text: str, metadata: dict) -> List[AtomicFact]:
    """
    This function takes a chunk of text and metadata as input and returns a list of atomic facts.
    """
    llm = get_llm()
    structured_llm = llm.with_structured_output(AtomicFactList)

    system_prompt = (
        "You are an expert Knowledge Graph Engineer. Your goal is to decompose the input text "
        "into a list of 'Atomic Facts' (Propositions).\n\n"
        "Follow these strict rules:\n"
        "1. DE-CONTEXTUALIZE: The input is a chunk from a larger document. You must resolve all pronouns "
        "(he, she, it, they, the company) to their specific names based on context or the provided metadata.\n"
        "2. TEMPORAL GROUNDING: If the text says 'last year' and the document date is 2023, change it to '2022'. "
        "Make every fact standalone in time.\n"
        "3. ATOMICITY: Each fact must be a single, simple sentence. Split compound sentences.\n"
        "4. PRESERVE DETAILS: Do not summarize away important numbers, metrics, or specific adjectives.\n"
        "5. KEY CONCEPTS: Extract high-level financial/economic concepts (e.g., 'Inflation', 'Rates').\n"
        "6. SUBJECT/OBJECT IDENTIFICATION: Identify the 'subject' (who did it) and 'object' (who received it). "
        "If the object is not a specific entity, extract the key concept (e.g., 'Revenue') as the object.\n\n"
        f"METADATA:\n{metadata}"
    )

    response = structured_llm.invoke([
        ("system", system_prompt),
        ("human", chunk_text)
    ])
    
    facts = response.atomic_facts

    # --- REFLEXION LOOP ---
    try:
        from src.util.services import get_services
        services = get_services()
        
        # Use shared enhancer via services instead of creating new instance
        from src.agents.graph_enhancer import GraphEnhancer
        enhancer = GraphEnhancer(services=services)
        
        # Check for missed facts / concept promotion
        missed_facts = enhancer.reflexion_check(chunk_text, facts)
        
        if missed_facts:
            print(f"   ✨ Reflexion found {len(missed_facts)} potential improvements.")
            
            # We need to structure these missed facts into AtomicFact objects
            # We can reuse the same structured_llm and system_prompt, 
            # but we pass the missed facts as the "human" input to force extraction.
            missed_facts_str = "\n".join(missed_facts)
            reflexion_prompt = (
                f"CONTEXT:\n{chunk_text}\n\n"
                f"The following facts were identified as missing or needing improvement (Concept -> Entity promotion).\n"
                f"Please structure them into Atomic Facts:\n\n{missed_facts_str}"
            )
            
            reflexion_response = structured_llm.invoke([
                ("system", system_prompt),
                ("human", reflexion_prompt)
            ])
            
            if reflexion_response.atomic_facts:
                facts.extend(reflexion_response.atomic_facts)
                print(f"   ✅ Added {len(reflexion_response.atomic_facts)} facts from Reflexion.")
                
    except Exception as e:
        print(f"   ⚠️ Reflexion step failed: {e}")

    # Deduplicate facts based on the 'fact' string
    unique_facts = {}
    for f in facts:
        if f.fact not in unique_facts:
            unique_facts[f.fact] = f
    
    return list(unique_facts.values())

if __name__ == "__main__":
    pass