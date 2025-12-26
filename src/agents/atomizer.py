"""
MODULE: Atomizer
SYSTEM: Financial-GraphRAG Ingestion Pipeline
AUTHOR: ZommaLabs
VERSION: 2.0.0 (Decomposition Only)

DESCRIPTION:
    This module defines the `Atomizer` agent. Its SOLE responsibility is to 
    decompose large, unstructured text chunks into granular, self-contained 
    "Propositions" (Atomic Facts).

    It performs "De-contextualization":
    1.  Resolves Coreferences: Replaces "he", "it", "the company" with specific names.
    2.  Temporal Grounding: Calculates specific dates for relative time terms.
    3.  Atomicity: Splits compound sentences into simple statements.

INPUT:
    - `chunk_text`: The raw string from the document.
    - `metadata`: Dict containing `doc_date`, `chunk_id`, and `section_header`.

OUTPUT:
    - `List[str]`: A list of standalone fact strings (Propositions).

DEPENDENCIES:
    - pydantic
    - langchain
"""

import os
from typing import List
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from src.util.llm_client import get_llm

# Control verbose output
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"

def log(msg: str):
    """Print only if VERBOSE mode is enabled."""
    if VERBOSE:
        print(msg)

class PropositionList(BaseModel):
    propositions: List[str] = Field(description="List of atomic, de-contextualized fact strings.")

def atomizer(chunk_text: str, metadata: dict) -> List[str]:
    """
    Decomposes a text chunk into a list of atomic propositions (strings).
    """
    llm = get_llm()
    structured_llm = llm.with_structured_output(PropositionList)

    system_prompt = (
        "You are an expert Text Decomposer. Your goal is to split the input text "
        "into a list of 'Atomic Facts' (Propositions).\n\n"
        "Follow these strict rules:\n\n"
        "1. DE-CONTEXTUALIZE: The input is a chunk from a larger document. You must resolve all pronouns "
        "(he, she, it, they, the company) to their specific names based on context or the provided metadata.\n\n"
        "2. TEMPORAL GROUNDING: If the text says 'last year' and the document date is 2023, change it to '2022'. "
        "Make every fact standalone in time.\n\n"
        "3. EXACT DATE PRESERVATION: When the text contains specific dates, preserve them EXACTLY:\n"
        "   - Full dates: 'January 16, 2020' or 'September 1, 2017' must appear verbatim\n"
        "   - Quarters: 'Q3 2023' stays as 'Q3 2023', not just '2023'\n"
        "   - Month/Year: 'October 2020' stays as 'October 2020'\n"
        "   - NEVER approximate or round dates. '2020' is NOT the same as 'January 16, 2020'\n\n"
        "4. EVENT DISAMBIGUATION: When multiple dated events are mentioned, create SEPARATE facts for each:\n"
        "   - BAD: 'Alphabet was created in 2015 and Google became an LLC in 2017'\n"
        "   - GOOD: Two facts: 'Alphabet Inc. was created on October 2, 2015' AND 'Google was reorganized as an LLC on September 1, 2017'\n\n"
        "5. ATOMICITY: Each fact must be a single, simple sentence. Split compound sentences.\n\n"
        "6. PRESERVE DETAILS: Do not summarize away important information.\n\n"
        "7. NUMERIC PRECISION - CRITICAL: Preserve ALL numeric details exactly as written:\n"
        "   - Percentages: '7.25%', '6% of workforce', '0.34% equity stake'\n"
        "   - Dollar amounts: '$7.5 million', '$70 billion', '$245 million'\n"
        "   - Counts: '12,000 employees', '14,000 documents', '11 subsidiaries'\n"
        "   - Rankings: 'second-highest', 'fourth company to reach $3 trillion'\n"
        "   - Valuations: '$1 trillion market cap', '$2 trillion valuation'\n\n"
        "8. LIST PRESERVATION: When text lists multiple items, create a fact that preserves the full list:\n"
        "   - Input: 'Subsidiaries include Google, Waymo, Verily, and DeepMind'\n"
        "   - Output: 'Alphabet Inc. subsidiaries include Google, Waymo, Verily, and DeepMind'\n\n"
        f"METADATA:\n{metadata}"
    )

    response = structured_llm.invoke([
        ("system", system_prompt),
        ("human", chunk_text)
    ])
    
    facts = response.propositions

    # --- REFLEXION LOOP ---
    try:
        from src.util.services import get_services
        services = get_services()
        
        # Use shared enhancer via services instead of creating new instance
        from src.agents.graph_enhancer import GraphEnhancer
        enhancer = GraphEnhancer(services=services)
        
        # Check for missed facts / concept promotion
        # Note: Enhancer expects List[Any], so strings are fine
        missed_facts = enhancer.reflexion_check(chunk_text, facts)
        
        if missed_facts:
            log(f"   ✨ Reflexion found {len(missed_facts)} potential improvements.")
            
            # We need to structure these missed facts into PropositionList
            missed_facts_str = "\n".join(missed_facts)
            reflexion_prompt = (
                f"CONTEXT:\n{chunk_text}\n\n"
                f"The following facts were identified as missing or needing improvement.\n"
                f"Please structure them into Atomic Propositions:\n\n{missed_facts_str}"
            )
            
            reflexion_response = structured_llm.invoke([
                ("system", system_prompt),
                ("human", reflexion_prompt)
            ])
            
            if reflexion_response.propositions:
                facts.extend(reflexion_response.propositions)
                log(f"   ✅ Added {len(reflexion_response.propositions)} facts from Reflexion.")
                
    except Exception as e:
        print(f"   ⚠️ Reflexion step failed: {e}")

    # Deduplicate facts based on the string value
    unique_facts = list(set(facts))
    
    return unique_facts

if __name__ == "__main__":
    pass