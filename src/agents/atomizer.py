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
from typing import List, Optional
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
        "You are a financial expert breaking down text into atomic facts for a knowledge graph.\n\n"
        "Think like you're briefing a colleague: extract facts that capture the key information "
        "a financial analyst would need to understand what happened.\n\n"

        "=== CORE PRINCIPLES ===\n\n"
        "Each fact should:\n"
        "1. Be FOCUSED on one key piece of information\n"
        "2. Contain enough CONTEXT that a financial expert would understand the story\n"
        "3. CLARIFY vague references using details from the same chunk\n"
        "4. Be SELF-CONTAINED - readable without the original text\n\n"

        "=== STRICT RULES ===\n\n"
        "1. DE-CONTEXTUALIZE: Resolve all pronouns and vague references to specific names.\n"
        "   - 'he', 'she', 'it', 'they', 'the company' → actual names\n"
        "   - Frame events in terms of the real entities involved, not generic descriptions\n\n"

        "2. TEMPORAL GROUNDING: Make every fact standalone in time.\n"
        "   - 'last year' with doc date 2023 → '2022'\n"
        "   - Preserve exact dates: 'January 16, 2020' stays as 'January 16, 2020'\n"
        "   - Quarters: 'Q3 2023' stays as 'Q3 2023', not just '2023'\n\n"

        "3. EVENT DISAMBIGUATION: Create SEPARATE facts for multiple dated events.\n"
        "   - BAD: 'The company was founded in 2010 and went public in 2015'\n"
        "   - GOOD: Two facts with specific dates for each event\n\n"

        "4. ATOMICITY: Each fact = one simple sentence. Split compound sentences.\n\n"

        "5. COMPLETENESS: Each fact must be SELF-CONTAINED.\n"
        "   - The test: 'Would a financial analyst understand this without reading the source?'\n"
        "   - Keep WHO said something with WHAT they said\n"
        "   - BAD: 'The scientist made a finding' (incomplete)\n"
        "   - GOOD: 'Dr. Smith found that water boils at 100C under standard pressure'\n\n"

        "6. NUMERIC PRECISION: Preserve ALL numeric details exactly as written.\n\n"

        "7. LIST PRESERVATION: When text enumerates multiple items, preserve the complete list with all details.\n\n"

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


def atomizer_reflexion_check(chunk_text: str, facts: List[str], metadata: dict) -> Optional[str]:
    """
    Reviews extracted atomic facts and returns a critique if issues are found.

    Returns:
        None if all facts are valid (approved)
        A critique string describing what needs to be fixed
    """
    if not facts:
        return None

    llm = get_llm()
    facts_str = "\n".join([f"- {fact}" for fact in facts])

    critique_prompt = (
        f"You are a quality control agent reviewing atomic facts extracted from a financial document.\n\n"
        f"ORIGINAL CHUNK:\n{chunk_text}\n\n"
        f"METADATA: {metadata}\n\n"
        f"EXTRACTED FACTS:\n{facts_str}\n\n"

        f"=== YOUR TASK ===\n"
        f"Review each fact against the de-contextualization requirements:\n\n"

        f"1. PRONOUN RESOLUTION: Are there unresolved pronouns?\n"
        f"   - 'he', 'she', 'it', 'they', 'the company' should be replaced with actual names\n\n"

        f"2. TEMPORAL GROUNDING: Are temporal references grounded?\n"
        f"   - 'last year', 'recently', 'next quarter' should be resolved to specific dates/years\n\n"

        f"3. SELF-CONTAINED: Can each fact be understood without the original text?\n"
        f"   - Facts should not reference 'the above', 'this', 'that' without context\n\n"

        f"4. COMPLETENESS: Does each fact contain enough context?\n"
        f"   - If someone said something, include WHO said it and WHAT they said\n\n"

        f"=== RESPONSE FORMAT ===\n"
        f"If ALL facts are valid: respond with exactly 'APPROVED'\n\n"
        f"If ANY fact has issues: provide a specific critique explaining:\n"
        f"1. Which fact(s) are problematic\n"
        f"2. What the specific issue is\n"
        f"3. How the atomizer should fix it\n\n"
        f"Be specific so the atomizer can fix its output."
    )

    try:
        response = llm.invoke([("human", critique_prompt)])
        critique = response.content.strip()

        if critique.upper() == "APPROVED" or critique.lower().startswith("approved"):
            return None

        return critique

    except Exception as e:
        print(f"   ⚠️ Atomizer reflexion check failed: {e}")
        return None


def _atomize_with_critique(chunk_text: str, metadata: dict, critique: str, previous_facts: List[str]) -> List[str]:
    """
    Re-extracts atomic facts after receiving a critique from the reflexion agent.
    Has access to: original chunk, previous facts, and the critique.
    """
    llm = get_llm()
    structured_llm = llm.with_structured_output(PropositionList)

    prev_facts_str = "\n".join([f"- {fact}" for fact in previous_facts])

    system_prompt = (
        "You are a financial expert breaking down text into atomic facts for a knowledge graph.\n\n"
        "Think like you're briefing a colleague: extract facts that capture the key information "
        "a financial analyst would need to understand what happened.\n\n"

        "=== CORE PRINCIPLES ===\n\n"
        "Each fact should:\n"
        "1. Be FOCUSED on one key piece of information\n"
        "2. Contain enough CONTEXT that a financial expert would understand the story\n"
        "3. CLARIFY vague references using details from the same chunk\n"
        "4. Be SELF-CONTAINED - readable without the original text\n\n"

        "=== STRICT RULES ===\n\n"
        "1. DE-CONTEXTUALIZE: Resolve all pronouns and vague references to specific names.\n"
        "2. TEMPORAL GROUNDING: Make every fact standalone in time.\n"
        "3. COMPLETENESS: Each fact must be SELF-CONTAINED.\n\n"

        f"METADATA:\n{metadata}"
    )

    human_prompt = (
        f"ORIGINAL CHUNK:\n{chunk_text}\n\n"

        f"=== YOUR PREVIOUS EXTRACTION ===\n"
        f"{prev_facts_str}\n\n"

        f"=== CRITIQUE FROM REVIEWER ===\n"
        f"{critique}\n\n"

        f"Please re-extract the atomic facts, fixing the issues identified in the critique."
    )

    try:
        response = structured_llm.invoke([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
        return response.propositions if response.propositions else []
    except Exception as e:
        print(f"   ⚠️ Re-atomization with critique failed: {e}")
        return previous_facts  # Return original on failure


def atomizer_with_reflexion(chunk_text: str, metadata: dict, max_iterations: int = 2) -> List[str]:
    """
    Extracts atomic facts with a reflexion loop that critiques and refines the output.

    1. Extract initial facts
    2. Send to reflexion agent for critique
    3. If critique found, re-extract with critique feedback
    4. Return final facts
    """
    # Initial extraction
    facts = atomizer(chunk_text, metadata)

    if not facts:
        return []

    # Reflexion loop
    for iteration in range(max_iterations):
        critique = atomizer_reflexion_check(chunk_text, facts, metadata)

        if critique is None:
            # Approved - no issues found
            log(f"   ✅ Atomizer reflexion approved on iteration {iteration + 1}")
            break

        log(f"   🔄 Atomizer reflexion found issues, re-extracting...")

        # Re-extract with critique feedback
        facts = _atomize_with_critique(chunk_text, metadata, critique, facts)

        if not facts:
            break

    # Deduplicate
    unique_facts = list(set(facts))
    return unique_facts


if __name__ == "__main__":
    pass