"""
MODULE: Entity Extractor
SYSTEM: Financial-GraphRAG Ingestion Pipeline
AUTHOR: ZommaLabs
VERSION: 1.0.0

DESCRIPTION:
    This module defines the `EntityExtractor` agent. 
    It takes a pre-decomposed "Atomic Propostion" (fact) and the ORIGINAL chunk context,
    and extracts the structured entities (Subject, Object, Topics) involved.

    It uses the "Context-Aware" extraction pattern requested by the user:
    "RESOLVE, EXTRACT THE ENTITIES in this fact... FOR HELP here is the chunk..."

INPUT:
    - `fact_text`: The distinct atomic fact string.
    - `chunk_text`: The full text of the chunk where the fact originated.

OUTPUT:
    - `AtomicFact`: Fully populated object with Subject, Object, Topics.
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from src.schemas.atomic_fact import AtomicFact
from src.util.llm_client import get_llm

class EntityExtractor:
    def __init__(self):
        self.llm = get_llm()
        self.structured_llm = self.llm.with_structured_output(AtomicFact)

    def extract(self, fact_text: str, chunk_text: str) -> AtomicFact:
        """
        Extracts entities from a fact using the chunk context.
        """
        prompt = (
            f"RESOLVE, EXTRACT THE ENTITIES in this fact:\n"
            f"\"{fact_text}\"\n\n"
            
            f"FOR HELP here is the chunk in which it is a part of:\n"
            f"\"{chunk_text}\"\n\n"
            
            f"USE THIS INFORMATION to be specific to the entities in the fact at hand.\n\n"
            
            f"STRICT RULES:\n"
            f"1. KEY CONCEPTS: Extract high-level financial/economic concepts (e.g., 'Inflation', 'Rates').\n"
            f"2. SUBJECT/OBJECT IDENTIFICATION:\n"
            f"   - DIFFERENTIATE between ENTITIES and TOPICS.\n"
            f"   - 'Entity': Proper nouns, Companies, People (e.g., 'Apple').\n"
            f"   - 'Topic': General concepts, economic states (e.g., 'Inflation').\n"
            f"   - *CRITICAL*: If a concept performs an action (e.g. 'Inflation HURT earnings'), it is a SUBJECT with subject_type='Topic'.\n"
            f"3. GENERIC CONTAINERS: Ignore terms like 'Reports', 'Table', 'Data'.\n"
            f"4. FINANCIAL PRECISION & CANONICALIZATION:\n"
            f"   - Act as a Financial Expert.\n"
            f"   - CANONICALIZE TOPICS: 'Price Increases' -> 'Inflation', 'Level of Employment' -> 'Employment'.\n"
            f"   - If no specific object exists, leave it null.\n"
        )

        try:
            # We use invoke with a single string prompt because we are using structured output
            # which usually handles the system/human mapping under the hood or accepts a string.
            # But standard pattern is list of messages.
            # Let's use a simple system/human split for clarity, though the user's prompt is one block.
            # We will put the whole instruction in the system or human message?
            # User's prompt looks like a command. Let's put it in Human message to be safe with context.
            
            response = self.structured_llm.invoke([
                ("human", prompt)
            ])
            
            # The structured LLM returns an AtomicFact object directly (if configured correctly)
            # Depending on langchain version, it might verify fields.
            # We need to manually inject the 'fact' field back into the object if the LLM hallucinates it or modifies it?
            # Actually, AtomicFact has a 'fact' field. Ideally the LLM should just copy it back or we override it to be safe.
            if response.fact != fact_text:
                # Force the fact text to be the original proposition (to ensure exact match with upstream)
                # But wait, maybe the LLM *refined* the fact text too?
                # User's prompt says "RESOLVE... in this fact". It doesn't explicitly say "rewrite the fact".
                # But AtomicFact schema says "RESOLVE PRONOUNS...". 
                # Atomizer ALREADY did pronoun resolution.
                # So we should trust the input fact_text mostly.
                # Let's allow the LLM to return it, but generally we expect it to match.
                pass
                
            return response
            
        except Exception as e:
            print(f"Prop extraction failed for '{fact_text[:20]}...': {e}")
            # Fallback: Return a barebones AtomicFact
            return AtomicFact(
                fact=fact_text,
                subject="Unknown",
                subject_type="Entity"
            )

if __name__ == "__main__":
    pass
