from pydantic import BaseModel, Field
from typing import List, Optional

class AtomicFact(BaseModel):
    fact: str = Field(
        ..., 
        description=(
            "A standalone sentence that captures a single event or relationship. "
            "RULES:\n"
            "1. RESOLVE PRONOUNS: Replace 'he', 'it', 'they' with specific names (e.g., 'Buffett').\n"
            "2. RESOLVE TIME: If specific dates are known, use them. Change 'A year later' to 'In 1963'.\n"
            "3. PRESERVE METRICS: Do not round numbers. Keep '$11.375' exact.\n"
            "4. PRESERVE NUANCE: Keep descriptive adjectives like 'undercutting' or 'declining' as they appear in text."
        )
    )
    subject: str = Field(
        ...,
        description="The entity performing the action (e.g., 'Apple', 'The Federal Reserve')."
    )
    object: Optional[str] = Field(
        None,
        description=(
            "The entity or concept receiving the action. "
            "If a specific entity is involved (e.g., 'Microsoft'), use it. "
            "If no specific entity is found, extract the key Financial/Economic Concept (e.g., 'Revenue', 'Inflation')."
        )
    )
    date_context: Optional[str] = Field(
        None,
        description="List of all distinct facts found. Order them chronologically if possible."
    )
    key_concepts: Optional[List[str]] = Field(
        default_factory=list,
        description="Financial or Economic concepts mentioned or alluded to in the fact. Some examples: 'Inflation', 'Economic Activity', 'Interest Rates'."
    )

class AtomicFactList(BaseModel):
    atomic_facts: List[AtomicFact]