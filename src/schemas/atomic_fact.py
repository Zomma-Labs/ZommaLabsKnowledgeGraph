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
    subject_type: str = Field(
        "Entity",
        description="The type of the subject. 'Entity' for specific actors (Companies, People), 'Topic' for concepts acting as subjects (e.g. 'Inflation' hurt earnings)."
    )
    object: Optional[str] = Field(
        None,
        description=(
            "The entity or concept receiving the action. "
            "If a specific entity is involved (e.g., 'Microsoft'), use it. "
            "If no specific entity is found, extract the key Financial/Economic Concept (e.g., 'Revenue', 'Inflation')."
        )
    )
    object_type: str = Field(
        "Entity",
        description="The type of the object. 'Entity' for specific actors, 'Topic' for concepts."
    )
    relationship_description: Optional[str] = Field(
        None,
        description="A short phrase (2-5 words) describing the action between subject and object. Examples: 'acquired', 'partnered with', 'filed lawsuit against'."
    )
    date_context: Optional[str] = Field(
        None,
        description="Temporal context for the fact, e.g., 'Q3 2024', 'In 1963', '1962'. Used to ground facts in time, you should always resolve things like 'A year later' or 'Last year' to a specific year."
    )
    topics: Optional[List[str]] = Field(
        default_factory=list,
        description="Financial or Economic concepts mentioned or alluded to in the fact. Some examples: 'Inflation', 'Economic Activity', 'Interest Rates'."
    )

class AtomicFactList(BaseModel):
    atomic_facts: List[AtomicFact]