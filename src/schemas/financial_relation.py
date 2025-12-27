from pydantic import BaseModel, Field
from typing import List, Optional

class FinancialRelation(BaseModel):
    thinking: Optional[str] = Field(
        None,
        description="Your reasoning about whether these are valid, searchable entities that a financial analyst would research."
    )
    subject: str = Field(
        ...,
        description="The entity performing the action (e.g., 'Apple', 'The Federal Reserve')."
    )
    subject_type: str = Field(
        "Entity",
        description="The type of the subject. 'Entity' for specific actors (Companies, People), 'Topic' for concepts acting as subjects (e.g. 'Inflation' hurt earnings)."
    )
    subject_summary: Optional[str] = Field(
        None,
        description=(
            "A 1-2 sentence definition of what the subject IS based on context. "
            "For people: role/title, organization. For companies: industry, what they do. "
            "For topics: what this concept means in financial terms."
        )
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
    object_summary: Optional[str] = Field(
        None,
        description=(
            "A 1-2 sentence definition of what the object IS based on context. "
            "For people: role/title, organization. For companies: industry, what they do. "
            "For topics: what this concept means in financial terms."
        )
    )
    relationship_description: str = Field(
        ...,
        description=(
            "A concise phrase describing the ACTION between subject and object. "
            "Focus on the VERB, not the full sentence. Keep it to 2-5 words. "
            "Examples: 'acquired', 'partnered with', 'filed lawsuit against', "
            "'reported earnings of', 'appointed as CEO', 'expanded operations to'."
        )
    )
    date_context: Optional[str] = Field(
        None,
        description="Specific timeframe for this relationship. Preserve EXACT dates: 'January 16, 2020', 'September 1, 2017', 'Q3 2023', 'October 2020'."
    )
    topics: Optional[List[str]] = Field(
        default_factory=list,
        description="Financial or Economic concepts mentioned or alluded to in the fact. Some examples: 'Inflation', 'Economic Activity', 'Interest Rates'."
    )

class FinancialRelationList(BaseModel):
    relations: List[FinancialRelation]
