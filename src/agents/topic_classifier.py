from enum import Enum
from pydantic import BaseModel, Field
from src.util.llm_client import get_llm

class HeaderType(str, Enum):
    TOPIC = "TOPIC"
    ENTITY = "ENTITY"

class ClassificationResult(BaseModel):
    header_type: HeaderType = Field(description="The classification of the header: TOPIC or ENTITY")
    reasoning: str = Field(description="Brief reasoning for the classification")

class TopicClassifier:
    def __init__(self):
        self.llm = get_llm()
        self.structured_llm = self.llm.with_structured_output(ClassificationResult)

    def classify(self, header_text: str) -> ClassificationResult:
        """
        Classifies a header string as either a TOPIC (Theme) or an ENTITY (Actor).
        """
        system_prompt = (
            "You are a Knowledge Graph Architect.\n"
            "Your task is to classify a document header as either a TOPIC (Theme/Concept) or an ENTITY (Actor/Organization/Person).\n\n"
            "Definitions:\n"
            "- TOPIC: A general theme, concept, or subject matter. Examples: 'Inflation', 'Labor Markets', 'Risk Factors', 'Financial Results', 'Overview'.\n"
            "- ENTITY: A specific actor, organization, person, or named place. Examples: 'Federal Reserve', 'Apple Inc.', 'Elon Musk', 'United States'.\n\n"
            "Rules:\n"
            "1. If the header contains BOTH (e.g., 'Apple's Revenue'), classify based on the SUBJECT. 'Apple's Revenue' is about Apple (ENTITY) performing/having something.\n"
            "   HOWEVER, if it's a generic section about a specific entity (e.g. 'Business Overview of Apple'), it might be better as an ENTITY node for 'Apple'.\n"
            "   Let's stick to: If it names a specific Actor, it's likely an ENTITY. If it names a Concept, it's a TOPIC.\n"
            "2. 'Management's Discussion' -> TOPIC (it's a section type).\n"
            "3. 'Consolidated Financial Statements' -> TOPIC.\n"
            "4. 'Board of Directors' -> TOPIC (group of people, but usually a section listing them). If it lists a specific person, that person is an ENTITY.\n"
        )

        try:
            return self.structured_llm.invoke([
                ("system", system_prompt),
                ("human", f"Header: {header_text}")
            ])
        except Exception as e:
            # Fallback to TOPIC if LLM fails
            return ClassificationResult(header_type=HeaderType.TOPIC, reasoning=f"Error: {e}")
