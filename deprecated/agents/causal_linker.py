from typing import List, Optional, TYPE_CHECKING
from pydantic import BaseModel, Field
from src.schemas.atomic_fact import AtomicFact

if TYPE_CHECKING:
    from src.util.services import Services

class CausalPair(BaseModel):
    cause_index: int = Field(description="Index of the cause fact in the list (0-based)")
    effect_index: int = Field(description="Index of the effect fact in the list (0-based)")
    reasoning: str = Field(description="Why A causes B")

class CausalList(BaseModel):
    links: List[CausalPair]

class CausalLinker:
    def __init__(self, services: Optional["Services"] = None):
        if services is None:
            from src.util.services import get_services
            services = get_services()
        self.llm = services.llm

    def extract_causality(self, facts: List[AtomicFact], text: str) -> List[CausalPair]:
        """
        Identifies causal relationships between the provided facts.
        Returns a list of pairs (cause_index, effect_index).
        """
        if len(facts) < 2:
            return []

        structured_llm = self.llm.with_structured_output(CausalList)
        
        facts_str = "\n".join([f"{i}. {fact.fact}" for i, fact in enumerate(facts)])
        
        system_prompt = (
            "You are a Causal Logic Expert.\n"
            "Identify CAUSAL relationships between the provided facts based on the source text.\n"
            "Return a list of pairs where Fact A CAUSES Fact B.\n"
            "Only return pairs where the causality is EXPLICIT or STRONGLY IMPLIED by the text.\n"
            "Do not force connections if none exist."
        )
        
        prompt = f"SOURCE TEXT:\n{text}\n\nFACTS:\n{facts_str}"
        
        try:
            response = structured_llm.invoke([
                ("system", system_prompt),
                ("human", prompt)
            ])
            return response.links
        except Exception as e:
            print(f"Causal extraction failed: {e}")
            return []

if __name__ == "__main__":
    pass
