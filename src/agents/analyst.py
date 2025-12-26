import json
import os
from typing import List, Optional, TYPE_CHECKING
from pydantic import BaseModel, Field

from src.util.vector_store import VectorStore
from src.schemas.relationship import RelationshipType, RelationshipDefinition, RelationshipClassification

if TYPE_CHECKING:
    from src.util.services import Services

# Control verbose output
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"

def log(msg: str):
    """Print only if VERBOSE mode is enabled."""
    if VERBOSE:
        print(msg)

class AnalystAgent:
    def __init__(self, services: Optional["Services"] = None):
        if services is None:
            from src.util.services import get_services
            services = get_services()
        self.vector_store = VectorStore(client=services.qdrant_relationships)
        self.llm = services.llm
        self.max_retries = 1

    def classify_relationship(self, fact: str) -> Optional[RelationshipClassification]:
        """
        Classifies the relationship in the given fact using a RAG approach.
        """
        
        # Step 1: Initial description generation (used as query)
        query = self._generate_initial_description(fact)
        
        for attempt in range(self.max_retries):
            log(f"Attempt {attempt + 1}: Querying with '{query}'")
            
            # Step 2: Retrieve candidates
            candidates = self.vector_store.search_relationships(query, limit=20)
            
            # Step 3: Select or Refine
            decision = self._select_or_refine(fact, query, candidates)
            
            if isinstance(decision, RelationshipClassification):
                return decision
            else:
                # Decision is a refined query string
                query = decision
                log(f"Refining query to: '{query}'")
        
        log("Max retries reached. Could not classify with high confidence.")
        return None

    def _generate_initial_description(self, fact: str) -> str:
        prompt = (
            f"Analyze the following fact and describe the relationship between the entities in detail. "
            f"Focus on the nature of the action (verb) and the type of interaction.\n"
            f"KEEP IT CONCISE (max 2 sentences).\n\n"
            f"Fact: {fact}\n\n"
            f"Description:"
        )
        response = self.llm.invoke(prompt)
        return response.content.strip()

    def _select_or_refine(self, fact: str, current_query: str, candidates: List[RelationshipDefinition]) -> RelationshipClassification | str:
        """
        Returns either a RelationshipClassification (if a match is found) or a string (new query).
        """
        
        candidates_text = ""
        for i, cand in enumerate(candidates):
            candidates_text += f"{i+1}. {cand.name}: {cand.description}\n"

        # Define a structured output for the decision
        class Decision(BaseModel):
            selected_relationship: Optional[RelationshipType] = Field(None, description="The selected relationship if a good match is found.")
            confidence: float = Field(..., description="Confidence score (0-1).")
            refined_query: Optional[str] = Field(None, description="A refined search query if no good match is found.")

        structured_llm = self.llm.with_structured_output(Decision)

        prompt = (
            f"You are an expert financial analyst. Your task is to classify the relationship in the given fact "
            f"into one of the provided candidate categories.\n\n"
            f"Fact: {fact}\n"
            f"Current Search Query: {current_query}\n\n"
            f"Candidates:\n{candidates_text}\n\n"
            f"Instructions:\n"
            f"1. Evaluate if any of the candidates accurately describe the relationship in the fact.\n"
            f"2. If a good match is found (confidence > 0.8), select it.\n"
            f"3. If NO good match is found, provide a 'refined_query' that might better retrieve the correct relationship. "
            f"The refined query should be a better description of the relationship."
        )

        result = structured_llm.invoke(prompt)

        if result.selected_relationship and result.confidence > 0.7: # Threshold
            return RelationshipClassification(
                relationship=result.selected_relationship,
                confidence=result.confidence
            )
        elif result.refined_query:
            return result.refined_query
        else:
            # Fallback if model doesn't select or refine (shouldn't happen with strict schema, but good to handle)
            return "No classification made." # Or some default behavior
