"""
Entity Drill-Down for ENUMERATION Questions.

For enumeration questions, allows the agent to select specific entities
from the unique entities list to fetch additional facts for complete coverage.
"""

import os
from pydantic import BaseModel, Field

from .schemas import ScoredFact, QueryDecomposition, QuestionType
from src.util.llm_client import get_critique_llm

VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"


def log(msg: str):
    if VERBOSE:
        print(f"[EntityDrillDown] {msg}")


class EntitySelection(BaseModel):
    """LLM output for entity drill-down selection."""
    entities_to_explore: list[str] = Field(
        default_factory=list,
        description="Entity names to fetch additional facts for"
    )
    reasoning: str = Field(
        default="",
        description="Brief reasoning for selection"
    )


DRILLDOWN_PROMPT = """You are analyzing an ENUMERATION question that requires listing specific items.

QUESTION: {question}

REQUIRED INFORMATION:
{required_info}

UNIQUE ENTITIES FOUND (connected to search nodes):
{entities_list}

CURRENT FACTS RETRIEVED ({fact_count} facts):
{facts_summary}

YOUR TASK:
Looking at the unique entities list, identify which entities might have relevant facts
that would help complete the enumeration. Select entities that:
1. Seem relevant to the question but may not have enough facts in current results
2. Could help complete the enumeration (e.g., if question asks "which districts" and
   you see district names in the list)

Select ANY NUMBER of entities you feel are relevant - there is no limit.
Only select entities that appear in the UNIQUE ENTITIES FOUND list above.

If current facts seem sufficient, return an empty list.
"""


class EntityDrillDown:
    """
    Selects entities for additional fact retrieval in ENUMERATION questions.

    Flow:
    1. Show unique entities and current facts summary to gpt-5.1
    2. Agent selects entities needing more facts
    3. Pipeline fetches additional facts for those entities
    """

    def __init__(self, llm=None):
        if llm is None:
            llm = get_critique_llm()  # gpt-5.1 for quality decisions

        self.llm = llm
        self.structured_selector = llm.with_structured_output(
            EntitySelection, include_raw=True
        )

    def select_entities(
        self,
        question: str,
        decomposition: QueryDecomposition,
        unique_entities_by_node: dict[str, list[str]],
        current_facts: list[ScoredFact],
    ) -> list[str]:
        """
        Select entities that need additional fact retrieval.

        Only runs for ENUMERATION questions.

        Returns:
            List of entity names to fetch more facts for (no limit)
        """
        if decomposition.question_type != QuestionType.ENUMERATION:
            return []

        if not unique_entities_by_node:
            log("No unique entities to drill down on")
            return []

        # Flatten unique entities
        all_entities = []
        for node_name, entities in unique_entities_by_node.items():
            for entity in entities:
                if entity not in all_entities:
                    all_entities.append(entity)

        if not all_entities:
            return []

        log(f"Analyzing {len(all_entities)} unique entities for drill-down...")

        # Format entities list
        entities_list = "\n".join(f"  - {e}" for e in all_entities)

        # Summarize current facts (show entities mentioned)
        fact_entities = set()
        for fact in current_facts[:50]:
            if fact.subject:
                fact_entities.add(fact.subject)
            if fact.object:
                fact_entities.add(fact.object)

        facts_summary = f"Entities in current facts: {', '.join(list(fact_entities)[:20])}"

        prompt = DRILLDOWN_PROMPT.format(
            question=question,
            required_info="\n".join(f"- {info}" for info in decomposition.required_info),
            entities_list=entities_list,
            fact_count=len(current_facts),
            facts_summary=facts_summary,
        )

        try:
            response = self.structured_selector.invoke([("human", prompt)])

            if response.get("parsing_error") or response.get("parsed") is None:
                log("Drill-down parsing error")
                return []

            result = response["parsed"]

            # Validate selected entities exist in our list
            valid_entities = [
                e for e in result.entities_to_explore
                if e in all_entities
            ]

            if valid_entities:
                log(f"Selected {len(valid_entities)} entities: {valid_entities}")
                log(f"Reasoning: {result.reasoning}")
            else:
                log("No entities selected for drill-down")

            return valid_entities

        except Exception as e:
            log(f"Drill-down error: {e}")
            return []
