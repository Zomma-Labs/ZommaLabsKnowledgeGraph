"""
State definitions for Deep Research KG-RAG.

Adapted from LangChain's Open Deep Research pattern.
"""

from typing import Annotated, TypedDict, Optional
from dataclasses import dataclass, field
from langgraph.graph import MessagesState
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


# === Reducer for State Updates ===

def override_or_append(existing: list, new_value) -> list:
    """Reducer that supports both append and override semantics."""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", [])
    if isinstance(new_value, list):
        return existing + new_value
    return existing + [new_value]


# === Pydantic Models for Structured Output ===

class ConductResearch(BaseModel):
    """Tool call to spawn a researcher for a specific topic."""
    research_topic: str = Field(description="Specific topic/question to research")
    search_hints: list[str] = Field(default_factory=list, description="Entities or terms to search for")


class ResearchComplete(BaseModel):
    """Signal that research is complete."""
    pass


class ResearchFinding(BaseModel):
    """Compressed finding from a researcher."""
    topic: str = Field(description="What was researched")
    finding: str = Field(description="Key finding/answer")
    confidence: float = Field(ge=0, le=1, description="Confidence in finding")
    evidence_chunks: list[str] = Field(default_factory=list, description="Chunk IDs supporting this")
    raw_content: str = Field(default="", description="Raw evidence text")


# === State Definitions ===

class SupervisorState(TypedDict):
    """State for the supervisor agent."""
    messages: Annotated[list[BaseMessage], override_or_append]
    research_brief: str  # Focused version of user question
    findings: Annotated[list[ResearchFinding], override_or_append]
    iterations: int
    max_iterations: int


class ResearcherState(TypedDict):
    """State for a researcher sub-agent."""
    messages: Annotated[list[BaseMessage], override_or_append]
    research_topic: str
    search_hints: list[str]
    iterations: int
    max_iterations: int
    collected_evidence: Annotated[list[str], override_or_append]  # Raw chunk content


class ResearcherOutput(BaseModel):
    """Output from a researcher subgraph."""
    finding: ResearchFinding


class AgentState(TypedDict):
    """Top-level agent state."""
    messages: Annotated[list[BaseMessage], override_or_append]
    original_question: str
    research_brief: str
    findings: Annotated[list[ResearchFinding], override_or_append]
    final_answer: str


# === Dataclass for Final Result ===

@dataclass
class DeepResearchResult:
    """Final result from the deep research pipeline."""
    question: str
    answer: str
    research_brief: str
    findings: list[ResearchFinding]
    planning_time_ms: int = 0
    research_time_ms: int = 0
    synthesis_time_ms: int = 0

    @property
    def total_time_ms(self) -> int:
        return self.planning_time_ms + self.research_time_ms + self.synthesis_time_ms
