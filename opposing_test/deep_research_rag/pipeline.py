"""
Deep Research Pipeline with RAG backend.
"""

import asyncio
import time
from dataclasses import dataclass
from langchain_core.messages import SystemMessage, HumanMessage

from .supervisor import run_supervisor
from .researcher import ResearchFinding
from src.util.llm_client import get_llm


@dataclass
class DeepResearchResult:
    question: str
    answer: str
    research_brief: str
    findings: list[ResearchFinding]
    research_time_ms: int
    synthesis_time_ms: int

    @property
    def total_time_ms(self) -> int:
        return self.research_time_ms + self.synthesis_time_ms


SYNTHESIZER_PROMPT = """You are synthesizing research findings into a direct answer.

## Core Principles

**1. Only use information explicitly in the findings**
- Do not add background knowledge, speculation, or general explanations
- If information is not in the findings, do not include it

**2. Preserve attribution for every fact**
- Information belongs to specific sources, entities, time periods, or contexts
- Every fact must be tied to its specific source or scope
- Do not generalize source-specific findings into broad statements

**3. Be concise**
- Answer the question directly
- No unnecessary preambles or over-explanation

**4. Match your answer's specificity to the question**
- Specific questions require specific answers
- If asked to enumerate or identify, provide the specific items found"""


class DeepResearchRAGPipeline:
    """Deep research pipeline using simple RAG backend."""

    def __init__(
        self,
        max_supervisor_iterations: int = 3,
        max_concurrent_researchers: int = 8,
    ):
        self.max_supervisor_iterations = max_supervisor_iterations
        self.max_concurrent_researchers = max_concurrent_researchers
        self.llm = get_llm()

    async def query_async(
        self,
        question: str,
        verbose: bool = False
    ) -> DeepResearchResult:
        """Answer a question using deep research with RAG backend."""

        if verbose:
            print(f"\n{'='*60}")
            print(f"DEEP RESEARCH (RAG): {question}")
            print(f"{'='*60}\n")

        # Phase 1: Research
        if verbose:
            print(">>> PHASE 1: Research")

        research_start = time.time()
        findings, research_brief, _ = await run_supervisor(
            question=question,
            max_iterations=self.max_supervisor_iterations,
            max_concurrent=self.max_concurrent_researchers
        )
        research_time = int((time.time() - research_start) * 1000)

        if verbose:
            print(f"    Brief: {research_brief[:100]}...")
            print(f"    Findings: {len(findings)}")
            print(f"    Time: {research_time}ms\n")

        # Phase 2: Synthesize
        if verbose:
            print(">>> PHASE 2: Synthesizing")

        synthesis_start = time.time()

        findings_text = ""
        if findings:
            for i, f in enumerate(findings, 1):
                findings_text += f"""
### Finding {i}: {f.topic}
**Result:** {f.finding}
**Confidence:** {f.confidence}
**Raw Evidence:**
{f.raw_content if f.raw_content else 'No raw evidence'}

---
"""
        else:
            findings_text = "No research findings available."

        prompt = f"""Original Question: {question}

Research Brief: {research_brief}

Research Findings:
{findings_text}

Based on the research findings above, provide an answer to the original question."""

        response = await asyncio.to_thread(
            self.llm.invoke,
            [
                SystemMessage(content=SYNTHESIZER_PROMPT),
                HumanMessage(content=prompt)
            ]
        )

        synthesis_time = int((time.time() - synthesis_start) * 1000)

        if verbose:
            print(f"    Time: {synthesis_time}ms\n")
            print(f"{'='*60}")
            print("ANSWER:")
            print(response.content)

        return DeepResearchResult(
            question=question,
            answer=response.content,
            research_brief=research_brief,
            findings=findings,
            research_time_ms=research_time,
            synthesis_time_ms=synthesis_time
        )

    def query(self, question: str, verbose: bool = False) -> DeepResearchResult:
        """Synchronous wrapper."""
        return asyncio.run(self.query_async(question, verbose))
