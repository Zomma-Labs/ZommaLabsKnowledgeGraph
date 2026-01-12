"""
Synthesizer: Combines research findings into a coherent answer.
"""

import time
from langchain_core.messages import SystemMessage, HumanMessage

from .state import ResearchFinding
from .prompts import SYNTHESIZER_SYSTEM_PROMPT
from src.util.llm_client import get_llm


class Synthesizer:
    """Synthesizes research findings into a final answer."""

    def __init__(self):
        self.llm = get_llm()

    def synthesize(
        self,
        question: str,
        research_brief: str,
        findings: list[ResearchFinding]
    ) -> tuple[str, int]:
        """
        Synthesize findings into a coherent answer.

        Args:
            question: Original user question
            research_brief: The focused research brief
            findings: List of research findings

        Returns:
            tuple of (answer, elapsed_time_ms)
        """
        start = time.time()

        # Format findings - include full evidence
        findings_text = ""
        if findings:
            for i, f in enumerate(findings, 1):
                findings_text += f"""
### Finding {i}: {f.topic}
**Result:** {f.finding}
**Confidence:** {f.confidence}
**Raw Evidence:**
{f.raw_content if f.raw_content else 'No raw evidence available'}

---
"""
        else:
            findings_text = "No research findings available."

        prompt = f"""Original Question: {question}

Research Brief: {research_brief}

Research Findings:
{findings_text}

Based on the research findings above, provide a comprehensive answer to the original question."""

        messages = [
            SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]

        response = self.llm.invoke(messages)
        answer = response.content

        elapsed_ms = int((time.time() - start) * 1000)
        return answer, elapsed_ms
