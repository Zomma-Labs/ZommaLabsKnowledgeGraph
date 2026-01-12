"""
Deep Research Pipeline: Main orchestrator.

Three-phase workflow:
1. Supervisor: Plans and delegates research
2. Researchers: Parallel investigation of sub-topics
3. Synthesizer: Combines findings into final answer
"""

import asyncio
import time
from typing import Optional

from .state import DeepResearchResult, ResearchFinding
from .supervisor import run_supervisor
from .synthesizer import Synthesizer


class DeepResearchPipeline:
    """
    Main pipeline for deep research on complex questions.

    Flow:
    Question → Supervisor (plans & delegates) → Researchers (parallel) → Synthesizer → Answer
    """

    def __init__(
        self,
        user_id: str = "default",
        max_supervisor_iterations: int = 3,
        max_concurrent_researchers: int = 8,
    ):
        """
        Initialize the pipeline.

        Args:
            user_id: User/tenant ID for graph scoping
            max_supervisor_iterations: Max times supervisor can spawn researchers
            max_concurrent_researchers: Max parallel researcher agents
        """
        self.user_id = user_id
        self.max_supervisor_iterations = max_supervisor_iterations
        self.max_concurrent_researchers = max_concurrent_researchers
        self.synthesizer = Synthesizer()

    async def query_async(
        self,
        question: str,
        verbose: bool = False
    ) -> DeepResearchResult:
        """
        Answer a question using deep research (async version).

        Args:
            question: The user's question
            verbose: Print progress if True

        Returns:
            DeepResearchResult with full details
        """
        total_start = time.time()

        if verbose:
            print(f"\n{'='*60}")
            print(f"DEEP RESEARCH: {question}")
            print(f"{'='*60}\n")

        # Phase 1: Supervisor plans and executes research
        if verbose:
            print(">>> PHASE 1: Research Planning & Execution")

        research_start = time.time()
        findings, research_brief, _ = await run_supervisor(
            question=question,
            user_id=self.user_id,
            max_iterations=self.max_supervisor_iterations,
            max_concurrent=self.max_concurrent_researchers
        )
        research_time = int((time.time() - research_start) * 1000)

        if verbose:
            print(f"    Research Brief: {research_brief[:100]}...")
            print(f"    Findings: {len(findings)}")
            for f in findings:
                print(f"      - {f.topic}: {f.finding[:80]}... (conf: {f.confidence})")
            print(f"    Research Time: {research_time}ms\n")

        # Phase 2: Synthesize findings into answer
        if verbose:
            print(">>> PHASE 2: Synthesizing Answer")

        answer, synthesis_time = self.synthesizer.synthesize(
            question=question,
            research_brief=research_brief,
            findings=findings
        )

        if verbose:
            print(f"    Synthesis Time: {synthesis_time}ms\n")
            print(f"{'='*60}")
            print("ANSWER:")
            print(f"{'='*60}")
            print(answer)

        total_time = int((time.time() - total_start) * 1000)

        return DeepResearchResult(
            question=question,
            answer=answer,
            research_brief=research_brief,
            findings=findings,
            planning_time_ms=0,  # Included in research
            research_time_ms=research_time,
            synthesis_time_ms=synthesis_time
        )

    def query(
        self,
        question: str,
        verbose: bool = False
    ) -> DeepResearchResult:
        """
        Synchronous wrapper for query_async.

        Args:
            question: The user's question
            verbose: Print progress if True

        Returns:
            DeepResearchResult with full details
        """
        return asyncio.run(self.query_async(question, verbose))


# === CLI Entry Point ===

def main():
    """CLI for testing the deep research pipeline."""
    import sys

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "Which Federal Reserve Bank prepared the October 2025 Beige Book report?"

    pipeline = DeepResearchPipeline(user_id="default")

    result = pipeline.query(question, verbose=True)

    print(f"\n{'='*60}")
    print("METRICS:")
    print(f"{'='*60}")
    print(f"Total Time: {result.total_time_ms}ms")
    print(f"  - Research: {result.research_time_ms}ms")
    print(f"  - Synthesis: {result.synthesis_time_ms}ms")
    print(f"Findings: {len(result.findings)}")


if __name__ == "__main__":
    main()
