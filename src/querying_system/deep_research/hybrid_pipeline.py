"""
Hybrid Deep Research Pipeline
=============================

Combines deterministic retrieval with optional agent-based exploration.

Modes:
1. DETERMINISTIC: Pure multi-strategy retrieval, no agent decisions
2. HYBRID: Deterministic base + agent exploration for gaps
3. AGENT: Original agent-based approach (for comparison)

The deterministic mode eliminates variance from LLM tool selection.
"""

import asyncio
import time
from typing import Literal, Optional
from enum import Enum

from .state import DeepResearchResult, ResearchFinding
from .supervisor import run_supervisor
from .deterministic_researcher import run_deterministic_researcher, run_deterministic_research_batch
from .synthesizer import Synthesizer
from src.util.deterministic_retrieval import DeterministicRetriever


class ResearchMode(str, Enum):
    DETERMINISTIC = "deterministic"  # Pure deterministic retrieval
    HYBRID = "hybrid"                # Deterministic + agent exploration
    AGENT = "agent"                  # Original agent-based (for comparison)


class HybridDeepResearchPipeline:
    """
    Hybrid pipeline supporting multiple research modes.

    DETERMINISTIC mode:
    - Single deterministic retrieval on the main question
    - No LLM decisions in retrieval
    - Most consistent results

    HYBRID mode:
    - Deterministic retrieval as base
    - If confidence is low, spawn researchers for specific sub-topics
    - Balance of consistency and depth

    AGENT mode:
    - Original supervisor/researcher approach
    - LLM decides what to search
    - Most flexible but least consistent
    """

    def __init__(
        self,
        user_id: str = "default",
        mode: ResearchMode = ResearchMode.DETERMINISTIC,
        max_supervisor_iterations: int = 3,
        max_concurrent_researchers: int = 5,
    ):
        self.user_id = user_id
        self.mode = mode
        self.max_supervisor_iterations = max_supervisor_iterations
        self.max_concurrent_researchers = max_concurrent_researchers
        self.synthesizer = Synthesizer()

    async def _deterministic_research(
        self,
        question: str,
        top_k: int = 20,
        verbose: bool = False
    ) -> tuple[list[ResearchFinding], str]:
        """
        Pure deterministic research - no agent decisions.

        Returns:
            (findings, research_brief)
        """
        if verbose:
            print("  Mode: DETERMINISTIC (no LLM decisions in retrieval)")

        retriever = DeterministicRetriever(group_id=self.user_id)

        # Single deterministic retrieval
        evidence = await retriever.search(question, top_k=top_k)

        if verbose:
            print(f"  Retrieved: {len(evidence)} facts")
            strategies = set(s for e in evidence for s in e.found_by)
            print(f"  Strategies used: {strategies}")
            multi_strategy = sum(1 for e in evidence if len(e.found_by) > 1)
            print(f"  Multi-strategy hits: {multi_strategy}")

        # Convert to research findings format
        findings = []

        # Group evidence by topic/theme
        if evidence:
            # Create a single comprehensive finding
            evidence_text = retriever.format_evidence_for_llm(evidence)

            finding = ResearchFinding(
                topic=question,
                finding=evidence_text,
                confidence=0.9 if len(evidence) >= 10 else 0.7,
                evidence_chunks=[e.chunk_id for e in evidence if e.chunk_id],
                raw_content=evidence_text
            )
            findings.append(finding)

        research_brief = f"Deterministic retrieval found {len(evidence)} relevant facts using vector, keyword, and graph strategies."

        return findings, research_brief

    async def _hybrid_research(
        self,
        question: str,
        verbose: bool = False
    ) -> tuple[list[ResearchFinding], str]:
        """
        Hybrid: deterministic base + targeted agent exploration.

        1. Run deterministic retrieval first
        2. Analyze gaps in coverage
        3. If needed, spawn targeted researchers for specific sub-topics
        """
        if verbose:
            print("  Mode: HYBRID (deterministic base + targeted exploration)")

        # Phase 1: Deterministic base
        base_findings, base_brief = await self._deterministic_research(
            question, top_k=15, verbose=verbose
        )

        # Check if we need more exploration
        base_confidence = base_findings[0].confidence if base_findings else 0

        if base_confidence >= 0.8:
            if verbose:
                print(f"  Base confidence: {base_confidence:.1%} - skipping agent exploration")
            return base_findings, base_brief

        # Phase 2: Targeted agent exploration for gaps
        if verbose:
            print(f"  Base confidence: {base_confidence:.1%} - running targeted exploration")

        # Use supervisor to identify sub-topics, but use deterministic researchers
        from .supervisor import plan_research
        sub_topics = await plan_research(question, self.user_id)

        if sub_topics:
            # Run deterministic research on sub-topics
            sub_findings = await run_deterministic_research_batch(
                topics=[{"topic": t["topic"], "hints": t.get("hints", [])} for t in sub_topics],
                user_id=self.user_id,
                top_k=10,
                max_concurrent=self.max_concurrent_researchers
            )

            # Combine base + sub findings
            all_findings = base_findings + sub_findings
            research_brief = f"{base_brief} Additional targeted research on {len(sub_topics)} sub-topics."
        else:
            all_findings = base_findings
            research_brief = base_brief

        return all_findings, research_brief

    async def _agent_research(
        self,
        question: str,
        verbose: bool = False
    ) -> tuple[list[ResearchFinding], str]:
        """
        Original agent-based research (for comparison).
        """
        if verbose:
            print("  Mode: AGENT (original LLM-controlled)")

        findings, research_brief, _ = await run_supervisor(
            question=question,
            user_id=self.user_id,
            max_iterations=self.max_supervisor_iterations,
            max_concurrent=self.max_concurrent_researchers
        )

        return findings, research_brief

    async def query_async(
        self,
        question: str,
        verbose: bool = False
    ) -> DeepResearchResult:
        """
        Answer a question using the configured research mode.
        """
        total_start = time.time()

        if verbose:
            print(f"\n{'='*60}")
            print(f"HYBRID DEEP RESEARCH: {question}")
            print(f"{'='*60}\n")
            print(">>> PHASE 1: Research")

        research_start = time.time()

        # Route to appropriate research method
        if self.mode == ResearchMode.DETERMINISTIC:
            findings, research_brief = await self._deterministic_research(question, verbose=verbose)
        elif self.mode == ResearchMode.HYBRID:
            findings, research_brief = await self._hybrid_research(question, verbose=verbose)
        else:  # AGENT
            findings, research_brief = await self._agent_research(question, verbose=verbose)

        research_time = int((time.time() - research_start) * 1000)

        if verbose:
            print(f"  Research Time: {research_time}ms")
            print(f"  Findings: {len(findings)}")

        # Phase 2: Synthesize
        if verbose:
            print("\n>>> PHASE 2: Synthesizing Answer")

        answer, synthesis_time = self.synthesizer.synthesize(
            question=question,
            research_brief=research_brief,
            findings=findings
        )

        if verbose:
            print(f"  Synthesis Time: {synthesis_time}ms")
            print(f"\n{'='*60}")
            print("ANSWER:")
            print(f"{'='*60}")
            print(answer[:500] + "..." if len(answer) > 500 else answer)

        total_time = int((time.time() - total_start) * 1000)

        return DeepResearchResult(
            question=question,
            answer=answer,
            research_brief=research_brief,
            findings=findings,
            planning_time_ms=0,
            research_time_ms=research_time,
            synthesis_time_ms=synthesis_time
        )

    def query(self, question: str, verbose: bool = False) -> DeepResearchResult:
        """Synchronous wrapper."""
        return asyncio.run(self.query_async(question, verbose))


# === Convenience Functions ===

def query_deterministic(question: str, user_id: str = "default", verbose: bool = False) -> DeepResearchResult:
    """Quick deterministic query."""
    pipeline = HybridDeepResearchPipeline(user_id=user_id, mode=ResearchMode.DETERMINISTIC)
    return pipeline.query(question, verbose)


def query_hybrid(question: str, user_id: str = "default", verbose: bool = False) -> DeepResearchResult:
    """Quick hybrid query."""
    pipeline = HybridDeepResearchPipeline(user_id=user_id, mode=ResearchMode.HYBRID)
    return pipeline.query(question, verbose)


def query_agent(question: str, user_id: str = "default", verbose: bool = False) -> DeepResearchResult:
    """Quick agent query (original behavior)."""
    pipeline = HybridDeepResearchPipeline(user_id=user_id, mode=ResearchMode.AGENT)
    return pipeline.query(question, verbose)


# === CLI Entry Point ===

def main():
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid Deep Research")
    parser.add_argument("question", nargs="?", default="Which districts reported slight to modest economic growth?")
    parser.add_argument("--mode", choices=["deterministic", "hybrid", "agent"], default="deterministic")
    parser.add_argument("--user-id", default="default")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    mode = ResearchMode(args.mode)
    pipeline = HybridDeepResearchPipeline(user_id=args.user_id, mode=mode)

    result = pipeline.query(args.question, verbose=args.verbose)

    print(f"\n{'='*60}")
    print("METRICS:")
    print(f"{'='*60}")
    print(f"Mode: {mode.value}")
    print(f"Total Time: {result.total_time_ms}ms")
    print(f"  - Research: {result.research_time_ms}ms")
    print(f"  - Synthesis: {result.synthesis_time_ms}ms")
    print(f"Findings: {len(result.findings)}")


if __name__ == "__main__":
    main()
