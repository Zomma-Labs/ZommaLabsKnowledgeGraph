"""
MODULE: LLM Judge
DESCRIPTION: LLM-based judge for evaluating RAG system answers against expected answers.
             Distinguishes between 4 verdict categories: correct, partially correct,
             abstained (said "don't know" when answer exists), and incorrect.
"""

import asyncio
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from testing.common.schemas import JudgeVerdict


class JudgeOutput(BaseModel):
    """Structured output from the LLM judge."""

    reasoning: str = Field(
        description="Step-by-step reasoning for the verdict, analyzing the system answer against the expected answer"
    )
    verdict: str = Field(
        description="One of: 'correct', 'partially', 'abstained', 'incorrect'"
    )


JUDGE_SYSTEM_PROMPT = """You are an expert judge evaluating RAG (Retrieval-Augmented Generation) system answers.

Your task is to compare a system's answer against the expected (ground truth) answer and determine the verdict.

## Verdict Categories

You MUST classify the answer into exactly ONE of these 4 categories:

1. **CORRECT** ("correct")
   - All key facts from the expected answer are present in the system answer
   - No factual errors or contradictions
   - Minor differences in wording or phrasing are acceptable
   - Additional correct details beyond the expected answer are GOOD, not penalized
   - More comprehensive answers that include the expected content = CORRECT

2. **PARTIALLY_CORRECT** ("partially")
   - Some (but not all) key facts from the expected answer are present
   - The facts that ARE present are accurate
   - Nothing in the answer is factually wrong or contradictory
   - The answer is incomplete but not misleading
   - Missing some items from an enumeration (e.g., lists 4 of 5 districts) = PARTIALLY, not incorrect

3. **ABSTAINED** ("abstained")
   - The system explicitly states it cannot find or doesn't have the information
   - Common patterns include:
     * "I don't have information about..."
     * "The context doesn't contain..."
     * "I cannot find..."
     * "Based on the available information, I'm unable to..."
     * "No relevant information was found..."
     * Empty or blank responses
   - The system declined to answer when a correct answer exists
   - This is NOT the same as giving wrong information

4. **INCORRECT** ("incorrect")
   - The system CONTRADICTS the expected answer on a key factual point
   - Example: Expected "demand was unchanged", System says "demand declined" = INCORRECT
   - Example: Expected "conditions were mixed", System says "conditions were uniformly negative" = INCORRECT
   - Reserved for genuine factual errors, not stylistic differences or additional detail

## CRITICAL GUIDELINES

**Comprehensiveness is GOOD:**
- A more detailed answer that includes the expected facts + additional context = CORRECT
- Adding contextual information (dates, years, explanations) = helpful, not an error
- Answering a broad question with multi-district data = thorough, not wrong

**INCORRECT is reserved for contradictions:**
- Only use INCORRECT when the answer directly contradicts expected facts
- Missing details = PARTIALLY_CORRECT, not incorrect
- Different phrasing with same meaning = CORRECT
- More detail than expected = CORRECT

## Evaluation Process

1. Identify the key facts in the expected answer
2. Check if each key fact is present in the system answer
3. Check for any factual CONTRADICTIONS (not just differences)
4. Check if the system abstained from answering
5. Assign the appropriate verdict

## Output Format

Provide your reasoning first, then the verdict. Be specific about which facts matched, which were missing, and any contradictions found."""


JUDGE_USER_PROMPT = """## Question
{question}

## Expected Answer (Ground Truth)
{expected_answer}

## System Answer (To Evaluate)
{system_answer}

Evaluate the system answer and provide your verdict."""


class LLMJudge:
    """LLM-based judge for evaluating RAG system answers.

    Uses structured output to ensure consistent verdict format.

    Usage:
        judge = LLMJudge(model="gpt-5.1")
        verdict, reasoning = await judge.judge(question, expected, system_answer)
    """

    def __init__(self, model: str = "gpt-5.1", temperature: float = 0.0):
        """Initialize the LLM judge.

        Args:
            model: OpenAI model to use for judging. Defaults to gpt-5.1.
            temperature: LLM temperature. Defaults to 0.0 for deterministic output.
        """
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.structured_llm = self.llm.with_structured_output(JudgeOutput)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", JUDGE_SYSTEM_PROMPT),
            ("user", JUDGE_USER_PROMPT),
        ])

        self.chain = self.prompt | self.structured_llm

    def _parse_verdict(self, verdict_str: str) -> JudgeVerdict:
        """Parse a verdict string into a JudgeVerdict enum.

        Args:
            verdict_str: The verdict string from the LLM output.

        Returns:
            The corresponding JudgeVerdict enum value.
        """
        verdict_str = verdict_str.lower().strip()

        # Map possible outputs to enum values
        verdict_map = {
            "correct": JudgeVerdict.CORRECT,
            "partially": JudgeVerdict.PARTIALLY_CORRECT,
            "partially_correct": JudgeVerdict.PARTIALLY_CORRECT,
            "partial": JudgeVerdict.PARTIALLY_CORRECT,
            "abstained": JudgeVerdict.ABSTAINED,
            "abstain": JudgeVerdict.ABSTAINED,
            "incorrect": JudgeVerdict.INCORRECT,
            "wrong": JudgeVerdict.INCORRECT,
        }

        if verdict_str in verdict_map:
            return verdict_map[verdict_str]

        # Default to INCORRECT if unknown verdict (should not happen with structured output)
        return JudgeVerdict.INCORRECT

    def _is_empty_or_abstained(self, answer: str) -> bool:
        """Check if an answer is empty or clearly an abstention.

        Args:
            answer: The system answer to check.

        Returns:
            True if the answer is empty or a clear abstention.
        """
        if not answer or not answer.strip():
            return True

        answer_lower = answer.lower().strip()

        # Check for common abstention patterns
        abstention_phrases = [
            "i don't have",
            "i do not have",
            "i cannot find",
            "i can't find",
            "unable to find",
            "no information available",
            "no relevant information",
            "the context doesn't contain",
            "the context does not contain",
            "not mentioned in",
            "no data available",
            "i'm unable to",
            "i am unable to",
        ]

        # Short answers that are just abstentions
        if len(answer_lower) < 100:
            for phrase in abstention_phrases:
                if phrase in answer_lower:
                    return True

        return False

    async def judge(
        self,
        question: str,
        expected_answer: str,
        system_answer: str,
    ) -> tuple[JudgeVerdict, str]:
        """Judge a system answer against the expected answer.

        Args:
            question: The question that was asked.
            expected_answer: The expected (ground truth) answer.
            system_answer: The system's answer to evaluate.

        Returns:
            Tuple of (verdict, reasoning).
        """
        # Handle empty system answers as ABSTAINED without calling LLM
        if not system_answer or not system_answer.strip():
            return (
                JudgeVerdict.ABSTAINED,
                "System answer is empty or blank, which counts as abstaining from answering.",
            )

        try:
            result: JudgeOutput = await self.chain.ainvoke({
                "question": question,
                "expected_answer": expected_answer,
                "system_answer": system_answer,
            })

            verdict = self._parse_verdict(result.verdict)
            return (verdict, result.reasoning)

        except Exception as e:
            # On error, return INCORRECT with error message
            return (
                JudgeVerdict.INCORRECT,
                f"Error during judgment: {str(e)}",
            )

    async def batch_judge(
        self,
        judgments: list[tuple[str, str, str]],
        max_concurrency: int = 10,
    ) -> list[tuple[JudgeVerdict, str]]:
        """Judge multiple answers concurrently.

        Args:
            judgments: List of (question, expected_answer, system_answer) tuples.
            max_concurrency: Maximum number of concurrent LLM calls. Defaults to 10.

        Returns:
            List of (verdict, reasoning) tuples in the same order as input.
        """
        if not judgments:
            return []

        # Use semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrency)

        async def judge_with_semaphore(
            question: str, expected: str, system: str
        ) -> tuple[JudgeVerdict, str]:
            async with semaphore:
                return await self.judge(question, expected, system)

        # Create tasks for all judgments
        tasks = [
            judge_with_semaphore(q, e, s) for q, e, s in judgments
        ]

        # Run all tasks concurrently
        results = await asyncio.gather(*tasks)

        return list(results)

    def judge_sync(
        self,
        question: str,
        expected_answer: str,
        system_answer: str,
    ) -> tuple[JudgeVerdict, str]:
        """Synchronous wrapper for judge().

        Args:
            question: The question that was asked.
            expected_answer: The expected (ground truth) answer.
            system_answer: The system's answer to evaluate.

        Returns:
            Tuple of (verdict, reasoning).
        """
        return asyncio.run(self.judge(question, expected_answer, system_answer))
