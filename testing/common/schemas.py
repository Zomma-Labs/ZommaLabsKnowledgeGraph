"""Result schema dataclasses for the RAG evaluation framework."""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class JudgeVerdict(str, Enum):
    """Verdict categories for evaluating RAG system answers."""

    CORRECT = "correct"  # All key facts present and accurate
    PARTIALLY_CORRECT = "partially"  # Some facts present, nothing wrong
    ABSTAINED = "abstained"  # Says "couldn't find" when answer exists
    INCORRECT = "incorrect"  # Wrong or contradictory information


@dataclass
class EvalResult:
    """Result of evaluating a single question against a RAG system."""

    question_id: int
    question: str
    expected_answer: str
    system_answer: str
    verdict: JudgeVerdict
    judge_reasoning: str
    retrieved_chunks: list[dict] = field(default_factory=list)  # For manual review
    timing_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "question_id": self.question_id,
            "question": self.question,
            "expected_answer": self.expected_answer,
            "system_answer": self.system_answer,
            "verdict": self.verdict.value,
            "judge_reasoning": self.judge_reasoning,
            "retrieved_chunks": self.retrieved_chunks,
            "timing_ms": self.timing_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalResult":
        """Create an EvalResult from a dictionary."""
        return cls(
            question_id=data["question_id"],
            question=data["question"],
            expected_answer=data["expected_answer"],
            system_answer=data["system_answer"],
            verdict=JudgeVerdict(data["verdict"]),
            judge_reasoning=data["judge_reasoning"],
            retrieved_chunks=data.get("retrieved_chunks", []),
            timing_ms=data.get("timing_ms", 0),
        )

    @property
    def is_correct(self) -> bool:
        """Check if the verdict is correct."""
        return self.verdict == JudgeVerdict.CORRECT

    @property
    def is_partially_correct(self) -> bool:
        """Check if the verdict is partially correct."""
        return self.verdict == JudgeVerdict.PARTIALLY_CORRECT

    @property
    def is_abstained(self) -> bool:
        """Check if the system abstained from answering."""
        return self.verdict == JudgeVerdict.ABSTAINED

    @property
    def is_incorrect(self) -> bool:
        """Check if the verdict is incorrect."""
        return self.verdict == JudgeVerdict.INCORRECT


@dataclass
class EvalSummary:
    """Aggregated summary of evaluation results for a RAG system."""

    system_name: str
    total_questions: int
    correct: int
    partially_correct: int
    abstained: int
    incorrect: int
    avg_time_ms: float
    results: list[EvalResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "system_name": self.system_name,
            "total_questions": self.total_questions,
            "correct": self.correct,
            "partially_correct": self.partially_correct,
            "abstained": self.abstained,
            "incorrect": self.incorrect,
            "avg_time_ms": self.avg_time_ms,
            "results": [r.to_dict() for r in self.results],
            # Include computed percentages
            "correct_pct": self.correct_pct,
            "partially_correct_pct": self.partially_correct_pct,
            "abstained_pct": self.abstained_pct,
            "incorrect_pct": self.incorrect_pct,
            "accuracy_pct": self.accuracy_pct,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalSummary":
        """Create an EvalSummary from a dictionary."""
        results = [EvalResult.from_dict(r) for r in data.get("results", [])]
        return cls(
            system_name=data["system_name"],
            total_questions=data["total_questions"],
            correct=data["correct"],
            partially_correct=data["partially_correct"],
            abstained=data["abstained"],
            incorrect=data["incorrect"],
            avg_time_ms=data["avg_time_ms"],
            results=results,
        )

    @classmethod
    def from_results(
        cls, system_name: str, results: list[EvalResult]
    ) -> "EvalSummary":
        """Create an EvalSummary by aggregating a list of EvalResults."""
        if not results:
            return cls(
                system_name=system_name,
                total_questions=0,
                correct=0,
                partially_correct=0,
                abstained=0,
                incorrect=0,
                avg_time_ms=0.0,
                results=[],
            )

        correct = sum(1 for r in results if r.verdict == JudgeVerdict.CORRECT)
        partially_correct = sum(
            1 for r in results if r.verdict == JudgeVerdict.PARTIALLY_CORRECT
        )
        abstained = sum(1 for r in results if r.verdict == JudgeVerdict.ABSTAINED)
        incorrect = sum(1 for r in results if r.verdict == JudgeVerdict.INCORRECT)
        avg_time_ms = sum(r.timing_ms for r in results) / len(results)

        return cls(
            system_name=system_name,
            total_questions=len(results),
            correct=correct,
            partially_correct=partially_correct,
            abstained=abstained,
            incorrect=incorrect,
            avg_time_ms=avg_time_ms,
            results=results,
        )

    @property
    def correct_pct(self) -> float:
        """Percentage of questions answered correctly."""
        if self.total_questions == 0:
            return 0.0
        return (self.correct / self.total_questions) * 100

    @property
    def partially_correct_pct(self) -> float:
        """Percentage of questions answered partially correctly."""
        if self.total_questions == 0:
            return 0.0
        return (self.partially_correct / self.total_questions) * 100

    @property
    def abstained_pct(self) -> float:
        """Percentage of questions where the system abstained."""
        if self.total_questions == 0:
            return 0.0
        return (self.abstained / self.total_questions) * 100

    @property
    def incorrect_pct(self) -> float:
        """Percentage of questions answered incorrectly."""
        if self.total_questions == 0:
            return 0.0
        return (self.incorrect / self.total_questions) * 100

    @property
    def accuracy_pct(self) -> float:
        """Overall accuracy (correct + partially correct) percentage."""
        if self.total_questions == 0:
            return 0.0
        return ((self.correct + self.partially_correct) / self.total_questions) * 100

    def __str__(self) -> str:
        """Human-readable summary string."""
        return (
            f"EvalSummary({self.system_name}): "
            f"{self.correct}/{self.total_questions} correct ({self.correct_pct:.1f}%), "
            f"{self.partially_correct} partial ({self.partially_correct_pct:.1f}%), "
            f"{self.abstained} abstained ({self.abstained_pct:.1f}%), "
            f"{self.incorrect} incorrect ({self.incorrect_pct:.1f}%), "
            f"avg time: {self.avg_time_ms:.0f}ms"
        )
