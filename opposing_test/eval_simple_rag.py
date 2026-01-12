"""
Evaluate Simple RAG on Previously Incorrect Questions
======================================================

Compares simple chunk-based RAG against the knowledge graph approach.

Usage:
    uv run python opposing_test/eval_simple_rag.py
"""

import os
import sys
import json
import asyncio
import time
from datetime import datetime
from enum import Enum
from typing import List

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_voyageai import VoyageAIEmbeddings
from pydantic import BaseModel, Field
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.util.llm_client import get_llm, get_critique_llm


class JudgeVerdict(str, Enum):
    CORRECT = "correct"
    PARTIALLY_CORRECT = "partially_correct"
    INCORRECT = "incorrect"


class JudgeResult(BaseModel):
    verdict: JudgeVerdict
    reasoning: str


JUDGE_PROMPT = """You are an impartial judge evaluating whether an AI agent's answer correctly addresses a question.

## Verdict Options:
- **correct**: The agent's answer contains all key facts from the expected answer
- **partially_correct**: The agent's answer contains some but not all key facts
- **incorrect**: The agent's answer is wrong or contradicts the expected answer

Be fair but rigorous. Focus on factual accuracy, not exact wording."""


RAG_SYSTEM_PROMPT = """You are a financial analyst assistant. Answer the user's question based ONLY on the provided context.

Rules:
1. Only use information explicitly in the context
2. Preserve attribution - note which source/section information comes from
3. Be concise and direct
4. If the context doesn't contain the answer, say so"""


def load_chunks(filepath: str) -> List[Document]:
    """Load chunks from JSONL file."""
    documents = []
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line)
            text = data.get('body') or data.get('chunk_text') or data.get('text', '')
            header = data.get('header_path', '')

            # Prepend header to content for attribution
            if header:
                full_text = f"[{header}]\n{text}"
            else:
                full_text = text

            if full_text.strip():
                documents.append(Document(
                    page_content=full_text,
                    metadata={"header": header, "uuid": data.get('uuid', '')}
                ))
    return documents


class SimpleRAG:
    """Simple vector-search RAG over chunks."""

    def __init__(self, chunk_file: str):
        print("Loading chunks...")
        self.docs = load_chunks(chunk_file)
        print(f"Loaded {len(self.docs)} chunks")

        print("Building vector index...")
        self.embeddings = VoyageAIEmbeddings(model="voyage-finance-2")

        # Embed all chunks
        texts = [doc.page_content for doc in self.docs]
        self.doc_embeddings = np.array(self.embeddings.embed_documents(texts))
        print(f"Embedded {len(self.doc_embeddings)} chunks")

        self.llm = get_llm()
        print("Simple RAG ready")

    def _retrieve(self, query: str, k: int = 15) -> List[Document]:
        """Retrieve top-k most similar chunks."""
        query_embedding = np.array(self.embeddings.embed_query(query))

        # Cosine similarity
        similarities = np.dot(self.doc_embeddings, query_embedding) / (
            np.linalg.norm(self.doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        top_indices = np.argsort(similarities)[-k:][::-1]
        return [self.docs[i] for i in top_indices]

    def query(self, question: str) -> tuple[str, int, int]:
        """
        Answer a question using simple RAG.

        Returns: (answer, num_chunks_retrieved, time_ms)
        """
        start = time.time()

        # Retrieve relevant chunks
        relevant_docs = self._retrieve(question, k=15)
        num_chunks = len(relevant_docs)

        # Build context
        context_parts = []
        for i, doc in enumerate(relevant_docs, 1):
            context_parts.append(f"--- Chunk {i} ---\n{doc.page_content}")
        context = "\n\n".join(context_parts)

        # Generate answer
        prompt = f"""Question: {question}

Context:
{context}

Based on the context above, answer the question."""

        response = self.llm.invoke([
            SystemMessage(content=RAG_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ])

        elapsed_ms = int((time.time() - start) * 1000)
        return response.content, num_chunks, elapsed_ms


def load_incorrect_questions(prev_eval_file: str, qa_file: str) -> tuple[list, dict]:
    """Load questions that were incorrect in previous eval."""
    with open(prev_eval_file) as f:
        prev_results = json.load(f)

    incorrect_ids = set()
    for r in prev_results.get("results", []):
        if r.get("verdict", "").lower() == "incorrect":
            incorrect_ids.add(r["question_id"])

    print(f"Found {len(incorrect_ids)} incorrect questions: {sorted(incorrect_ids)}")

    with open(qa_file) as f:
        qa_data = json.load(f)

    qa_pairs = qa_data.get("qa_pairs", qa_data)
    incorrect_pairs = [p for p in qa_pairs if p.get("id") in incorrect_ids]

    return incorrect_pairs, prev_results


def get_judge():
    """Create the LLM judge."""
    llm = get_critique_llm()
    return llm.with_structured_output(JudgeResult)


def evaluate_question(rag: SimpleRAG, judge, question: str, expected: str, q_id: int) -> dict:
    """Evaluate a single question."""
    # Get RAG answer
    answer, num_chunks, rag_time = rag.query(question)

    # Judge the answer
    judge_start = time.time()
    try:
        judge_result = judge.invoke([
            SystemMessage(content=JUDGE_PROMPT),
            HumanMessage(content=f"""## QUESTION
{question}

## EXPECTED ANSWER
{expected}

## AGENT'S ANSWER
{answer}

Evaluate whether the agent's answer is correct.""")
        ])
    except Exception as e:
        judge_result = JudgeResult(
            verdict=JudgeVerdict.INCORRECT,
            reasoning=f"Judge error: {e}"
        )
    judge_time = time.time() - judge_start

    return {
        "question_id": q_id,
        "question": question,
        "expected": expected,
        "answer": answer,
        "verdict": judge_result.verdict.value,
        "reasoning": judge_result.reasoning,
        "num_chunks": num_chunks,
        "rag_time_ms": rag_time,
        "judge_time_sec": judge_time
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prev-eval", default="eval_two_agent_20260107_000831.json")
    parser.add_argument("--qa-file", default="Biege_OA.json")
    parser.add_argument("--chunk-file", default="src/chunker/SAVED/beigebook_20251015.jsonl")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    prev_eval = args.prev_eval if os.path.isabs(args.prev_eval) else os.path.join(project_root, args.prev_eval)
    qa_file = args.qa_file if os.path.isabs(args.qa_file) else os.path.join(project_root, args.qa_file)
    chunk_file = args.chunk_file if os.path.isabs(args.chunk_file) else os.path.join(project_root, args.chunk_file)

    # Load incorrect questions
    qa_pairs, _ = load_incorrect_questions(prev_eval, qa_file)
    if not qa_pairs:
        print("No incorrect questions found!")
        return

    # Initialize RAG
    rag = SimpleRAG(chunk_file)
    judge = get_judge()

    print(f"\n{'='*70}")
    print(f"SIMPLE RAG EVALUATION")
    print(f"Questions to evaluate: {len(qa_pairs)}")
    print(f"{'='*70}\n")

    results = []
    for i, pair in enumerate(qa_pairs):
        q_id = pair.get("id", i + 1)
        question = pair["question"]
        expected = pair["answer"]

        print(f"[{i+1}/{len(qa_pairs)}] Q{q_id}: {question[:50]}...", end=" ", flush=True)

        result = evaluate_question(rag, judge, question, expected, q_id)
        results.append(result)

        verdict = result["verdict"]
        if verdict == "correct":
            icon = "✅"
        elif verdict == "partially_correct":
            icon = "⚠️"
        else:
            icon = "❌"

        print(f"[{icon}] ({result['rag_time_ms']}ms, {result['num_chunks']} chunks)")

    # Summary
    total = len(results)
    correct = sum(1 for r in results if r["verdict"] == "correct")
    partial = sum(1 for r in results if r["verdict"] == "partially_correct")
    incorrect = sum(1 for r in results if r["verdict"] == "incorrect")
    avg_time = sum(r["rag_time_ms"] for r in results) / total

    print(f"\n{'='*70}")
    print("SIMPLE RAG RESULTS")
    print(f"{'='*70}")
    print(f"  ✅ Correct:          {correct:3d} ({correct/total*100:5.1f}%)")
    print(f"  ⚠️  Partially Correct: {partial:3d} ({partial/total*100:5.1f}%)")
    print(f"  ❌ Incorrect:        {incorrect:3d} ({incorrect/total*100:5.1f}%)")
    print(f"\n  Avg Time: {avg_time:.0f}ms")
    print(f"{'='*70}")

    # Save results
    output = f"eval_simple_rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "system": "simple_rag",
            "total_questions": total,
            "summary": {
                "correct": correct,
                "partially_correct": partial,
                "incorrect": incorrect,
                "avg_time_ms": avg_time
            },
            "results": results
        }, f, indent=2)
    print(f"\nResults saved to: {output}")


if __name__ == "__main__":
    main()
