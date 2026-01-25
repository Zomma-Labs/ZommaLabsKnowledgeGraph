#!/usr/bin/env python3
"""Compare GraphRAG vs V7 on Beige Book questions."""

import sys
import pandas as pd
import asyncio
import json
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import numpy as np


class SimpleGraphRAG:
    """Simple GraphRAG using community reports."""

    def __init__(self, index_path: str = "graphrag_beige"):
        self.reports = pd.read_parquet(f"{index_path}/output/community_reports.parquet")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self._report_embeddings = None

    async def _embed_reports(self):
        """Embed reports once for reuse."""
        if self._report_embeddings is None:
            report_texts = self.reports['summary'].fillna('').tolist()
            self._report_embeddings = await asyncio.to_thread(
                self.embeddings.embed_documents, report_texts
            )
        return self._report_embeddings

    async def query(self, question: str, top_k: int = 5) -> tuple[str, int]:
        """Query and return (answer, time_ms)."""
        start = time.time()

        # Embed question
        q_emb = await asyncio.to_thread(self.embeddings.embed_query, question)
        report_embs = await self._embed_reports()

        # Calculate similarities
        q_vec = np.array(q_emb)
        similarities = []
        for i, r_emb in enumerate(report_embs):
            sim = np.dot(q_vec, np.array(r_emb)) / (np.linalg.norm(q_vec) * np.linalg.norm(r_emb))
            similarities.append((i, sim))

        # Get top reports
        top_reports = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

        # Build context
        context = "\n\n---\n\n".join([
            f"## {self.reports.iloc[idx]['title']}\n{self.reports.iloc[idx]['full_content']}"
            for idx, _ in top_reports
        ])

        # Query LLM
        prompt = f"""Based on the following community reports about the Beige Book economic reports, answer this question concisely:

Question: {question}

Community Reports:
{context}

Answer based only on the information provided. If the information isn't available, say so."""

        response = await asyncio.to_thread(self.llm.invoke, prompt)
        time_ms = int((time.time() - start) * 1000)

        return response.content, time_ms


async def run_comparison(questions: list[dict], limit: int = 5):
    """Run comparison between GraphRAG and V7."""

    # Initialize systems
    print("Initializing GraphRAG...")
    graphrag = SimpleGraphRAG()

    print("Initializing V7...")
    from src.querying_system.v7 import V7Pipeline
    v7 = V7Pipeline(group_id="default")

    results = []
    questions = questions[:limit]

    for i, qa in enumerate(questions):
        print(f"\n{'='*60}")
        print(f"Question {qa['id']}: {qa['question'][:60]}...")
        print(f"Expected: {qa['answer'][:100]}...")

        # GraphRAG
        print("\n[GraphRAG]")
        gr_answer, gr_time = await graphrag.query(qa['question'])
        print(f"  Time: {gr_time}ms")
        print(f"  Answer: {gr_answer[:200]}...")

        # V7
        print("\n[V7]")
        v7_start = time.time()
        v7_result = await v7.query(qa['question'])
        v7_time = int((time.time() - v7_start) * 1000)
        print(f"  Time: {v7_time}ms")
        print(f"  Answer: {v7_result.answer[:200]}...")
        print(f"  Confidence: {v7_result.confidence:.2f}")

        results.append({
            "question_id": qa['id'],
            "question": qa['question'],
            "expected": qa['answer'],
            "graphrag_answer": gr_answer,
            "graphrag_time_ms": gr_time,
            "v7_answer": v7_result.answer,
            "v7_time_ms": v7_time,
            "v7_confidence": v7_result.confidence,
        })

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    gr_avg_time = sum(r['graphrag_time_ms'] for r in results) / len(results)
    v7_avg_time = sum(r['v7_time_ms'] for r in results) / len(results)
    print(f"GraphRAG avg time: {gr_avg_time:.0f}ms")
    print(f"V7 avg time: {v7_avg_time:.0f}ms")

    return results


if __name__ == "__main__":
    # Load Q&A dataset
    with open("eval/Biege_OA.json") as f:
        data = json.load(f)

    questions = data["qa_pairs"]
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    results = asyncio.run(run_comparison(questions, limit=limit))

    # Save results
    output_file = f"eval/compare_graphrag_v7_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
