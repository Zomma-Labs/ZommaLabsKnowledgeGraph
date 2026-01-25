#!/usr/bin/env python3
"""Test Microsoft GraphRAG on Beige Book Q&A dataset."""

import sys
import pandas as pd
import asyncio
import json
import time
from pathlib import Path

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
        print(f"Loaded {len(self.reports)} community reports")

    async def _embed_reports(self):
        """Embed reports once for reuse."""
        if self._report_embeddings is None:
            print("Embedding reports (one-time)...")
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

Answer based only on the information provided. Be specific and cite data when available."""

        response = await asyncio.to_thread(self.llm.invoke, prompt)
        time_ms = int((time.time() - start) * 1000)

        return response.content, time_ms


async def run_test(questions: list[dict], limit: int = 10):
    """Run GraphRAG test."""

    graphrag = SimpleGraphRAG()

    results = []
    questions = questions[:limit]

    for i, qa in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] Q{qa['id']}: {qa['question'][:50]}...")

        answer, time_ms = await graphrag.query(qa['question'])

        print(f"  Time: {time_ms}ms")
        print(f"  Answer: {answer[:150]}...")

        results.append({
            "question_id": qa['id'],
            "question": qa['question'],
            "expected": qa['answer'],
            "graphrag_answer": answer,
            "time_ms": time_ms,
        })

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    avg_time = sum(r['time_ms'] for r in results) / len(results)
    print(f"Questions: {len(results)}")
    print(f"Avg time: {avg_time:.0f}ms")

    return results


if __name__ == "__main__":
    with open("eval/Biege_OA.json") as f:
        data = json.load(f)

    questions = data["qa_pairs"]
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    results = asyncio.run(run_test(questions, limit=limit))

    # Save results
    output_file = f"eval/graphrag_beige_test_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
