#!/usr/bin/env python3
"""Simple GraphRAG query test using pre-built community reports."""

import pandas as pd
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import numpy as np


async def query_graphrag_simple(question: str, top_k: int = 5):
    """Simple GraphRAG-style query using community reports."""

    # Load data
    reports = pd.read_parquet('graphrag_beige/output/community_reports.parquet')
    print(f"Loaded {len(reports)} community reports")

    # Embed the question
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    q_emb = embeddings.embed_query(question)

    # Embed report summaries and find top matches
    print("Embedding reports...")
    report_texts = reports['summary'].fillna('').tolist()
    report_embeddings = embeddings.embed_documents(report_texts)

    # Calculate similarities
    q_vec = np.array(q_emb)
    similarities = []
    for i, r_emb in enumerate(report_embeddings):
        sim = np.dot(q_vec, np.array(r_emb)) / (np.linalg.norm(q_vec) * np.linalg.norm(r_emb))
        similarities.append((i, sim))

    # Get top reports
    top_reports = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    print(f"\nTop {top_k} relevant reports (by similarity):")
    for idx, sim in top_reports:
        print(f"  - {reports.iloc[idx]['title']} (sim: {sim:.3f})")

    # Build context from top reports
    context = "\n\n---\n\n".join([
        f"## {reports.iloc[idx]['title']}\n{reports.iloc[idx]['full_content']}"
        for idx, _ in top_reports
    ])

    # Query LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = f"""Based on the following community reports from a knowledge graph about the Beige Book economic reports, answer this question:

Question: {question}

Community Reports:
{context}

Provide a comprehensive answer based only on the information in these reports. If the information isn't available, say so."""

    print("\nGenerating answer...")
    response = await asyncio.to_thread(llm.invoke, prompt)
    return response.content


if __name__ == "__main__":
    import sys

    question = sys.argv[1] if len(sys.argv) > 1 else "How did economic activity change in the San Francisco District (Twelfth District)?"

    answer = asyncio.run(query_graphrag_simple(question))
    print("\n" + "="*60)
    print("ANSWER:")
    print("="*60)
    print(answer)
