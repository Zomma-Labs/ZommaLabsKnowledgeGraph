#!/usr/bin/env python3
"""
V6 Knowledge Graph vs Simple RAG Comparison Test
=================================================

Tests both pipelines on:
1. Beige Book QA (Biege_OA.json) - 75 questions
2. Alphabet QA (Alphabet_QA.json) - 50 questions

Usage:
    uv run scripts/test_v6_vs_rag_comparison.py                    # Run all tests
    uv run scripts/test_v6_vs_rag_comparison.py --v6-only          # Only V6
    uv run scripts/test_v6_vs_rag_comparison.py --rag-only         # Only RAG
    uv run scripts/test_v6_vs_rag_comparison.py --beige-only       # Only Beige Book QA
    uv run scripts/test_v6_vs_rag_comparison.py --alphabet-only    # Only Alphabet QA
    uv run scripts/test_v6_vs_rag_comparison.py --limit 10         # Limit questions per dataset
    uv run scripts/test_v6_vs_rag_comparison.py --concurrency 5    # Parallel queries for V6
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.querying_system.v6 import query_v6, ResearcherConfig

load_dotenv()


# =============================================================================
# Simple RAG Implementation
# =============================================================================

class SimpleRAG:
    """Simple RAG baseline: Embed raw chunks -> FAISS -> retrieve -> synthesize."""

    def __init__(self, chunks_dir: str, top_k: int = 15, doc_filters: list[str] = None):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.top_k = top_k
        self.chunks = []
        self.index = None
        self.doc_filters = doc_filters  # List of doc prefixes to include

        self._load_chunks(chunks_dir)
        self._build_index()

    def _load_chunks(self, chunks_dir: str):
        """Load chunks from JSONL files."""
        print(f"Loading chunks from {chunks_dir}...")
        chunks_path = Path(chunks_dir)

        for jsonl_file in chunks_path.glob("*.jsonl"):
            # Filter by doc prefixes if specified
            if self.doc_filters:
                matches = any(
                    jsonl_file.stem.lower().startswith(prefix.lower())
                    for prefix in self.doc_filters
                )
                if not matches:
                    continue

            with open(jsonl_file) as f:
                for line in f:
                    chunk = json.loads(line)
                    self.chunks.append({
                        "content": chunk.get("body", chunk.get("content", "")),
                        "header_path": chunk.get("header_path", ""),
                        "doc_id": chunk.get("doc_id", jsonl_file.stem),
                    })

        print(f"Loaded {len(self.chunks)} chunks from {self.doc_filters or 'all files'}")

    def _build_index(self):
        """Embed all chunks and build FAISS index."""
        # Create cache filename based on filters
        filter_str = "_".join(sorted(self.doc_filters)) if self.doc_filters else "all"
        cache_file = f"rag_cache_{filter_str}.npz"

        if os.path.exists(cache_file):
            print(f"Loading cached embeddings from {cache_file}...")
            data = np.load(cache_file)
            embeddings_array = data["embeddings"]
        else:
            print(f"Embedding {len(self.chunks)} chunks...")
            texts = [c["content"] for c in self.chunks]

            batch_size = 100
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                batch_embeddings = self.embeddings.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
                print(f"  Embedded {min(i+batch_size, len(texts))}/{len(texts)}")

            embeddings_array = np.array(all_embeddings, dtype=np.float32)
            np.savez(cache_file, embeddings=embeddings_array)
            print(f"Cached embeddings to {cache_file}")

        # Build FAISS index
        dim = embeddings_array.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings_array)
        self.index.add(embeddings_array)
        print(f"Built FAISS index with {self.index.ntotal} vectors")

    def query(self, question: str) -> dict:
        """Query the RAG system."""
        start = time.time()

        # Embed query
        q_embedding = np.array([self.embeddings.embed_query(question)], dtype=np.float32)
        faiss.normalize_L2(q_embedding)

        # Search
        scores, indices = self.index.search(q_embedding, self.top_k)

        # Get chunks
        retrieved = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:
                chunk = self.chunks[idx]
                retrieved.append({
                    "content": chunk["content"],
                    "header": chunk["header_path"],
                    "doc_id": chunk["doc_id"],
                    "score": float(score)
                })

        if not retrieved:
            return {
                "answer": "No relevant information found.",
                "confidence": 0.0,
                "num_evidence": 0,
                "evidence": [],
                "time_ms": int((time.time() - start) * 1000)
            }

        # Format context
        context = "\n\n".join([
            f"[{c['doc_id']} | {c['header']}] (score: {c['score']:.3f})\n{c['content'][:1500]}"
            for c in retrieved
        ])

        # Synthesis
        prompt = f"""Answer the following question using ONLY the provided context.
If the answer is not in the context, say "Information not found."
Be specific and cite document sources when possible.

QUESTION: {question}

CONTEXT:
{context}

ANSWER:"""

        response = self.llm.invoke(prompt)
        answer = response.content

        # Estimate confidence based on top retrieval scores
        avg_score = sum(c["score"] for c in retrieved[:5]) / min(5, len(retrieved))
        confidence = min(1.0, avg_score)  # Scores are cosine similarity [0,1]

        return {
            "answer": answer,
            "confidence": confidence,
            "num_evidence": len(retrieved),
            "evidence": retrieved[:5],  # Top 5 for analysis
            "time_ms": int((time.time() - start) * 1000)
        }


# =============================================================================
# Test Runner
# =============================================================================

@dataclass
class TestResult:
    id: int
    question: str
    expected: str
    predicted: str
    confidence: float
    num_evidence: int
    time_ms: int
    question_type: Optional[str] = None
    error: Optional[str] = None
    evidence: list = field(default_factory=list)


async def run_v6_question(qa: dict, config: ResearcherConfig) -> TestResult:
    """Run a single question through V6 pipeline."""
    qid = qa["id"]
    question = qa["question"]
    expected = qa["answer"]
    qtype = qa.get("type", "unknown")

    start = time.time()
    try:
        result = await query_v6(question, config=config)
        elapsed = int((time.time() - start) * 1000)

        evidence_list = [
            {
                "fact_id": ev.fact_id,
                "content": ev.content,
                "subject": ev.subject,
                "object": ev.object,
                "score": ev.score,
                "source_doc": ev.source_doc,
            }
            for ev in result.evidence[:10]
        ]

        return TestResult(
            id=qid,
            question=question,
            expected=expected,
            predicted=result.answer,
            confidence=result.confidence,
            num_evidence=len(result.evidence),
            time_ms=elapsed,
            question_type=result.question_type or qtype,
            evidence=evidence_list,
        )
    except Exception as e:
        elapsed = int((time.time() - start) * 1000)
        return TestResult(
            id=qid,
            question=question,
            expected=expected,
            predicted="",
            confidence=0.0,
            num_evidence=0,
            time_ms=elapsed,
            question_type=qtype,
            error=str(e),
        )


def run_rag_question(qa: dict, rag: SimpleRAG) -> TestResult:
    """Run a single question through RAG pipeline."""
    qid = qa["id"]
    question = qa["question"]
    expected = qa["answer"]
    qtype = qa.get("type", "unknown")

    try:
        result = rag.query(question)
        return TestResult(
            id=qid,
            question=question,
            expected=expected,
            predicted=result["answer"],
            confidence=result["confidence"],
            num_evidence=result["num_evidence"],
            time_ms=result["time_ms"],
            question_type=qtype,
            evidence=result["evidence"],
        )
    except Exception as e:
        return TestResult(
            id=qid,
            question=question,
            expected=expected,
            predicted="",
            confidence=0.0,
            num_evidence=0,
            time_ms=0,
            question_type=qtype,
            error=str(e),
        )


def calculate_term_coverage(expected: str, predicted: str) -> float:
    """Calculate what % of key terms from expected appear in predicted."""
    stop_words = {
        'about', 'after', 'before', 'being', 'between', 'could', 'during',
        'would', 'should', 'these', 'those', 'their', 'there', 'where',
        'which', 'while', 'other', 'through', 'that', 'this', 'with', 'from',
        'have', 'been', 'were', 'also', 'some', 'more', 'into'
    }

    expected_words = set(
        word.lower().strip('.,;:()"\'')
        for word in expected.split()
        if len(word) > 3 and word.lower() not in stop_words
    )

    if not expected_words:
        return 1.0

    predicted_lower = predicted.lower()
    matches = sum(1 for w in expected_words if w in predicted_lower)
    return matches / len(expected_words)


def compute_metrics(results: list[TestResult]) -> dict:
    """Compute summary metrics for a set of results."""
    successful = [r for r in results if not r.error]
    errors = [r for r in results if r.error]

    if not successful:
        return {
            "total": len(results),
            "successful": 0,
            "errors": len(errors),
            "avg_confidence": 0.0,
            "avg_evidence": 0.0,
            "avg_time_ms": 0,
            "avg_term_coverage": 0.0,
        }

    coverages = [calculate_term_coverage(r.expected, r.predicted) for r in successful]

    return {
        "total": len(results),
        "successful": len(successful),
        "errors": len(errors),
        "avg_confidence": sum(r.confidence for r in successful) / len(successful),
        "avg_evidence": sum(r.num_evidence for r in successful) / len(successful),
        "avg_time_ms": sum(r.time_ms for r in successful) / len(successful),
        "avg_term_coverage": sum(coverages) / len(coverages),
        "coverage_buckets": {
            ">=90%": sum(1 for c in coverages if c >= 0.9),
            ">=70%": sum(1 for c in coverages if c >= 0.7),
            ">=50%": sum(1 for c in coverages if c >= 0.5),
            "<50%": sum(1 for c in coverages if c < 0.5),
        }
    }


async def run_v6_tests(qa_pairs: list[dict], concurrency: int = 3) -> list[TestResult]:
    """Run V6 pipeline on all questions with limited concurrency."""
    config = ResearcherConfig(
        relevance_threshold=0.65,
        enable_gap_expansion=True,
        enable_entity_drilldown=True,
        enable_refinement_loop=True,
    )

    semaphore = asyncio.Semaphore(concurrency)

    async def limited_query(qa):
        async with semaphore:
            print(f"  [V6] Q{qa['id']}: {qa['question'][:50]}...")
            result = await run_v6_question(qa, config)
            status = "✓" if not result.error else "✗"
            print(f"  [V6] Q{qa['id']}: {status} conf={result.confidence:.2f} ev={result.num_evidence} t={result.time_ms}ms")
            return result

    tasks = [limited_query(qa) for qa in qa_pairs]
    return await asyncio.gather(*tasks)


def run_rag_tests(qa_pairs: list[dict], rag: SimpleRAG) -> list[TestResult]:
    """Run RAG pipeline on all questions."""
    results = []
    for qa in qa_pairs:
        print(f"  [RAG] Q{qa['id']}: {qa['question'][:50]}...")
        result = run_rag_question(qa, rag)
        status = "✓" if not result.error else "✗"
        print(f"  [RAG] Q{qa['id']}: {status} conf={result.confidence:.2f} ev={result.num_evidence} t={result.time_ms}ms")
        results.append(result)
    return results


def print_comparison_table(v6_metrics: dict, rag_metrics: dict, dataset_name: str):
    """Print a comparison table for V6 vs RAG."""
    print(f"\n{'='*70}")
    print(f" {dataset_name} - V6 vs RAG Comparison")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'V6':>20} {'RAG':>20}")
    print(f"{'-'*65}")
    print(f"{'Total Questions':<25} {v6_metrics['total']:>20} {rag_metrics['total']:>20}")
    print(f"{'Successful':<25} {v6_metrics['successful']:>20} {rag_metrics['successful']:>20}")
    print(f"{'Errors':<25} {v6_metrics['errors']:>20} {rag_metrics['errors']:>20}")
    print(f"{'Avg Confidence':<25} {v6_metrics['avg_confidence']:>20.2f} {rag_metrics['avg_confidence']:>20.2f}")
    print(f"{'Avg Evidence Count':<25} {v6_metrics['avg_evidence']:>20.1f} {rag_metrics['avg_evidence']:>20.1f}")
    print(f"{'Avg Time (ms)':<25} {v6_metrics['avg_time_ms']:>20.0f} {rag_metrics['avg_time_ms']:>20.0f}")
    print(f"{'Avg Term Coverage':<25} {v6_metrics['avg_term_coverage']:>19.0%} {rag_metrics['avg_term_coverage']:>19.0%}")
    print(f"{'-'*65}")
    print(f"{'Coverage >= 90%':<25} {v6_metrics['coverage_buckets']['>=90%']:>20} {rag_metrics['coverage_buckets']['>=90%']:>20}")
    print(f"{'Coverage >= 70%':<25} {v6_metrics['coverage_buckets']['>=70%']:>20} {rag_metrics['coverage_buckets']['>=70%']:>20}")
    print(f"{'Coverage >= 50%':<25} {v6_metrics['coverage_buckets']['>=50%']:>20} {rag_metrics['coverage_buckets']['>=50%']:>20}")
    print(f"{'Coverage < 50%':<25} {v6_metrics['coverage_buckets']['<50%']:>20} {rag_metrics['coverage_buckets']['<50%']:>20}")


async def main():
    parser = argparse.ArgumentParser(description="V6 vs RAG Comparison Test")
    parser.add_argument("--v6-only", action="store_true", help="Only run V6 tests")
    parser.add_argument("--rag-only", action="store_true", help="Only run RAG tests")
    parser.add_argument("--beige-only", action="store_true", help="Only test Beige Book QA")
    parser.add_argument("--alphabet-only", action="store_true", help="Only test Alphabet QA")
    parser.add_argument("--limit", type=int, default=None, help="Limit questions per dataset")
    parser.add_argument("--concurrency", type=int, default=3, help="V6 parallel queries")
    args = parser.parse_args()

    run_v6 = not args.rag_only
    run_rag = not args.v6_only
    test_beige = not args.alphabet_only
    test_alphabet = not args.beige_only

    # Load QA datasets
    project_root = Path(__file__).parent.parent

    beige_qa = []
    alphabet_qa = []

    if test_beige:
        with open(project_root / "Biege_OA.json") as f:
            data = json.load(f)
            beige_qa = data["qa_pairs"]
            if args.limit:
                beige_qa = beige_qa[:args.limit]
        print(f"Loaded {len(beige_qa)} Beige Book questions")

    if test_alphabet:
        with open(project_root / "Alphabet_QA.json") as f:
            data = json.load(f)
            alphabet_qa = data["qa_pairs"]
            if args.limit:
                alphabet_qa = alphabet_qa[:args.limit]
        print(f"Loaded {len(alphabet_qa)} Alphabet questions")

    # Initialize RAG with both document sets
    rag = None
    if run_rag:
        print("\nInitializing RAG with Alphabet + Beige Book chunks...")
        rag = SimpleRAG(
            chunks_dir=str(project_root / "src" / "chunker" / "SAVED"),
            top_k=15,
            doc_filters=["alphabet", "BeigeBook_20251015"]  # Both docs
        )

    # Results storage
    all_results = {
        "beige": {"v6": [], "rag": []},
        "alphabet": {"v6": [], "rag": []},
    }

    total_start = time.time()

    # =========================
    # Test Beige Book QA
    # =========================
    if test_beige and beige_qa:
        print(f"\n{'='*70}")
        print(" BEIGE BOOK QA TESTS")
        print(f"{'='*70}")

        if run_v6:
            print(f"\n[V6] Running on {len(beige_qa)} Beige Book questions...")
            v6_start = time.time()
            all_results["beige"]["v6"] = await run_v6_tests(beige_qa, args.concurrency)
            print(f"[V6] Beige Book completed in {time.time() - v6_start:.1f}s")

        if run_rag:
            print(f"\n[RAG] Running on {len(beige_qa)} Beige Book questions...")
            rag_start = time.time()
            all_results["beige"]["rag"] = run_rag_tests(beige_qa, rag)
            print(f"[RAG] Beige Book completed in {time.time() - rag_start:.1f}s")

    # =========================
    # Test Alphabet QA
    # =========================
    if test_alphabet and alphabet_qa:
        print(f"\n{'='*70}")
        print(" ALPHABET QA TESTS")
        print(f"{'='*70}")

        if run_v6:
            print(f"\n[V6] Running on {len(alphabet_qa)} Alphabet questions...")
            v6_start = time.time()
            all_results["alphabet"]["v6"] = await run_v6_tests(alphabet_qa, args.concurrency)
            print(f"[V6] Alphabet completed in {time.time() - v6_start:.1f}s")

        if run_rag:
            print(f"\n[RAG] Running on {len(alphabet_qa)} Alphabet questions...")
            rag_start = time.time()
            all_results["alphabet"]["rag"] = run_rag_tests(alphabet_qa, rag)
            print(f"[RAG] Alphabet completed in {time.time() - rag_start:.1f}s")

    total_time = time.time() - total_start

    # =========================
    # Compute and Print Results
    # =========================
    print(f"\n{'='*70}")
    print(" RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Total test time: {total_time:.1f}s")

    # Beige Book comparison
    if test_beige and all_results["beige"]["v6"] and all_results["beige"]["rag"]:
        v6_metrics = compute_metrics(all_results["beige"]["v6"])
        rag_metrics = compute_metrics(all_results["beige"]["rag"])
        print_comparison_table(v6_metrics, rag_metrics, "BEIGE BOOK")
    elif test_beige:
        if all_results["beige"]["v6"]:
            metrics = compute_metrics(all_results["beige"]["v6"])
            print(f"\nBeige Book V6: {metrics['successful']}/{metrics['total']} successful, "
                  f"avg_conf={metrics['avg_confidence']:.2f}, "
                  f"avg_coverage={metrics['avg_term_coverage']:.0%}")
        if all_results["beige"]["rag"]:
            metrics = compute_metrics(all_results["beige"]["rag"])
            print(f"\nBeige Book RAG: {metrics['successful']}/{metrics['total']} successful, "
                  f"avg_conf={metrics['avg_confidence']:.2f}, "
                  f"avg_coverage={metrics['avg_term_coverage']:.0%}")

    # Alphabet comparison
    if test_alphabet and all_results["alphabet"]["v6"] and all_results["alphabet"]["rag"]:
        v6_metrics = compute_metrics(all_results["alphabet"]["v6"])
        rag_metrics = compute_metrics(all_results["alphabet"]["rag"])
        print_comparison_table(v6_metrics, rag_metrics, "ALPHABET")
    elif test_alphabet:
        if all_results["alphabet"]["v6"]:
            metrics = compute_metrics(all_results["alphabet"]["v6"])
            print(f"\nAlphabet V6: {metrics['successful']}/{metrics['total']} successful, "
                  f"avg_conf={metrics['avg_confidence']:.2f}, "
                  f"avg_coverage={metrics['avg_term_coverage']:.0%}")
        if all_results["alphabet"]["rag"]:
            metrics = compute_metrics(all_results["alphabet"]["rag"])
            print(f"\nAlphabet RAG: {metrics['successful']}/{metrics['total']} successful, "
                  f"avg_conf={metrics['avg_confidence']:.2f}, "
                  f"avg_coverage={metrics['avg_term_coverage']:.0%}")

    # =========================
    # Save Detailed Results
    # =========================
    output_file = project_root / "v6_vs_rag_comparison_results.json"

    output_data = {
        "total_time_s": total_time,
        "config": {
            "limit": args.limit,
            "concurrency": args.concurrency,
            "v6_only": args.v6_only,
            "rag_only": args.rag_only,
        },
        "beige_book": {
            "v6": {
                "metrics": compute_metrics(all_results["beige"]["v6"]) if all_results["beige"]["v6"] else None,
                "results": [asdict(r) for r in all_results["beige"]["v6"]]
            },
            "rag": {
                "metrics": compute_metrics(all_results["beige"]["rag"]) if all_results["beige"]["rag"] else None,
                "results": [asdict(r) for r in all_results["beige"]["rag"]]
            }
        },
        "alphabet": {
            "v6": {
                "metrics": compute_metrics(all_results["alphabet"]["v6"]) if all_results["alphabet"]["v6"] else None,
                "results": [asdict(r) for r in all_results["alphabet"]["v6"]]
            },
            "rag": {
                "metrics": compute_metrics(all_results["alphabet"]["rag"]) if all_results["alphabet"]["rag"] else None,
                "results": [asdict(r) for r in all_results["alphabet"]["rag"]]
            }
        }
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
