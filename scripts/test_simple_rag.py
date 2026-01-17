"""
Simple RAG baseline: Embed raw chunks -> FAISS -> retrieve -> synthesize.
No decomposition, no fact extraction - just standard RAG.
"""
import json
import time
import os
from pathlib import Path
import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()

class SimpleRAG:
    def __init__(self, chunks_dir: str, top_k: int = 10, filter_prefix: str = None):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.top_k = top_k
        self.chunks = []
        self.index = None
        self.filter_prefix = filter_prefix

        # Load and index chunks
        self._load_chunks(chunks_dir, filter_prefix)
        self._build_index()

    def _load_chunks(self, chunks_dir: str, filter_prefix: str = None):
        """Load chunks from JSONL files, optionally filtering by filename prefix."""
        print(f"Loading chunks from {chunks_dir}...")
        chunks_path = Path(chunks_dir)

        for jsonl_file in chunks_path.glob("*.jsonl"):
            # Filter by prefix if specified
            if filter_prefix and not jsonl_file.stem.lower().startswith(filter_prefix.lower()):
                continue

            with open(jsonl_file) as f:
                for line in f:
                    chunk = json.loads(line)
                    self.chunks.append({
                        "content": chunk.get("body", chunk.get("content", "")),
                        "header_path": chunk.get("header_path", ""),
                        "doc_id": chunk.get("doc_id", jsonl_file.stem),
                    })

        print(f"Loaded {len(self.chunks)} chunks")

    def _build_index(self):
        """Embed all chunks and build FAISS index."""
        cache_suffix = f"_{self.filter_prefix}" if self.filter_prefix else ""
        cache_file = f"simple_rag_faiss_cache{cache_suffix}.npz"

        if os.path.exists(cache_file):
            print("Loading cached embeddings...")
            data = np.load(cache_file)
            embeddings_array = data["embeddings"]
        else:
            print(f"Embedding {len(self.chunks)} chunks (this may take a while)...")
            texts = [c["content"] for c in self.chunks]

            # Batch embed
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
        self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine with normalized vectors)

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings_array)
        self.index.add(embeddings_array)
        print(f"Built FAISS index with {self.index.ntotal} vectors")

    def query(self, question: str) -> str:
        # 1. Embed the question
        q_embedding = np.array([self.embeddings.embed_query(question)], dtype=np.float32)
        faiss.normalize_L2(q_embedding)

        # 2. Search FAISS
        scores, indices = self.index.search(q_embedding, self.top_k)

        # 3. Get chunks
        retrieved = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:  # Valid index
                chunk = self.chunks[idx]
                retrieved.append({
                    "content": chunk["content"],
                    "header": chunk["header_path"],
                    "score": float(score)
                })

        if not retrieved:
            return "No relevant information found."

        # 4. Format context
        context = "\n\n".join([
            f"[{c['header']}]\n{c['content'][:2000]}"
            for c in retrieved
        ])

        # 5. Simple synthesis prompt
        prompt = f"""Answer the following question using ONLY the provided context.
If the answer is not in the context, say "Information not found."

QUESTION: {question}

CONTEXT:
{context}

ANSWER:"""

        response = self.llm.invoke(prompt)
        return response.content


def main():
    # Load questions
    with open("Alphabet_QA.json") as f:
        data = json.load(f)

    questions = data["qa_pairs"]

    # Initialize RAG with Alphabet chunks only
    rag = SimpleRAG(
        chunks_dir="src/chunker/SAVED",
        top_k=15,
        filter_prefix="alphabet"  # Only use alphabet.jsonl
    )

    results = []
    total_start = time.time()

    print(f"\nTesting Simple RAG on {len(questions)} questions...")
    print("="*80)

    for q in questions:
        qid = q["id"]
        question = q["question"]
        expected = q["answer"]

        start = time.time()
        try:
            predicted = rag.query(question)
            elapsed = int((time.time() - start) * 1000)
            error = None
        except Exception as e:
            predicted = f"ERROR: {e}"
            elapsed = int((time.time() - start) * 1000)
            error = str(e)

        results.append({
            "id": qid,
            "question": question,
            "expected": expected,
            "predicted": predicted,
            "time_ms": elapsed,
            "error": error
        })

        print(f"Q{qid}: {question[:60]}...")
        print(f"  Expected: {expected[:80]}...")
        print(f"  Predicted: {predicted[:80]}...")
        print(f"  Time: {elapsed}ms")
        print()

    total_time = time.time() - total_start

    # Save results
    with open("simple_rag_results.json", "w") as f:
        json.dump({
            "total_time_s": total_time,
            "results": results
        }, f, indent=2)

    print("="*80)
    print(f"Total time: {total_time:.1f}s")
    print(f"Results saved to simple_rag_results.json")


if __name__ == "__main__":
    main()
