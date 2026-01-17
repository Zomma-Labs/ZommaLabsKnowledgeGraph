"""
Fact-RAG: Simple RAG using fact vector search in Neo4j.

Approach:
1. Embed the query
2. Global vector search on fact_embeddings index
3. Map facts back to their source chunks
4. Use chunks for synthesis

This is simpler than V6 - no decomposition, no entity resolution, just global fact search.
"""
import asyncio
import os
import sys
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv

from src.util.neo4j_client import Neo4jClient

load_dotenv()


@dataclass
class FactResult:
    """A fact with its source chunk."""
    fact_id: str
    fact_content: str
    score: float
    subject: str
    object: str
    chunk_id: str
    chunk_content: str
    chunk_header: str
    doc_id: str


class FactRAG:
    """Simple RAG using fact vector search."""

    def __init__(self, group_id: str = "default", top_k_facts: int = 50, top_k_chunks: int = 15):
        self.group_id = group_id
        self.top_k_facts = top_k_facts
        self.top_k_chunks = top_k_chunks

        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.neo4j = Neo4jClient()

        # Warmup Neo4j
        print("Warming up Neo4j connection...")
        self.neo4j.warmup()
        print("Neo4j ready.")

    def search_facts(self, query: str, threshold: float = 0.3) -> list[FactResult]:
        """Search facts by vector similarity and return with their source chunks."""
        # Embed the query
        query_embedding = self.embeddings.embed_query(query)

        # Global vector search on facts with chunk info
        results = self.neo4j.query(
            """
            CALL db.index.vector.queryNodes('fact_embeddings', $top_k, $vec)
            YIELD node, score
            WHERE node.group_id = $uid AND score > $threshold

            // Get the chunk this fact came from via the relationship pattern
            OPTIONAL MATCH (subj)-[r1 {fact_id: node.uuid}]->(c:EpisodicNode {group_id: $uid})
                           -[r2 {fact_id: node.uuid}]->(obj)
            WHERE (subj:EntityNode OR subj:TopicNode) AND (obj:EntityNode OR obj:TopicNode)
              AND subj.group_id = $uid AND obj.group_id = $uid

            OPTIONAL MATCH (d:DocumentNode {group_id: $uid})-[:CONTAINS_CHUNK]->(c)

            RETURN DISTINCT
                node.uuid as fact_id,
                node.content as fact_content,
                score,
                subj.name as subject,
                obj.name as object,
                c.uuid as chunk_id,
                c.content as chunk_content,
                c.header_path as chunk_header,
                d.name as doc_id
            ORDER BY score DESC
            LIMIT $top_k
            """,
            {
                "vec": query_embedding,
                "uid": self.group_id,
                "threshold": threshold,
                "top_k": self.top_k_facts,
            }
        )

        # Convert to FactResult
        facts = []
        seen_facts = set()
        for r in results:
            fact_id = r.get("fact_id")
            if not fact_id or fact_id in seen_facts:
                continue
            seen_facts.add(fact_id)

            facts.append(FactResult(
                fact_id=fact_id,
                fact_content=r.get("fact_content") or "",
                score=r.get("score") or 0.0,
                subject=r.get("subject") or "",
                object=r.get("object") or "",
                chunk_id=r.get("chunk_id") or "",
                chunk_content=r.get("chunk_content") or "",
                chunk_header=r.get("chunk_header") or "",
                doc_id=r.get("doc_id") or "",
            ))

        return facts

    def dedupe_chunks(self, facts: list[FactResult]) -> list[dict]:
        """Dedupe and rank chunks from facts."""
        # Group facts by chunk
        chunk_facts = {}
        for f in facts:
            if not f.chunk_id:
                continue
            if f.chunk_id not in chunk_facts:
                chunk_facts[f.chunk_id] = {
                    "chunk_id": f.chunk_id,
                    "content": f.chunk_content,
                    "header": f.chunk_header,
                    "doc_id": f.doc_id,
                    "facts": [],
                    "max_score": 0.0,
                    "total_score": 0.0,
                }
            chunk_facts[f.chunk_id]["facts"].append(f.fact_content)
            chunk_facts[f.chunk_id]["max_score"] = max(chunk_facts[f.chunk_id]["max_score"], f.score)
            chunk_facts[f.chunk_id]["total_score"] += f.score

        # Sort by total score (chunks with more relevant facts rank higher)
        chunks = sorted(chunk_facts.values(), key=lambda x: x["total_score"], reverse=True)
        return chunks[:self.top_k_chunks]

    def synthesize(self, question: str, chunks: list[dict]) -> str:
        """Synthesize answer from chunks."""
        if not chunks:
            return "No relevant information found in the knowledge graph."

        # Format context with chunk headers and content
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            header = chunk["header"] or "Unknown Section"
            doc = chunk["doc_id"] or "Unknown Document"
            facts_summary = "; ".join(chunk["facts"][:5])  # Top 5 facts from this chunk
            context_parts.append(
                f"[{i}] {doc} > {header}\n"
                f"Relevant facts: {facts_summary}\n"
                f"Full context: {chunk['content'][:2000]}"
            )

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""Answer the following question using ONLY the provided context.
Be thorough and list ALL relevant items found in the context.
If the answer requires listing multiple items (districts, sectors, etc.), list ALL of them.

QUESTION: {question}

CONTEXT:
{context}

ANSWER:"""

        response = self.llm.invoke(prompt)
        return response.content

    def query(self, question: str) -> dict:
        """Full RAG pipeline: search facts -> dedupe chunks -> synthesize."""
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}")

        # 1. Search facts
        print("\n1. Searching facts...")
        facts = self.search_facts(question)
        print(f"   Found {len(facts)} relevant facts")

        # Show top facts
        print("\n   Top 10 facts:")
        for f in facts[:10]:
            print(f"   - [{f.score:.3f}] {f.fact_content[:80]}...")

        # 2. Dedupe to chunks
        print("\n2. Deduping to chunks...")
        chunks = self.dedupe_chunks(facts)
        print(f"   Using {len(chunks)} unique chunks")

        # Show chunk sources
        print("\n   Chunk sources:")
        for c in chunks:
            print(f"   - {c['doc_id']} > {c['header'][:50]}... ({len(c['facts'])} facts, score={c['total_score']:.2f})")

        # 3. Synthesize
        print("\n3. Synthesizing answer...")
        answer = self.synthesize(question, chunks)

        print(f"\n{'='*60}")
        print("ANSWER:")
        print(f"{'='*60}")
        print(answer)

        return {
            "question": question,
            "answer": answer,
            "num_facts": len(facts),
            "num_chunks": len(chunks),
            "facts": [
                {"content": f.fact_content, "score": f.score, "subject": f.subject, "object": f.object}
                for f in facts[:20]
            ],
            "chunks": [
                {"header": c["header"], "doc_id": c["doc_id"], "num_facts": len(c["facts"])}
                for c in chunks
            ],
        }

    def close(self):
        """Clean up connections."""
        self.neo4j.close()


async def test_q52():
    """Test on Q52 with query decomposition - uses CHUNK CONTENT for evidence matching."""
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel, Field

    class DecomposedQuery(BaseModel):
        sub_queries: list[str] = Field(description="List of simpler sub-queries to search for")

    # Decompose the question
    decomposer = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(DecomposedQuery)

    question = "Which districts mentioned both tariff impacts on manufacturing AND labor shortages from immigration policies?"

    print(f"\nOriginal question: {question}")
    print("\nDecomposing question...")

    decomposed = decomposer.invoke(
        f"Break this question into 2-3 simpler search queries:\n{question}"
    )

    print(f"Sub-queries: {decomposed.sub_queries}")

    rag = FactRAG(
        group_id="default",
        top_k_facts=100,
        top_k_chunks=25,
    )

    try:
        # Search for each sub-query and collect chunks by district
        # Use CHUNK CONTENT to check for keywords, not extracted facts
        district_chunks = {}  # district -> list of chunks

        for sq in decomposed.sub_queries:
            print(f"\n--- Searching: {sq} ---")
            facts = rag.search_facts(sq)
            chunks = rag.dedupe_chunks(facts)

            for c in chunks:
                header = c["header"]
                if "Federal Reserve Bank of" in header:
                    district = header.split("Federal Reserve Bank of")[1].split(">")[0].strip()
                    if district not in district_chunks:
                        district_chunks[district] = []
                    # Store the full chunk content
                    district_chunks[district].append({
                        "header": c["header"],
                        "content": c["content"],  # FULL CHUNK CONTENT
                    })

            print(f"   Found {len(chunks)} chunks")

        # Now check CHUNK CONTENT for both conditions
        print("\n" + "="*60)
        print("ANALYZING CHUNK CONTENT FOR EACH DISTRICT")
        print("="*60)

        both_districts = []
        for district, chunks in district_chunks.items():
            # Combine all chunk content for this district
            all_content = " ".join(c["content"].lower() for c in chunks)

            has_tariff = "tariff" in all_content
            has_immigration = "immigration" in all_content or "visa" in all_content

            if has_tariff and has_immigration:
                both_districts.append(district)
                print(f"\n{district}: ✓ BOTH")
                # Find specific evidence from chunks
                for c in chunks:
                    content_lower = c["content"].lower()
                    if "tariff" in content_lower:
                        # Extract sentence with tariff
                        for sent in c["content"].split("."):
                            if "tariff" in sent.lower():
                                print(f"  Tariff: {sent.strip()[:80]}...")
                                break
                    if "immigration" in content_lower or "visa" in content_lower:
                        for sent in c["content"].split("."):
                            if "immigration" in sent.lower() or "visa" in sent.lower():
                                print(f"  Immigration: {sent.strip()[:80]}...")
                                break
            else:
                status = []
                if has_tariff:
                    status.append("tariff")
                if has_immigration:
                    status.append("immigration")
                print(f"\n{district}: only {', '.join(status) if status else 'neither'}")

        print("\n" + "="*60)
        print("COMPARISON")
        print("="*60)
        print(f"\nExpected (QA file):   St. Louis, Chicago")
        print(f"NotebookLM found:     Chicago, St. Louis, San Francisco, Philadelphia, Atlanta, New York")
        print(f"Fact-RAG found:       {', '.join(sorted(both_districts)) if both_districts else 'None'}")

        return {"both_districts": sorted(both_districts)}
    finally:
        rag.close()


if __name__ == "__main__":
    asyncio.run(test_q52())
