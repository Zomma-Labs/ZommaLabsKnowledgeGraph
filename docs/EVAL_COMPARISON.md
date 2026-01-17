# QA System Evaluation Comparison

Evaluation on 75 Beige Book questions.

## Results Summary

| System | Strict Accuracy | Lenient Accuracy | Correct | Partial | Incorrect |
|--------|-----------------|------------------|---------|---------|-----------|
| **Two-Agent (KG)** | **41.3%** | 84.0% | 31 | 32 | 12 |
| **Deep Research (KG)** | 36.0% | **88.0%** | 27 | 39 | **9** |
| **Deep Research (RAG)** | 33.3% | 80.0% | 25 | 35 | 15 |

---

## System Descriptions

### Two-Agent (KG)
- **Architecture:** SearchAgent → AnswerAgent
- **Backend:** Knowledge Graph (Neo4j)
- **Avg Time:** ~15s per question

### Deep Research (KG)
- **Architecture:** Supervisor → Parallel Researchers → Synthesizer
- **Backend:** Knowledge Graph (Neo4j)
- **Avg Time:** ~780s per question (with high concurrency)

### Deep Research (RAG)
- **Architecture:** Supervisor → Parallel Researchers → Synthesizer
- **Backend:** Simple vector search over chunks (no graph)
- **Avg Time:** ~679s per question (with high concurrency)

---

## Key Findings

### What the Knowledge Graph Adds
- **+8% strict accuracy** over simple RAG with same orchestration
- Entity attribution - links facts to specific districts/companies
- Relationship traversal - connects related facts across chunks
- Deduplication - same entity mentioned differently gets unified

### What Multi-Agent Orchestration Adds
- Decomposition - breaks complex questions into simpler searches
- Multiple search angles - parallel researchers cast wider net
- Better coverage - fewer totally incorrect answers (12% vs 16%)
- **+4% lenient accuracy** over two-agent approach

### Tradeoffs
- Deep Research finds MORE info but loses some precision in synthesis
- Two-Agent is more direct: search → answer (less information loss)
- Deep Research: better for coverage, worse for precision
- Two-Agent: better for precision, worse for coverage

---

## Where All Systems Struggle

- Very specific details (e.g., "fire at aluminum plant in Kentucky")
- Exact enumeration (e.g., "which retail categories gained vs declined")
- Questions requiring info not well-indexed in the source data
- Cross-referencing multiple criteria (e.g., "districts with BOTH X and Y")

---

## Recommendations

1. **For precision-critical tasks:** Use Two-Agent (KG)
2. **For comprehensive research:** Use Deep Research (KG)
3. **For speed:** Use Two-Agent (KG) - 50x faster
4. **Simple RAG is insufficient:** KG structure adds meaningful value
