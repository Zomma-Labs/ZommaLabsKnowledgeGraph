# Financial GraphRAG Ingestion Workflow

This document outlines the step-by-step workflow of the ZommaLabs Financial Knowledge Graph ingestion pipeline (`main_pipeline.py`). It details the sequence of LLM calls, embedding operations, and graph interactions.

## High-Level Overview

1.  **Header Analysis**: Extracts document structure and creates "Container" nodes.
2.  **Atomization**: Decomposes text into atomic, self-contained facts.
3.  **Entity Extraction**: Identifies subjects, objects, and topics from facts.
4.  **Parallel Resolution**: Resolves entities against FIBO and the Graph (Optimized).
5.  **Relationship Classification**: Classifies the interaction type (e.g., Acquisition).
6.  **Causal Linking**: Links facts causally (e.g., Inflation -> Rates Up).
7.  **Graph Assembly**: Writes nodes and edges to Neo4j.

---

## Detailed Steps

### 1. Initialize Episode & Header Analysis
**Goal**: Establish the "Where" (Document Context) and create structural nodes.

*   **Context Extraction (LLM)**: `HeaderAnalyzer` reads the first few chunks to determine the document type (e.g., "Federal Reserve Beige Book").
*   **Dimension Analysis (LLM)**: `HeaderAnalyzer` parses the breadcrumbs (e.g., `Regional Reports > New York`) to split them into **Topics** or **Entities**.
*   **Resolution (Embedding)**: Each dimension is resolved (FIBO/Graph) and created as a node.
    *   *Edge*: `(TopicNode)-[:ABOUT]->(EpisodicNode)` creates the structural link.

### 2. Atomization (Decomposition)
**Goal**: Break complex text into single-event statements.

*   **Decomposition (LLM)**: `Atomizer` splits the chunk into "Propositions".
    *   *Task*: Resolves pronouns ("The district" -> "The Minneapolis District").
    *   *Task*: Grounds relative time ("Last year" -> "2023").
*   **Reflexion (LLM)**: `GraphEnhancer` critiques the facts against the source text.
    *   *Task*: Checks for missed facts or concepts that should be promoted to specific entities (e.g., "The tech giant" -> "Google").

### 3. Context-Aware Entity Extraction
**Goal**: Extract structured relationships from the text.

*   **Extraction (LLM)**: `EntityExtractor` processes each Atomic Fact.
    *   *Context*: Uses the **Header Path** to resolve ambiguous terms immediately (e.g., "The Sector" -> "Manufacturing").
    *   *Output*: One or more `FinancialRelation` objects (Subject, Object, Topics).
*   **Reflexion (LLM)**: Critiques the output to split aggregates ("Contacts in 3 districts" -> 3 relations).

### 4. Parallel Resolution (Optimized Core Loop)
**Goal**: Deduplicate entities against the Ontology (FIBO) and the Private Graph.
*Triggered in parallel for every unique Subject, Object, and Topic.*

1.  **Summary Extraction (LLM)**: 
    *   We ask the LLM to define *what* the entity is in this specific context (e.g., "Apple" -> "A multinational technology company").
    
2.  **Embedding Generation (Single Pass)**:
    *   We generate a single embedding for `Name: Summary`.
    *   *Optimization*: This vector is computed **once** and reused for Locking, FIBO, Graph Search, and Node Creation.

3.  **Similarity Locking**:
    *   We acquire a lock using the `Name: Summary` vector.
    *   *Purpose*: Prevents race conditions where two threads try to create "Apple Inc" simultaneously.

4.  **FIBO Resolution**:
    *   **Vector Search**: Uses the cached embedding to search the **FIBO Qdrant** index.
    *   **Result**: If a match is found (e.g., `fibo:AppleInc`), we use that standardized URI.

5.  **Graph Deduplication**:
    *   **Vector Search**: Uses the cached embedding to search the **Neo4j Node Index** for existing private nodes.
    *   **Match Verification (LLM)**: If candidates are found, an LLM compares the New Entity vs. Candidate (using Summaries) to decide: **MERGE** or **CREATE NEW**.

6.  **Node Creation**:
    *   If **CREATE NEW**: We write the `EntityNode`/`TopicNode` to Neo4j, using the cached embedding for the `embedding` property.

### 5. Relationship Classification
**Goal**: Determine the specific edge type using Iterative RAG.

*   **Initial Description (LLM)**: Analyzes the fact and generates a verbose specific description (e.g., "Microsoft generated a profit of... resulting in financial gain").
*   **Iterative Search Loop (Attempts 1-3)**:
    1.  **Vector Search**: Embeds the current description/query to find candidate `RelationshipType` definitions.
    2.  **Evaluation (LLM)**: Checks if any candidate matches with **Confidence > 0.7**.
    3.  **Refinement**: If no good match is found, the LLM generates a **Refined Query** (e.g., "Corporate financial performance reporting") and retries.
    *   *Logs*: You will see "Attempt 1: Querying...", "Refining query to...", "Attempt 2...".
*   **Fallback**: If max retries are reached without high confidence, returns a default or generic relationship.

### 6. Causal Linking
**Goal**: Capture cause-and-effect chains.

*   **Causality Check (LLM)**: Reviews the list of Atomic Facts relative to the source text.
*   **Output**: Pairs of `(Cause_Index, Effect_Index)` (e.g., Fact 1 causes Fact 3).

### 7. Graph Assembly
**Goal**: Write the final structure to Neo4j.

*   **Fact Deduplication**:
    *   **Vector Search**: Embeds the **Fact Text** to check if this exact event already exists.
    *   **Verification (LLM)**: If a similar fact exists, confirms if they are 100% identical events.
*   **Writing**:
    *   Creates/Merges `FactNode`.
    *   Links `(Entity)-[Active_Edge]->(FactNode)`. (Note: Pipeline writes Fact->Episode, and Entity->Episode via Fact, specific schemas may vary).
    *   Current Schema Pattern:
        *   `FactNode -[:MENTIONED_IN]-> EpisodicNode`
        *   `Subject -[:Active_Edge {fact_id: ...}]-> EpisodicNode`
        *   `EpisodicNode -[:Passive_Edge {fact_id: ...}]-> Object`
        *   `FactNode -[:CAUSES]-> FactNode`

---

## Performance Notes

*   **LLM Intensity**: High. Requires multiple calls per chunk (Atomizer, Extractor, Resolution x Entities).
*   **Embedding Optimization**: The "Summary First" pattern ensures we only embed each entity once per processing cycle, significantly reducing latency and API costs.
