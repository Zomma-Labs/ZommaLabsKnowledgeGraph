# Knowledge Graph for ZommaLabsKG 

ZommaLabsKG is an ingestion pipeline that transforms messy PDFs (10-Ks, Earnings Calls, Press Releases) into a Strict, Typed Knowledge Graph. We provide Deterministic Grounding for your Agents:

- FIBO-Standardized Entities: Every company is resolved to its unique Financial Industry Business Ontology ID.
- Atomic Facts: Text is broken into single-sentence propositions with precise time-stamping.
- Strict Edges: Relationships are classified into a fixed enum (e.g., ACQUIRED, SUED), making them SQL/CYPHER-queryable.

## Diagram
![Knowledge Graph Diagram](images/diagram.png) 

### The Pipeline Flow
1. Hierarchical Ingestion: Documents are split by Headers/Sections to preserve context (e.g., separating "Risk Factors" from "Financial Results").
2. Atomic Fact Extraction: Chunks are atomized into single-sentence "Propositions" (Fact Nodes) to ensure granular citation.
3. Parallel Resolution (The "Split Brain"):
    1. Agent A (The Librarian): Resolves Entities (Nouns) against the FIBO Ontology using a Hybrid Vector + Fuzzy Search (handling typos like "Berkshire Hathway").
    2. Agent B (The Analyst): Classifies Relationships (Verbs) against a strict Semantic Enum (e.g., ACQUIRED, SUED) to prevent graph noise.
4. Graph Assembly: Merges Entities and Facts into the target schema.
5. Self-Correction Loop: If Neo4j rejects a write (e.g., "Entity Constraint Violation"), a Feedback Agent repairs the data and retries.

### System Architecture
The pipeline uses a Parallel Agentic Workflow (powered by LangGraph) to ensure high fidelity.

1. The "Atomizer" (Ingestion)
    - Splits documents by Section Headers (preserving context like "Risk Factors").
    - Explodes paragraphs into Atomic Facts (standalone sentences).

    Value for PMs: You get granular lineage. Click a fact, see the exact source sentence.

2. The "Dual-Brain" Resolver (Transformation)
    - We split processing into two parallel streams to maximize accuracy:
    - Stream A (Nouns): Resolves entities against the FIBO Vector Index.
        - Input: "Buffett"
        - Output: fibo-person:Warren_E_Buffett (URI)
    - Stream B (Verbs): Maps actions to our Strict Edge Taxonomy.
        - Input: "Snapped up shares"
        - Output: INVESTED_IN (Edge Type)

3. The "Self-Healing" Assembler (Loading)
    - Writes to Neo4j.
    - Feedback Loop: If the graph rejects an edge (e.g., "Entity not found"), a Repair Agent fixes the data and retries automatically.

### To prevent "Schema Drift," the LLM is restricted to a specific set of high-value verbs. EX:
- Corporate: ACQUIRED, MERGED_WITH, SPUN_OFF, INVESTED_IN
- Legal: SUED, FINED, INVESTIGATED_BY
- People: WORKS_FOR, FIRED, RESIGNED_FROM


### Directory Structure
src/
├── agents/                     # The "Brains" (Logic & Prompts)
│   ├── init.py
│   ├── router.py               # The "Traffic Cop" (Topic Classifier)
│   ├── atomizer.py             # The "Atomic Fact" Extractor
│   ├── entity_resolver.py      # The "Librarian" (FIBO Vector Search)
│   ├── relationship_matcher.py # The "Analyst" (Edge Classification)
│   └── graph_assembler.py      # The "Builder" (Neo4j Writer + Feedback)
│
├── schemas/                    # The "Contracts" (Pydantic Models)
│   ├── init.py
│   ├── atomic_fact.py          # The output format for the Atomizer
│   ├── financial_events.py     # The specific schemas (Earnings, Labor)
│   ├── fibo.py                 # The FIBO Entity definitions
│   └── graph_edge.py           # The strict Enum for Relationships
│
├── workflows/                  # The "Orchestration" (LangGraph)
│   ├── init.py
│   ├── main_pipeline.py        # Defines the StateGraph, Nodes, and Edges
│   └── subgraphs/              # (Optional) If you have complex sub-loops
│
├── tools/                      # The "Hands" (External Interactions)
│   ├── init.py
│   ├── vector_db.py            # Qdrant/Pinecone Client
│   ├── neo4j_client.py         # Neo4j Driver wrapper
│   └── fibo_loader.py          # Scripts to ingest FIBO RDF into Vector DB
│
├── config/                     # Configuration
│   ├── settings.py             # Env vars (API Keys, URI)
│   └── prompts/                # (Optional) Text files if prompts get huge
│       ├── atomizer_system.txt
│       └── resolver_system.txt
│
└── main.py                     # Entry point (API or CLI)