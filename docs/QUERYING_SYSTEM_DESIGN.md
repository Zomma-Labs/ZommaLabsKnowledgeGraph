# GNN-Inspired Querying System Design

## Executive Summary

This document describes a querying system that applies Graph Neural Network principles using LLMs and embeddings instead of trained neural networks. The system performs **iterative, structure-aware retrieval** that mimics message passing, attention mechanisms, and multi-hop aggregation.

---

## Core Concepts: GNN → LLM Translation

| GNN Concept | LLM Implementation |
|-------------|-------------------|
| Message Passing | Follow graph edges, retrieve neighbor nodes |
| Neighborhood Aggregation | LLM summarizes/filters retrieved context |
| Attention Weights | Embedding similarity + LLM relevance scoring |
| Multi-hop Layers | Iterative expansion with early stopping |
| Node Embeddings | Pre-computed vector embeddings (Voyage) |
| Graph Structure | Explicit edge types in Neo4j |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              QUERY INTERFACE                                 │
│                                                                             │
│  User Query: "What caused inflation concerns in the Boston district         │
│               and how did it affect hiring decisions?"                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 1: QUERY UNDERSTANDING                                               │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  LLM decomposes query into structured search plan:                          │
│                                                                             │
│  {                                                                          │
│    "entities": ["Boston district", "inflation"],                            │
│    "topics": ["Inflation", "Labor Markets", "Hiring"],                      │
│    "relationships": ["caused", "affected"],                                 │
│    "intent": "causal_chain",                                                │
│    "time_context": null,                                                    │
│    "multi_hop_required": true,                                              │
│    "expected_hops": 2                                                       │
│  }                                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 2: SEED RETRIEVAL (Parallel)                                         │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │ Entity Search   │  │ Fact Search     │  │ Topic Search    │              │
│  │                 │  │                 │  │                 │              │
│  │ Vector:         │  │ Vector:         │  │ Vector:         │              │
│  │ entity_name_    │  │ fact_embeddings │  │ topic_embeddings│              │
│  │ embeddings      │  │                 │  │                 │              │
│  │                 │  │ Fulltext:       │  │                 │              │
│  │ Fulltext:       │  │ fact_fulltext   │  │                 │              │
│  │ entity_fulltext │  │                 │  │                 │              │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘              │
│           │                    │                    │                       │
│           └────────────────────┼────────────────────┘                       │
│                                ▼                                            │
│                    ┌───────────────────────┐                                │
│                    │  Seed Node Pool       │                                │
│                    │  (deduplicated)       │                                │
│                    └───────────────────────┘                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 3: STRUCTURE-AWARE EXPANSION (Message Passing)                       │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  For each seed node, follow typed edges:                                    │
│                                                                             │
│  EntityNode seeds:                                                          │
│    ──[*]──> EpisodicNode ──[*_TARGET]──> EntityNode (objects)               │
│    <──[*]── EpisodicNode <──[*_TARGET]── EntityNode (subjects)              │
│                                                                             │
│  FactNode seeds:                                                            │
│    <──[CONTAINS_FACT]── EpisodicNode (source chunk)                         │
│    ──> Related entities via chunk                                           │
│                                                                             │
│  TopicNode seeds:                                                           │
│    <──[DISCUSSES]── EpisodicNode (chunks discussing topic)                  │
│    ──> Co-discussed topics                                                  │
│                                                                             │
│  EpisodicNode (always retrieved):                                           │
│    ──[CONTAINS_FACT]──> FactNode (all facts in chunk)                       │
│    ──[DISCUSSES]──> TopicNode (all topics)                                  │
│    <──[CONTAINS_CHUNK]── DocumentNode (parent doc)                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 4: ATTENTION SCORING                                                 │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Two-stage scoring:                                                         │
│                                                                             │
│  Stage A: Fast Embedding Similarity (filter)                                │
│    score_embed = cosine(query_embedding, node_embedding)                    │
│    Keep if score_embed > threshold (e.g., 0.3)                              │
│                                                                             │
│  Stage B: LLM Relevance Scoring (rank)                                      │
│    For top candidates from Stage A:                                         │
│    score_llm = LLM.score(query, node_content, relationship_context)         │
│    Returns: 0.0-1.0 relevance + reasoning                                   │
│                                                                             │
│  Combined score = α * score_embed + (1-α) * score_llm                       │
│                                                                             │
│  Output: Ranked list of (node, score, path_from_seed)                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 5: ITERATIVE DEEPENING (Multi-hop Control)                           │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  while hop < max_hops and not sufficient_context:                           │
│                                                                             │
│    # Check if we have enough                                                │
│    sufficient = LLM.check_sufficiency(                                      │
│      query=original_query,                                                  │
│      context=accumulated_context,                                           │
│      intent=query_intent                                                    │
│    )                                                                        │
│                                                                             │
│    if sufficient:                                                           │
│      break                                                                  │
│                                                                             │
│    # Generate expansion queries for next hop                                │
│    expansion_queries = LLM.generate_expansions(                             │
│      query=original_query,                                                  │
│      current_context=accumulated_context,                                   │
│      missing_info=what_we_still_need                                        │
│    )                                                                        │
│                                                                             │
│    # Expand from high-scoring nodes                                         │
│    frontier = top_k_nodes(scored_nodes, k=beam_width)                       │
│    for node in frontier:                                                    │
│      neighbors = expand_neighbors(node)  # Layer 3 again                    │
│      scored_neighbors = score_nodes(neighbors)  # Layer 4 again             │
│      accumulated_context.extend(scored_neighbors)                           │
│                                                                             │
│    hop += 1                                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 6: CONTEXT AGGREGATION                                               │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Deduplicate and structure retrieved information:                           │
│                                                                             │
│  {                                                                          │
│    "entities": [                                                            │
│      {"name": "Boston District", "summary": "...", "relevance": 0.95}       │
│    ],                                                                       │
│    "facts": [                                                               │
│      {                                                                      │
│        "content": "Rising input costs drove inflation concerns",            │
│        "source_chunk": "uuid-123",                                          │
│        "subject": "Input costs",                                            │
│        "object": "Inflation concerns",                                      │
│        "relationship": "DROVE",                                             │
│        "relevance": 0.92                                                    │
│      }                                                                      │
│    ],                                                                       │
│    "chunks": [                                                              │
│      {                                                                      │
│        "uuid": "uuid-123",                                                  │
│        "content": "...",                                                    │
│        "header_path": "Boston > Economic Activity > Inflation",             │
│        "document_date": "2024-10-15"                                        │
│      }                                                                      │
│    ],                                                                       │
│    "traversal_paths": [                                                     │
│      "Boston District -[EXPERIENCED]-> chunk-1 -[EXPERIENCED_TARGET]-> Inflation" │
│    ]                                                                        │
│  }                                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 7: SYNTHESIS                                                         │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  LLM generates final answer with:                                           │
│    - Direct answers to the query                                            │
│    - Citations to source chunks (header_path, document_date)                │
│    - Confidence level based on evidence strength                            │
│    - Traversal explanation (how facts connect)                              │
│                                                                             │
│  Output:                                                                    │
│  "Inflation concerns in the Boston district were primarily driven by        │
│   rising input costs and supply chain disruptions [Boston > Economic        │
│   Activity, Oct 2024]. This led employers to slow hiring as they            │
│   faced margin pressure [Boston > Labor Markets, Oct 2024]. Specifically,   │
│   manufacturing firms reported delaying expansion plans due to              │
│   uncertainty about future costs [Boston > Manufacturing, Oct 2024]."       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Component Specifications

### Component 1: Query Understanding

**Purpose:** Decompose natural language query into structured search plan.

**Input:** Raw user query string

**Output:**
```python
@dataclass
class QueryPlan:
    # Extracted elements
    entities: list[str]           # Named entities to search
    topics: list[str]             # Topic concepts to search
    relationships: list[str]      # Relationship types of interest

    # Query classification
    intent: Literal[
        "factual",           # Simple fact lookup
        "causal_chain",      # A caused B caused C
        "comparison",        # Compare X and Y
        "temporal",          # How did X change over time
        "aggregation",       # Summarize across multiple sources
        "exploration"        # Open-ended discovery
    ]

    # Search parameters
    time_range: tuple[date, date] | None
    group_id: str | None          # Tenant filter

    # Traversal hints
    multi_hop_required: bool
    expected_hops: int            # Estimated depth needed
    beam_width: int               # How many paths to explore
```

**LLM Prompt:**
```
You are a query analyzer for a knowledge graph about financial documents.

Given a user query, extract:
1. ENTITIES: Specific named entities mentioned (companies, people, places, districts)
2. TOPICS: Abstract concepts/themes (Inflation, Labor Markets, Risk, etc.)
3. RELATIONSHIPS: What connections the user is asking about (caused, affected, led to)
4. INTENT: What type of answer is needed
5. TRAVERSAL DEPTH: How many hops through the graph are likely needed

Query: {query}

Respond in JSON format:
{schema}
```

**Implementation Notes:**
- Use fast model (gemini-flash) for query understanding
- Cache common query patterns
- Fall back to keyword extraction if LLM fails

---

### Component 2: Seed Retrieval

**Purpose:** Find initial entry points into the graph using vector and fulltext search.

**Strategy:** Run multiple searches in parallel, then merge results.

**Search Types:**

#### 2a. Entity Vector Search
```cypher
// Find entities by semantic similarity
CALL db.index.vector.queryNodes(
  'entity_name_embeddings',
  $top_k,
  $query_embedding
) YIELD node, score
WHERE node.group_id = $group_id
RETURN node.uuid, node.name, node.summary, score
ORDER BY score DESC
```

#### 2b. Entity Fulltext Search (Backup)
```cypher
// Find entities by keyword match
CALL db.index.fulltext.queryNodes(
  'entity_fulltext',
  $query_text
) YIELD node, score
WHERE node.group_id = $group_id
RETURN node.uuid, node.name, node.summary, score
ORDER BY score DESC
LIMIT $top_k
```

#### 2c. Fact Vector Search
```cypher
// Find facts by semantic similarity
CALL db.index.vector.queryNodes(
  'fact_embeddings',
  $top_k,
  $query_embedding
) YIELD node, score
WHERE node.group_id = $group_id
RETURN node.uuid, node.content, score
ORDER BY score DESC
```

#### 2d. Fact Fulltext Search
```cypher
CALL db.index.fulltext.queryNodes(
  'fact_fulltext',
  $query_text
) YIELD node, score
WHERE node.group_id = $group_id
RETURN node.uuid, node.content, score
ORDER BY score DESC
LIMIT $top_k
```

#### 2e. Topic Vector Search
```cypher
CALL db.index.vector.queryNodes(
  'topic_embeddings',
  $top_k,
  $query_embedding
) YIELD node, score
WHERE node.group_id = $group_id
RETURN node.uuid, node.name, score
ORDER BY score DESC
```

#### 2f. Chunk Fulltext Search (for direct text matching)
```cypher
CALL db.index.fulltext.queryNodes(
  'chunk_fulltext',
  $query_text
) YIELD node, score
WHERE node.group_id = $group_id
RETURN node.uuid, node.content, node.header_path, score
ORDER BY score DESC
LIMIT $top_k
```

**Merge Strategy:**
```python
def merge_seed_results(
    entity_results: list[ScoredNode],
    fact_results: list[ScoredNode],
    topic_results: list[ScoredNode],
    chunk_results: list[ScoredNode],
    query_plan: QueryPlan
) -> list[ScoredNode]:
    """
    Merge and deduplicate results from multiple searches.
    Weight by query intent.
    """
    weights = {
        "factual": {"entity": 0.3, "fact": 0.5, "topic": 0.1, "chunk": 0.1},
        "causal_chain": {"entity": 0.2, "fact": 0.6, "topic": 0.1, "chunk": 0.1},
        "exploration": {"entity": 0.25, "fact": 0.25, "topic": 0.25, "chunk": 0.25},
        # ... other intents
    }

    intent_weights = weights.get(query_plan.intent, weights["factual"])

    all_nodes = []
    for node in entity_results:
        node.adjusted_score = node.score * intent_weights["entity"]
        all_nodes.append(node)
    # ... repeat for other types

    # Deduplicate by uuid, keep highest score
    seen = {}
    for node in sorted(all_nodes, key=lambda n: n.adjusted_score, reverse=True):
        if node.uuid not in seen:
            seen[node.uuid] = node

    return list(seen.values())[:top_k_total]
```

---

### Component 3: Structure-Aware Expansion (Message Passing)

**Purpose:** Follow graph edges from seed nodes to discover related information.

**Key Insight:** Different node types have different expansion patterns.

#### 3a. Entity Expansion
```cypher
// Find all relationships where entity is SUBJECT
MATCH (e:EntityNode {uuid: $entity_uuid})-[r]->(chunk:EpisodicNode)-[r2]->(obj:EntityNode)
WHERE type(r2) = type(r) + '_TARGET'
  AND chunk.group_id = $group_id
RETURN
  type(r) as relationship_type,
  r.description as relationship_description,
  r.fact_id as fact_id,
  r.date_context as date_context,
  chunk.uuid as chunk_uuid,
  chunk.content as chunk_content,
  chunk.header_path as header_path,
  obj.uuid as object_uuid,
  obj.name as object_name,
  obj.summary as object_summary

UNION

// Find all relationships where entity is OBJECT
MATCH (subj:EntityNode)-[r]->(chunk:EpisodicNode)-[r2]->(e:EntityNode {uuid: $entity_uuid})
WHERE type(r2) = type(r) + '_TARGET'
  AND chunk.group_id = $group_id
RETURN
  type(r) as relationship_type,
  r.description as relationship_description,
  r.fact_id as fact_id,
  r.date_context as date_context,
  chunk.uuid as chunk_uuid,
  chunk.content as chunk_content,
  chunk.header_path as header_path,
  subj.uuid as subject_uuid,
  subj.name as subject_name,
  subj.summary as subject_summary
```

#### 3b. Fact Expansion
```cypher
// Find the chunk containing this fact, and all related facts/entities
MATCH (f:FactNode {uuid: $fact_uuid})<-[:CONTAINS_FACT]-(chunk:EpisodicNode)
WHERE chunk.group_id = $group_id

// Get all other facts in the same chunk
OPTIONAL MATCH (chunk)-[:CONTAINS_FACT]->(other_fact:FactNode)
WHERE other_fact.uuid <> $fact_uuid

// Get all entities connected through this chunk
OPTIONAL MATCH (subj:EntityNode)-[r1]->(chunk)-[r2]->(obj:EntityNode)
WHERE type(r2) = type(r1) + '_TARGET'

// Get topics discussed
OPTIONAL MATCH (chunk)-[:DISCUSSES]->(topic:TopicNode)

RETURN
  chunk.uuid as chunk_uuid,
  chunk.content as chunk_content,
  chunk.header_path as header_path,
  collect(DISTINCT other_fact) as related_facts,
  collect(DISTINCT {subject: subj, relationship: type(r1), object: obj}) as entity_relationships,
  collect(DISTINCT topic.name) as topics
```

#### 3c. Topic Expansion
```cypher
// Find all chunks discussing this topic
MATCH (t:TopicNode {uuid: $topic_uuid})<-[:DISCUSSES]-(chunk:EpisodicNode)
WHERE chunk.group_id = $group_id

// Get facts from those chunks
OPTIONAL MATCH (chunk)-[:CONTAINS_FACT]->(fact:FactNode)

// Get entities mentioned in those chunks
OPTIONAL MATCH (entity:EntityNode)-[]->(chunk)

// Get co-discussed topics
OPTIONAL MATCH (chunk)-[:DISCUSSES]->(other_topic:TopicNode)
WHERE other_topic.uuid <> $topic_uuid

RETURN
  chunk.uuid as chunk_uuid,
  chunk.content as chunk_content,
  chunk.header_path as header_path,
  chunk.document_date as document_date,
  collect(DISTINCT fact.content) as facts,
  collect(DISTINCT entity.name) as entities,
  collect(DISTINCT other_topic.name) as co_topics
ORDER BY chunk.document_date DESC
```

#### 3d. Chunk Expansion (when chunk is the seed)
```cypher
MATCH (chunk:EpisodicNode {uuid: $chunk_uuid})
WHERE chunk.group_id = $group_id

// Get parent document
OPTIONAL MATCH (doc:DocumentNode)-[:CONTAINS_CHUNK]->(chunk)

// Get all facts
OPTIONAL MATCH (chunk)-[:CONTAINS_FACT]->(fact:FactNode)

// Get all entity relationships through this chunk
OPTIONAL MATCH (subj:EntityNode)-[r1]->(chunk)-[r2]->(obj:EntityNode)
WHERE type(r2) = type(r1) + '_TARGET'

// Get topics
OPTIONAL MATCH (chunk)-[:DISCUSSES]->(topic:TopicNode)

// Get sibling chunks (other chunks from same document)
OPTIONAL MATCH (doc)-[:CONTAINS_CHUNK]->(sibling:EpisodicNode)
WHERE sibling.uuid <> $chunk_uuid

RETURN
  chunk,
  doc.name as document_name,
  collect(DISTINCT fact) as facts,
  collect(DISTINCT {
    subject: subj.name,
    relationship: type(r1),
    description: r1.description,
    object: obj.name
  }) as relationships,
  collect(DISTINCT topic.name) as topics,
  collect(DISTINCT sibling.header_path) as sibling_headers
```

**Expansion Data Structure:**
```python
@dataclass
class ExpandedNode:
    """Result of expanding a seed node."""
    source_node: ScoredNode           # The seed we expanded from
    node_type: str                     # Entity, Fact, Topic, Chunk

    # Expanded context
    connected_entities: list[EntityInfo]
    connected_facts: list[FactInfo]
    connected_topics: list[str]
    source_chunks: list[ChunkInfo]

    # Relationship information
    relationships: list[RelationshipInfo]

    # Path tracking (for multi-hop)
    hop_distance: int
    traversal_path: list[str]         # e.g., ["Entity:Apple", "ACQUIRED", "Chunk:xyz", "Entity:Beats"]
```

---

### Component 4: Attention Scoring

**Purpose:** Score retrieved nodes for relevance to the original query.

**Two-Stage Approach:**

#### Stage A: Fast Embedding Filter
```python
async def fast_filter(
    nodes: list[ExpandedNode],
    query_embedding: list[float],
    threshold: float = 0.3
) -> list[ExpandedNode]:
    """
    Quick filter using embedding similarity.
    Removes obviously irrelevant nodes before expensive LLM scoring.
    """
    filtered = []
    for node in nodes:
        # Get the node's embedding (depends on type)
        node_embedding = get_node_embedding(node)

        if node_embedding is not None:
            similarity = cosine_similarity(query_embedding, node_embedding)
            if similarity >= threshold:
                node.embedding_score = similarity
                filtered.append(node)
        else:
            # No embedding available, include with neutral score
            node.embedding_score = 0.5
            filtered.append(node)

    return filtered
```

#### Stage B: LLM Relevance Scoring
```python
async def llm_score_batch(
    nodes: list[ExpandedNode],
    query: str,
    query_plan: QueryPlan,
    batch_size: int = 10
) -> list[ScoredExpandedNode]:
    """
    Use LLM to score relevance of each node.
    Batch for efficiency.
    """
    prompt = """
You are scoring the relevance of knowledge graph nodes to a user query.

Query: {query}
Query Intent: {intent}

For each node below, provide:
1. relevance_score: 0.0-1.0 (how relevant is this to answering the query)
2. reasoning: Brief explanation of why (1 sentence)
3. should_expand: true/false (should we explore this node's neighbors in the next hop)

Nodes to score:
{nodes_json}

Respond in JSON format:
[
  {{"node_id": "...", "relevance_score": 0.85, "reasoning": "...", "should_expand": true}},
  ...
]
"""

    results = []
    for batch in chunk_list(nodes, batch_size):
        nodes_json = format_nodes_for_scoring(batch)
        response = await llm.generate(
            prompt.format(
                query=query,
                intent=query_plan.intent,
                nodes_json=nodes_json
            )
        )
        scores = parse_scores(response)

        for node, score_info in zip(batch, scores):
            node.llm_score = score_info["relevance_score"]
            node.reasoning = score_info["reasoning"]
            node.should_expand = score_info["should_expand"]

            # Combined score
            alpha = 0.4  # Weight for embedding score
            node.combined_score = (
                alpha * node.embedding_score +
                (1 - alpha) * node.llm_score
            )
            results.append(node)

    return sorted(results, key=lambda n: n.combined_score, reverse=True)
```

**Scoring Considerations by Node Type:**

| Node Type | Scoring Focus |
|-----------|---------------|
| EntityNode | Name match, summary relevance to query |
| FactNode | Fact content relevance, relationship match |
| TopicNode | Topic relevance to query themes |
| ChunkNode | Overall content relevance, header_path match |
| Relationship | Relationship type matches query relationships |

---

### Component 5: Iterative Deepening

**Purpose:** Control multi-hop traversal depth and decide when to stop.

```python
@dataclass
class TraversalState:
    """Tracks state across iterations."""
    hop: int = 0
    max_hops: int = 3
    beam_width: int = 5

    # Accumulated context
    visited_nodes: set[str] = field(default_factory=set)
    accumulated_entities: list[EntityInfo] = field(default_factory=list)
    accumulated_facts: list[FactInfo] = field(default_factory=list)
    accumulated_chunks: list[ChunkInfo] = field(default_factory=list)
    traversal_paths: list[str] = field(default_factory=list)

    # Frontier for next expansion
    frontier: list[ScoredExpandedNode] = field(default_factory=list)

    # Stopping conditions
    sufficient_context: bool = False
    no_new_relevant_nodes: bool = False


async def iterative_traverse(
    query: str,
    query_plan: QueryPlan,
    initial_seeds: list[ScoredNode],
    state: TraversalState
) -> TraversalState:
    """
    Main traversal loop implementing GNN-style message passing.
    """

    # Initialize frontier with seeds
    state.frontier = initial_seeds

    while state.hop < state.max_hops and not state.sufficient_context:
        logger.info(f"Hop {state.hop}: Processing {len(state.frontier)} nodes")

        # STEP 1: Expand current frontier (Message Passing)
        expanded_nodes = []
        for node in state.frontier:
            if node.uuid in state.visited_nodes:
                continue

            expansion = await expand_node(node)
            expanded_nodes.append(expansion)
            state.visited_nodes.add(node.uuid)

        if not expanded_nodes:
            state.no_new_relevant_nodes = True
            break

        # STEP 2: Score expanded nodes (Attention)
        scored_nodes = await score_nodes(
            expanded_nodes,
            query,
            query_plan
        )

        # STEP 3: Aggregate into accumulated context
        for node in scored_nodes:
            if node.combined_score > 0.5:  # Relevance threshold
                accumulate_node(state, node)

        # STEP 4: Check sufficiency (Early Stopping)
        state.sufficient_context = await check_sufficiency(
            query=query,
            query_plan=query_plan,
            accumulated_context=state
        )

        if state.sufficient_context:
            logger.info(f"Sufficient context at hop {state.hop}")
            break

        # STEP 5: Select frontier for next hop (Beam Search)
        next_frontier = [
            n for n in scored_nodes
            if n.should_expand and n.uuid not in state.visited_nodes
        ]
        state.frontier = next_frontier[:state.beam_width]

        state.hop += 1

    return state


async def check_sufficiency(
    query: str,
    query_plan: QueryPlan,
    accumulated_context: TraversalState
) -> bool:
    """
    Ask LLM if we have enough context to answer the query.
    """
    prompt = """
You are evaluating whether we have sufficient context to answer a query.

Query: {query}
Query Intent: {intent}

Accumulated Context:
- Entities found: {num_entities}
- Facts found: {num_facts}
- Chunks examined: {num_chunks}

Sample facts:
{sample_facts}

Can we confidently answer the query with this context?
Consider:
1. Do we have the key entities mentioned in the query?
2. Do we have facts that address the relationships asked about?
3. For causal queries, do we have the causal chain?
4. For temporal queries, do we have the time range covered?

Respond with JSON:
{{"sufficient": true/false, "missing": "what information is still needed if not sufficient"}}
"""

    response = await llm.generate(prompt.format(
        query=query,
        intent=query_plan.intent,
        num_entities=len(accumulated_context.accumulated_entities),
        num_facts=len(accumulated_context.accumulated_facts),
        num_chunks=len(accumulated_context.accumulated_chunks),
        sample_facts=format_sample_facts(accumulated_context.accumulated_facts[:5])
    ))

    result = json.loads(response)
    return result["sufficient"]
```

**Stopping Conditions:**
1. `sufficient_context = True` (LLM says we have enough)
2. `hop >= max_hops` (depth limit reached)
3. `no_new_relevant_nodes = True` (frontier exhausted)
4. `len(accumulated_facts) >= max_facts` (context window management)

---

### Component 6: Context Aggregation

**Purpose:** Structure retrieved information for final synthesis.

```python
@dataclass
class AggregatedContext:
    """Final structured context for synthesis."""

    # Core content
    entities: list[EntityContext]
    facts: list[FactContext]
    chunks: list[ChunkContext]
    topics: list[str]

    # Relationship structure
    relationships: list[RelationshipContext]
    traversal_paths: list[TraversalPath]

    # Metadata
    total_hops: int
    nodes_examined: int
    query_coverage: float  # Estimated coverage of query entities/topics


@dataclass
class EntityContext:
    uuid: str
    name: str
    summary: str
    relevance_score: float
    mentioned_in_chunks: list[str]  # chunk uuids
    relationships: list[str]  # ["ACQUIRED -> Beats", "PARTNERED_WITH -> IBM"]


@dataclass
class FactContext:
    uuid: str
    content: str
    subject: str
    relationship: str
    object: str
    source_chunk_uuid: str
    source_header_path: str
    document_date: str
    relevance_score: float


@dataclass
class ChunkContext:
    uuid: str
    content: str
    header_path: str
    document_date: str
    document_name: str
    relevance_score: float
    contains_facts: list[str]  # fact uuids
    discusses_topics: list[str]


@dataclass
class TraversalPath:
    """Represents a path through the graph that answers part of the query."""
    path: list[str]  # ["Boston District", "-[EXPERIENCED]->", "chunk-1", "-[EXPERIENCED_TARGET]->", "Inflation"]
    path_description: str  # "Boston District experienced inflation concerns"
    supporting_facts: list[str]  # fact contents along this path
    relevance_to_query: str  # which part of query this addresses


def aggregate_context(state: TraversalState) -> AggregatedContext:
    """
    Transform traversal state into structured context for synthesis.
    """
    # Deduplicate entities
    entity_map = {}
    for entity in state.accumulated_entities:
        if entity.uuid not in entity_map:
            entity_map[entity.uuid] = EntityContext(
                uuid=entity.uuid,
                name=entity.name,
                summary=entity.summary,
                relevance_score=entity.relevance_score,
                mentioned_in_chunks=[],
                relationships=[]
            )
        # Accumulate mentions
        entity_map[entity.uuid].mentioned_in_chunks.extend(entity.source_chunks)

    # Deduplicate and structure facts
    fact_map = {}
    for fact in state.accumulated_facts:
        if fact.uuid not in fact_map:
            fact_map[fact.uuid] = FactContext(
                uuid=fact.uuid,
                content=fact.content,
                subject=fact.subject,
                relationship=fact.relationship,
                object=fact.object,
                source_chunk_uuid=fact.source_chunk,
                source_header_path=fact.header_path,
                document_date=fact.document_date,
                relevance_score=fact.relevance_score
            )

    # Structure chunks
    chunk_map = {}
    for chunk in state.accumulated_chunks:
        if chunk.uuid not in chunk_map:
            chunk_map[chunk.uuid] = ChunkContext(
                uuid=chunk.uuid,
                content=chunk.content,
                header_path=chunk.header_path,
                document_date=chunk.document_date,
                document_name=chunk.document_name,
                relevance_score=chunk.relevance_score,
                contains_facts=[f.uuid for f in state.accumulated_facts if f.source_chunk == chunk.uuid],
                discusses_topics=chunk.topics
            )

    # Build traversal paths
    paths = build_traversal_paths(state.traversal_paths, fact_map)

    return AggregatedContext(
        entities=sorted(entity_map.values(), key=lambda e: e.relevance_score, reverse=True),
        facts=sorted(fact_map.values(), key=lambda f: f.relevance_score, reverse=True),
        chunks=sorted(chunk_map.values(), key=lambda c: c.relevance_score, reverse=True),
        topics=list(set(t for c in chunk_map.values() for t in c.discusses_topics)),
        relationships=extract_relationships(state),
        traversal_paths=paths,
        total_hops=state.hop,
        nodes_examined=len(state.visited_nodes),
        query_coverage=calculate_coverage(state, query_plan)
    )
```

---

### Component 7: Synthesis

**Purpose:** Generate final answer with citations and confidence.

```python
async def synthesize_answer(
    query: str,
    query_plan: QueryPlan,
    context: AggregatedContext
) -> SynthesizedAnswer:
    """
    Generate final answer using accumulated context.
    """

    prompt = """
You are answering a question using information from a knowledge graph about financial documents.

## Query
{query}

## Query Intent
{intent}

## Retrieved Context

### Key Entities
{entities_section}

### Relevant Facts
{facts_section}

### Source Chunks (for citations)
{chunks_section}

### Traversal Paths (showing how information connects)
{paths_section}

## Instructions

1. Answer the query directly and completely
2. Use ONLY information from the provided context
3. For each claim, cite the source using [Header Path, Date] format
4. If the query asks about causation, explain the causal chain using the traversal paths
5. If information is missing or uncertain, acknowledge it
6. Provide a confidence level (high/medium/low) based on evidence strength

## Response Format

Provide your response as JSON:
{{
  "answer": "Your complete answer with inline citations...",
  "confidence": "high|medium|low",
  "confidence_reasoning": "Why this confidence level",
  "key_facts_used": ["fact1", "fact2"],
  "entities_mentioned": ["entity1", "entity2"],
  "limitations": "Any caveats or missing information"
}}
"""

    # Format context sections
    entities_section = format_entities_for_prompt(context.entities[:10])
    facts_section = format_facts_for_prompt(context.facts[:20])
    chunks_section = format_chunks_for_prompt(context.chunks[:10])
    paths_section = format_paths_for_prompt(context.traversal_paths[:5])

    response = await llm.generate(
        prompt.format(
            query=query,
            intent=query_plan.intent,
            entities_section=entities_section,
            facts_section=facts_section,
            chunks_section=chunks_section,
            paths_section=paths_section
        ),
        temperature=0.3  # Lower temperature for factual accuracy
    )

    result = json.loads(response)

    return SynthesizedAnswer(
        answer=result["answer"],
        confidence=result["confidence"],
        confidence_reasoning=result["confidence_reasoning"],
        key_facts_used=result["key_facts_used"],
        entities_mentioned=result["entities_mentioned"],
        limitations=result["limitations"],

        # Attach metadata
        context_stats={
            "entities_retrieved": len(context.entities),
            "facts_retrieved": len(context.facts),
            "chunks_retrieved": len(context.chunks),
            "hops_taken": context.total_hops,
            "nodes_examined": context.nodes_examined
        },
        traversal_paths=context.traversal_paths
    )


@dataclass
class SynthesizedAnswer:
    """Final answer with full provenance."""
    answer: str
    confidence: Literal["high", "medium", "low"]
    confidence_reasoning: str
    key_facts_used: list[str]
    entities_mentioned: list[str]
    limitations: str

    context_stats: dict
    traversal_paths: list[TraversalPath]

    def to_user_response(self) -> str:
        """Format for user display."""
        response = self.answer

        if self.limitations:
            response += f"\n\n**Note:** {self.limitations}"

        response += f"\n\n*Confidence: {self.confidence}*"

        return response
```

---

## Query Type Handling

Different query intents require different traversal strategies:

### Factual Queries
```
"What was Apple's revenue in Q3 2024?"

Strategy:
- Shallow search (1-2 hops)
- High weight on exact entity match
- Look for FactNodes with numeric content
- Single chunk may suffice
```

### Causal Chain Queries
```
"What caused inflation in the Boston district and how did it affect hiring?"

Strategy:
- Deep search (2-4 hops)
- Follow relationship edges specifically
- Build traversal paths showing causation
- Multiple chunks likely needed
- Look for temporal ordering
```

### Comparison Queries
```
"Compare labor market conditions between Boston and Atlanta districts"

Strategy:
- Parallel search from both entities
- Same topics, different entities
- Aggregate by topic, compare across entities
- Synthesize similarities and differences
```

### Temporal Queries
```
"How has inflation changed from Q2 to Q4 2024?"

Strategy:
- Filter by document_date
- Same entity/topic across time periods
- Order results chronologically
- Track changes in facts over time
```

### Aggregation Queries
```
"Summarize the key economic trends across all Fed districts"

Strategy:
- Broad topic search
- Low depth, high breadth
- Group by topic
- Synthesize common themes
```

### Exploration Queries
```
"What are the main themes in the October 2024 Beige Book?"

Strategy:
- Start with document filter
- Expand to all topics discussed
- Cluster facts by topic
- Summarize top themes
```

---

## Performance Optimizations

### 1. Parallel Search Execution
```python
async def parallel_seed_search(query: str, query_plan: QueryPlan):
    """Run all seed searches in parallel."""
    async with asyncio.TaskGroup() as tg:
        entity_task = tg.create_task(search_entities(query))
        fact_task = tg.create_task(search_facts(query))
        topic_task = tg.create_task(search_topics(query))
        chunk_task = tg.create_task(search_chunks(query))

    return merge_results(
        entity_task.result(),
        fact_task.result(),
        topic_task.result(),
        chunk_task.result()
    )
```

### 2. Batch LLM Calls
```python
# Instead of scoring one node at a time
for node in nodes:
    score = await llm.score(node)  # BAD: N API calls

# Batch them
batched_scores = await llm.score_batch(nodes)  # GOOD: N/batch_size API calls
```

### 3. Caching
```python
# Cache embeddings
@lru_cache(maxsize=1000)
def get_query_embedding(query: str) -> list[float]:
    return embeddings.embed(query)

# Cache expansion results (short TTL)
@redis_cache(ttl=300)
async def expand_node(node_uuid: str) -> ExpandedNode:
    return await _expand_node_impl(node_uuid)
```

### 4. Early Termination
```python
# Stop as soon as we have enough high-confidence facts
if len([f for f in facts if f.relevance_score > 0.8]) >= 5:
    return early_synthesize(facts)
```

### 5. Hybrid Retrieval Order
```python
# Start with fast fulltext, fall back to vector if needed
results = await fulltext_search(query)
if len(results) < min_results:
    vector_results = await vector_search(query)
    results = merge_unique(results, vector_results)
```

---

## Error Handling and Fallbacks

### No Results Found
```python
if not seed_nodes:
    # Try broader search
    seed_nodes = await broader_search(query)

if not seed_nodes:
    return SynthesizedAnswer(
        answer="I couldn't find relevant information in the knowledge graph for this query.",
        confidence="low",
        limitations="No matching entities, facts, or topics found."
    )
```

### LLM Failures
```python
try:
    scores = await llm_score_batch(nodes)
except LLMError:
    # Fall back to embedding-only scoring
    scores = [ScoredNode(n, n.embedding_score, "Embedding-only score") for n in nodes]
```

### Timeout Handling
```python
async def traverse_with_timeout(query, max_seconds=30):
    try:
        async with asyncio.timeout(max_seconds):
            return await full_traverse(query)
    except asyncio.TimeoutError:
        # Return partial results
        return partial_synthesize(current_state)
```

---

## Monitoring and Observability

### Metrics to Track
```python
@dataclass
class QueryMetrics:
    query_id: str
    timestamp: datetime

    # Timing
    query_understanding_ms: float
    seed_retrieval_ms: float
    expansion_ms: float
    scoring_ms: float
    synthesis_ms: float
    total_ms: float

    # Volume
    seeds_found: int
    nodes_expanded: int
    nodes_scored: int
    hops_taken: int

    # Quality
    confidence: str
    user_feedback: Optional[int]  # 1-5 rating

    # Cost
    llm_tokens_used: int
    embedding_calls: int
```

### Logging
```python
logger.info(f"Query: {query[:100]}")
logger.info(f"Intent: {query_plan.intent}")
logger.info(f"Seeds found: {len(seeds)} (entities={e}, facts={f}, topics={t})")
logger.info(f"Hop {hop}: expanded {len(expanded)}, scored {len(scored)}, frontier={len(frontier)}")
logger.info(f"Synthesis: confidence={answer.confidence}, facts_used={len(answer.key_facts_used)}")
```

---

## File Structure

```
src/querying/
├── __init__.py
├── query_engine.py          # Main orchestrator
├── query_understanding.py   # Layer 1: Query decomposition
├── seed_retrieval.py        # Layer 2: Initial search
├── graph_expansion.py       # Layer 3: Structure-aware expansion
├── attention_scoring.py     # Layer 4: Relevance scoring
├── traversal_control.py     # Layer 5: Iterative deepening
├── context_aggregation.py   # Layer 6: Context structuring
├── synthesis.py             # Layer 7: Answer generation
├── schemas.py               # Data classes
├── prompts/
│   ├── query_understanding.txt
│   ├── relevance_scoring.txt
│   ├── sufficiency_check.txt
│   └── synthesis.txt
└── tests/
    ├── test_query_understanding.py
    ├── test_seed_retrieval.py
    ├── test_graph_expansion.py
    └── test_full_pipeline.py
```

---

## Example Query Trace

**Query:** "What caused inflation concerns in Boston and how did it affect hiring decisions?"

```
┌─ LAYER 1: Query Understanding ────────────────────────────────────────────┐
│ entities: ["Boston", "inflation"]                                         │
│ topics: ["Inflation", "Labor Markets", "Hiring"]                          │
│ relationships: ["caused", "affected"]                                     │
│ intent: "causal_chain"                                                    │
│ expected_hops: 2                                                          │
└───────────────────────────────────────────────────────────────────────────┘

┌─ LAYER 2: Seed Retrieval ─────────────────────────────────────────────────┐
│ Entity search "Boston": 3 results (Boston District, Boston Fed, ...)     │
│ Fact search "inflation concerns": 8 results                               │
│ Topic search "Inflation": 1 result (TopicNode: Inflation)                 │
│ Topic search "Hiring": 1 result (TopicNode: Labor Markets)                │
│ Merged seeds: 12 unique nodes                                             │
└───────────────────────────────────────────────────────────────────────────┘

┌─ LAYER 3: Expansion (Hop 0) ──────────────────────────────────────────────┐
│ Expanding EntityNode "Boston District":                                   │
│   → Found 15 outgoing relationships to chunks                             │
│   → Relationship types: EXPERIENCED, REPORTED, NOTED                      │
│                                                                           │
│ Expanding TopicNode "Inflation":                                          │
│   → Found 8 chunks discussing Inflation                                   │
│   → Co-discussed topics: Supply Chain, Input Costs, Pricing               │
│                                                                           │
│ Expanding TopicNode "Labor Markets":                                      │
│   → Found 12 chunks discussing Labor Markets                              │
│   → Co-discussed topics: Hiring, Wages, Employment                        │
│                                                                           │
│ Total expanded nodes: 42                                                  │
└───────────────────────────────────────────────────────────────────────────┘

┌─ LAYER 4: Attention Scoring ──────────────────────────────────────────────┐
│ Stage A (Embedding filter): 42 → 28 nodes (threshold 0.3)                 │
│ Stage B (LLM scoring):                                                    │
│   - Chunk "Boston > Prices" → 0.92 (directly discusses inflation causes)  │
│   - Chunk "Boston > Employment" → 0.88 (discusses hiring decisions)       │
│   - Fact "Rising input costs drove price increases" → 0.95                │
│   - Fact "Employers slowed hiring amid uncertainty" → 0.91                │
│   - Entity "Manufacturing sector" → 0.75 (related context)                │
│ Top 10 selected for aggregation                                           │
└───────────────────────────────────────────────────────────────────────────┘

┌─ LAYER 5: Sufficiency Check ──────────────────────────────────────────────┐
│ LLM evaluation:                                                           │
│   - Have entity "Boston District"? ✓                                      │
│   - Have facts about inflation causes? ✓                                  │
│   - Have facts about hiring effects? ✓                                    │
│   - Have causal connection? ⚠ (need one more hop)                         │
│ Result: NOT sufficient, need to expand for causal link                    │
└───────────────────────────────────────────────────────────────────────────┘

┌─ LAYER 3: Expansion (Hop 1) ──────────────────────────────────────────────┐
│ Frontier: top 5 nodes from Hop 0                                          │
│ Expanding from "Rising input costs" fact:                                 │
│   → Source chunk discusses both costs AND employer response               │
│   → Found linking fact: "Cost pressures led firms to delay hiring"        │
│                                                                           │
│ Total new nodes: 18                                                       │
└───────────────────────────────────────────────────────────────────────────┘

┌─ LAYER 4: Attention Scoring (Hop 1) ──────────────────────────────────────┐
│ New high-scoring nodes:                                                   │
│   - Fact "Cost pressures led firms to delay hiring plans" → 0.96          │
│   - Chunk "Boston > Manufacturing" → 0.85 (specific sector example)       │
└───────────────────────────────────────────────────────────────────────────┘

┌─ LAYER 5: Sufficiency Check (Hop 1) ──────────────────────────────────────┐
│ LLM evaluation:                                                           │
│   - Have causal chain: costs → inflation → hiring impact? ✓               │
│ Result: SUFFICIENT                                                        │
└───────────────────────────────────────────────────────────────────────────┘

┌─ LAYER 6: Context Aggregation ────────────────────────────────────────────┐
│ Entities: 4 (Boston District, Manufacturing sector, Employers, ...)      │
│ Facts: 8 (filtered from 60 total examined)                                │
│ Chunks: 5 (with full provenance)                                          │
│ Traversal paths:                                                          │
│   1. Boston District -[EXPERIENCED]-> chunk1 -[EXPERIENCED_TARGET]->      │
│      Input Cost Increases                                                 │
│   2. Input Costs -[DROVE]-> chunk2 -[DROVE_TARGET]-> Price Increases      │
│   3. Cost Pressures -[LED_TO]-> chunk3 -[LED_TO_TARGET]-> Hiring Delays   │
└───────────────────────────────────────────────────────────────────────────┘

┌─ LAYER 7: Synthesis ──────────────────────────────────────────────────────┐
│ Answer:                                                                   │
│   "Inflation concerns in the Boston district were primarily driven by     │
│    rising input costs, particularly in manufacturing and construction     │
│    [Boston > Prices, Oct 2024]. Supply chain disruptions continued to     │
│    pressure costs [Boston > Supply Chain, Oct 2024].                      │
│                                                                           │
│    These cost pressures directly affected hiring decisions, as employers  │
│    reported delaying expansion plans and slowing new hires amid           │
│    uncertainty about future margins [Boston > Employment, Oct 2024].      │
│    Manufacturing firms specifically noted postponing hiring until cost    │
│    pressures stabilized [Boston > Manufacturing, Oct 2024]."              │
│                                                                           │
│ Confidence: HIGH                                                          │
│ Facts used: 6                                                             │
│ Hops taken: 2                                                             │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Priority

### Phase 1: Core Pipeline (MVP)
1. Query understanding with basic intent classification
2. Parallel seed retrieval (entity + fact + topic)
3. Single-hop expansion
4. Embedding-only scoring (no LLM scoring yet)
5. Basic synthesis

### Phase 2: Multi-hop + Attention
1. Full iterative deepening loop
2. LLM-based relevance scoring
3. Sufficiency checking
4. Traversal path tracking

### Phase 3: Optimization
1. Caching layers
2. Parallel expansion
3. Batch LLM calls
4. Timeout handling

### Phase 4: Advanced Features
1. Query type-specific strategies
2. Temporal filtering
3. Comparison queries
4. Confidence calibration
