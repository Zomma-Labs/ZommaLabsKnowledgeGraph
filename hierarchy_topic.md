Graph Architecture: The "Dimensional Star" Paradigm

1. Executive Summary

Goal: Build a Financial Knowledge Graph optimized for Analytical Intersection (e.g., "Show me [Labor Markets] in [New York]").

The Shift:

❌ Old Model (Tree): Document -> H1 -> H2 -> H3 -> Chunk. (Hard to query across branches).

✅ New Model (Star): We flatten the document hierarchy into Context Hubs. A SectionNode acts as a central hub that links to Global Dimensions (Entity, Topic).

Key Principles:

Reification: Facts are Nodes (to support Causality).

Dimensionality: Sections are "Hubs" connecting Local Content to Global Concepts.

Security: Strict Multi-tenancy via group_id.

2. The Topology: The "Context Star"

Instead of a deep tree, every meaningful block of text is represented by a Hub that radiates connections to its semantic dimensions.

graph TD
    %% 1. THE GLOBAL DIMENSIONS (Shared across all files)
    Entity[EntityNode: "Fed of New York"]
    Topic[TopicNode: "Labor Markets"]
    
    %% 2. THE LOCAL HUB (Unique to this document block)
    %% Represents the intersection "NY + Labor"
    Hub[SectionNode: "Context Hub"]
    
    %% 3. THE CONNECTIONS (Star Shape)
    Hub -->|REPRESENTS| Entity
    Hub -->|DISCUSSES| Topic
    Hub -->|PART_OF| Doc[DocumentNode]
    
    %% 4. THE DATA (Hanging off the Hub)
    Hub -->|CONTAINS| Chunk[EpisodicNode]
    Fact1(Fact: "Hiring froze") -->|MENTIONED_IN| Chunk
    Entity -->|PERFORMED| Fact1


3. Node Schema Definitions

Global Rule: All nodes MUST have group_id (Tenant ID).

A. The Context Hub & Dimensions

class SectionNode(Node):
    """
    The Context Hub. Represents a thematic block within a file.
    Example: "Regional Reports > New York > Labor Markets"
    """
    header_path: str # "Regional Reports > New York > Labor Markets"
    doc_id: str

class TopicNode(Node):
    """
    Global Theme. Resolves to FIBO.
    Example: "Inflation", "Labor Markets".
    """
    name: str 
    fibo_id: Optional[str] # "fibo-ind-ei-ei:LaborMarketIndicator"

class EntityNode(Node):
    """Global Actor. Example: "Federal Reserve of NY"."""
    name: str
    labels: list[str]


B. The Data (Reified)

class FactNode(Node):
    """Atomic Event. Links to Actor (Entity) and Source (Chunk)."""
    content: str
    embedding: list[float] # voyage-finance-2


4. Ingestion Logic (The Assembler)

Embedding Model: voyage-finance-2

Step 1: Squash the Hierarchy

Input: A Chunk with header path ["2. Regional Reports", "New York", "Labor Markets"].

Action: Do NOT create 3 nested nodes. Analyze the entire path to extract dimensions.

Entity Extraction: "New York" -> EntityNode("Federal Reserve Bank of New York").

Topic Extraction: "Labor Markets" -> TopicNode("LaborMarketIndicator") (FIBO Resolved).

Step 2: Create the Hub

MERGE the SectionNode (The Hub).

Properties: header_path = "Regional... > Labor Markets".

Link Dimensions (Bi-Directional):

MERGE (Hub)-[:REPRESENTS]->(Entity)

MERGE (Entity)-[:IS_REPRESENTED_IN]->(Hub)

MERGE (Hub)-[:DISCUSSES]->(Topic)

MERGE (Topic)-[:IS_DISCUSSED_IN]->(Hub)

MERGE (Hub)-[:PART_OF]->(Document)

MERGE (Document)-[:TALKS_ABOUT]->(Hub)

Step 3: Link the Content

MERGE the EpisodicNode (Chunk).

Link: (Hub)-[:CONTAINS]->(Chunk).

Extract Facts: (Standard LLM extraction).

Link: (Fact)-[:MENTIONED_IN]->(Chunk).

5. Querying Logic (The Intersection)

This topology allows for high-precision "Slice and Dice" queries.

Query A: "How were Labor Markets in NY?"

Logic: Find the Hub where BOTH dimensions meet.

MATCH (hub:SectionNode)
WHERE (hub)-[:REPRESENTS]->(:EntityNode {name: "Fed of NY"})
  AND (hub)-[:DISCUSSES]->(:TopicNode {name: "Labor Markets"})

// Drill down to facts
MATCH (hub)-[:CONTAINS]->(chunk)<-[:MENTIONED_IN]-(fact:FactNode)
RETURN fact.content


Query B: "Show me all discussions about Inflation."

Logic: Lock the Topic, traverse to find context.

MATCH (t:TopicNode {name: "Inflation"})<-[:DISCUSSES]-(hub:SectionNode)
MATCH (hub)-[:CONTAINS]->(chunk)<-[:MENTIONED_IN]-(fact)
RETURN hub.header_path, collect(fact.content)


6. Implementation Checklist for Agent

Header Analysis: Implement a step that takes a List[str] (headers) and classifies each string as Entity, Topic, or Noise.

Hub Creation: Ensure the SectionNode is created once per unique header path per document.

FIBO: Ensure TopicNode creation attempts FIBO resolution (Threshold > 0.9).

Star Linking: Ensure the SectionNode links to ALL found dimensions with reciprocal edges.