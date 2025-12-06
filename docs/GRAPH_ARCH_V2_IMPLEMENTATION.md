# Graph Architecture V2 Implementation Plan

 This document outlines the step-by-step features to implement for the Graph Architecture V2 migration.

 ## 1. Schema Updates

 ### 1.1 Update `EpisodicNode`
 - **File**: `src/schemas/nodes.py`
 - **Task**: Add `header_path: str` to the `EpisodicNode` model.
 - **Reason**: To replace `SectionNode` and store hierarchy directly on the chunk.

 ### 1.2 Deprecate or Remove `SectionNode`
 - **File**: `src/schemas/nodes.py`
 - **Task**: Remove the `SectionNode` class definition

 ### 1.3 Expand `RelationshipType`
 - **File**: `src/schemas/relationship.py`
 - **Task**: Add Passive Edge variants for all existing Active Edges (e.g., `GOT_ACQUIRED`, `GOT_SUED`).
 - **Task**: Create a mapping dictionary `ACTIVE_TO_PASSIVE` (and `PASSIVE_TO_ACTIVE`) to facilitate bi-directional lookups.

 ## 2. Graph Assembler Refactoring

 ### 2.1 Update `assemble_fact_node` Signature
 - **File**: `src/agents/graph_assembler.py`
 - **Task**: Verify the method accepts `EpisodeNode` UUID (or creates edges to it).

 ### 2.2 Implement Entity-Chunk Linking
 - **File**: `src/agents/graph_assembler.py`
 - **Task**:
     - Identify the Active Relationship from the classification.
     - Create the Active Edge: `(SubjectEntity)-[:ACTIVE_REL]->(EpisodicNode)`.
     - Create the Passive Edge: `(EpisodicNode)-[:PASSIVE_REL]->(ObjectEntity)`.
     - Ensure `group_id` is included in all MERGE/CREATE operations.

 ### 2.3 Remove Entity-Fact Linking
 - **Task**: Remove the code that creates `[:PERFORMED]` and `[:TARGET]` edges between Entities and FactNodes.
 - **Note**: FactNodes remain for atomic storage and causation, linked via `[:MENTIONED_IN]` to the Episode.

 ## 3. Verification

 ### 3.1 Test Run
 - **Task**: Run `assemble_graph` pipeline with a sample document.
 - **Success Criteria**:
     - Entities are connected to the `EpisodicNode` directly.
     - No `SectionNode`s are in the graph.
     - Cypher queries can traverse `Entity -> Chunk -> Entity`.
