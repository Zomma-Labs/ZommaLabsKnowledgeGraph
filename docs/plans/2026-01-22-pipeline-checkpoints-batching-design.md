# Pipeline Checkpoints & Batched Writes Design

**Date:** 2026-01-22
**Status:** Approved

## Problem

The pipeline crashed during Phase 3c (Neo4j bulk write) after completing expensive LLM extraction and resolution phases. All work was lost because:
1. No checkpoint/save mechanism exists between phases
2. The bulk write sent 12,323 relationships in a single query, causing connection timeout

## Solution

### 1. Checkpoint System

#### Directory Structure
```
checkpoints/
└── {input_filename}_{group_id}_{run_id}/
    ├── metadata.json          # Run info, timestamps, current phase
    ├── phase1_extraction.pkl  # Extractions after Phase 1
    ├── phase2_resolution.pkl  # Entity + topic lookups after Phase 2
    └── phase3_buffer.pkl      # BulkWriteBuffer after Phase 3a/3b
```

- `run_id`: First 8 chars of UUID for uniqueness
- Pickle format for fast serialization of embeddings (user-requested for speed)
- Separate files allow resuming from any phase

**Security note:** Pickle files are only loaded from checkpoints created by this pipeline. They are not intended for external/untrusted data.

#### Checkpoint Contents

**phase1_extraction.pkl:**
```python
{
    "extractions": [...],        # List of extraction results
    "document_uuid": str,
    "document_name": str,
    "document_date_str": str,
    "chunks": [...],             # Original chunks for Phase 3
}
```

**phase2_resolution.pkl:**
```python
{
    "entity_lookup": {...},      # name -> resolved entity
    "topic_lookup": {...},       # name -> TopicResolution
    "dedup_canonical_map": {...} # original name -> canonical name
}
```

**phase3_buffer.pkl:**
```python
{
    "buffer": BulkWriteBuffer,   # Full buffer object
    "embeddings_done": bool,     # True if Phase 3b completed
}
```

#### Save Points
- After Phase 1 completes → save `phase1_extraction.pkl`
- After Phase 2 completes → save `phase2_resolution.pkl`
- After Phase 3a (collection) → save `phase3_buffer.pkl`
- After Phase 3b (embeddings) → update `phase3_buffer.pkl`
- After Phase 3c (write) → delete checkpoint directory (success)

### 2. CLI Interface

**New flags:**
```bash
--resume       # Resume from checkpoint (fail if none exists)
--fresh        # Force fresh start (ignore/delete existing checkpoint)
--batch-size N # Neo4j write batch size (default: 250)
```

**Default behavior (no flag):**
- If checkpoint exists for input file + group_id → auto-resume
- If no checkpoint → start fresh

**Example usage:**
```bash
# Normal run (auto-resume if checkpoint exists)
uv run src/pipeline.py --filter beige

# Explicit resume
uv run src/pipeline.py --filter beige --resume

# Force fresh start
uv run src/pipeline.py --filter beige --fresh

# Smaller batches for unstable connections
uv run src/pipeline.py --filter beige --batch-size 100
```

### 3. Batched Neo4j Writes

Modify `bulk_write_all()` to batch all operations:

```python
def bulk_write_all(buffer, neo4j, embeddings, llm, batch_size=250):
    # Batch entity nodes
    for i in range(0, len(buffer.entity_nodes), batch_size):
        batch = buffer.entity_nodes[i:i + batch_size]
        neo4j.query("UNWIND $nodes AS n CREATE ...", {"nodes": batch})

    # Batch fact nodes
    for i in range(0, len(buffer.fact_nodes), batch_size):
        batch = buffer.fact_nodes[i:i + batch_size]
        neo4j.query("UNWIND $nodes AS n MERGE ...", {"nodes": batch})

    # Batch topic nodes
    for i in range(0, len(buffer.topic_nodes), batch_size):
        batch = buffer.topic_nodes[i:i + batch_size]
        neo4j.query("UNWIND $nodes AS n MERGE ...", {"nodes": batch})

    # Batch relationships (grouped by type)
    for rel_type, rels in by_type.items():
        for i in range(0, len(rels), batch_size):
            batch = rels[i:i + batch_size]
            neo4j.query(f"UNWIND $rels AS r ...", {"rels": batch})
```

**Progress output:**
```
Phase 3c: Writing to Neo4j...
  Entities: 441/441 (2 batches)
  Facts: 2195/2195 (9 batches)
  Topics: 171/171 (1 batch)
  Relationships: 12323/12323 (50 batches)
```

### 4. Resume Flow

```
$ uv run src/pipeline.py --filter beige

============================================================
KNOWLEDGE GRAPH PIPELINE
============================================================

Found checkpoint: checkpoints/beige_book_default_a1b2c3d4/
  Input: src/chunker/SAVED/beige_book.jsonl
  Last completed: Phase 2 (resolution)
  Started: 2026-01-22 14:30:00

Resuming from Phase 3...

  Phase 3: Collecting operations...
    Collected 181 episodic, 441 entities, 2195 facts, 12323 rels
  Phase 3b: Generating embeddings (batched)...
    Generated embeddings in 30.2s
  Phase 3c: Writing to Neo4j...
    Entities: 441/441 (2 batches)
    ...

Checkpoint completed, cleaning up...
Done.
```

## Implementation Notes

1. Use `pickle` for checkpoints (fast for numpy arrays / embeddings)
2. `metadata.json` stays as JSON for easy inspection
3. Checkpoint directory naming: `{stem}_{group_id}_{run_id[:8]}`
4. On `--fresh`, delete existing checkpoint before starting
5. Keep `neo4j.warmup()` call, add periodic warmups during long batch sequences
