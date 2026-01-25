# Pipeline Checkpoints & Batched Writes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add checkpoint saving after each pipeline phase and batch Neo4j writes to prevent connection timeouts.

**Architecture:** Create a `CheckpointManager` class in a new module that handles save/load/cleanup. Modify `bulk_write_all()` to accept a `batch_size` parameter and iterate in chunks. Add CLI flags `--resume`, `--fresh`, `--batch-size` to `main()`.

**Tech Stack:** Python pickle for fast serialization (user-requested for embedding arrays), JSON for human-readable metadata, pathlib for path handling.

**Security Note:** Pickle files are only loaded from checkpoints created by this pipeline locally. They are not intended for external/untrusted data.

---

### Task 1: Create CheckpointManager Module

**Files:**
- Create: `src/util/checkpoint.py`

**Step 1: Create the checkpoint module with CheckpointManager class**

```python
"""
Checkpoint management for pipeline resume capability.

Saves intermediate state after each phase so the pipeline can resume
from the last completed phase if it crashes.

Security: Pickle files are only loaded from checkpoints created by this
pipeline locally. Do not load checkpoints from untrusted sources.
"""

import json
import pickle
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4


CHECKPOINTS_DIR = Path("checkpoints")


@dataclass
class CheckpointMetadata:
    """Metadata about a checkpoint run."""
    run_id: str
    input_file: str
    group_id: str
    started_at: str
    last_phase: int  # 0 = not started, 1 = extraction, 2 = resolution, 3 = buffer ready
    cli_args: Dict[str, Any]


class CheckpointManager:
    """Manages checkpoint save/load/cleanup for pipeline runs."""

    def __init__(self, input_file: str, group_id: str, cli_args: Optional[Dict] = None):
        self.input_file = input_file
        self.group_id = group_id
        self.cli_args = cli_args or {}

        # Generate checkpoint directory name
        input_stem = Path(input_file).stem
        self.run_id = uuid4().hex[:8]
        self.checkpoint_dir = CHECKPOINTS_DIR / f"{input_stem}_{group_id}_{self.run_id}"

        self.metadata = CheckpointMetadata(
            run_id=self.run_id,
            input_file=input_file,
            group_id=group_id,
            started_at=datetime.now().isoformat(),
            last_phase=0,
            cli_args=self.cli_args
        )

    @classmethod
    def find_existing(cls, input_file: str, group_id: str) -> Optional["CheckpointManager"]:
        """Find an existing checkpoint for the given input file and group_id."""
        if not CHECKPOINTS_DIR.exists():
            return None

        input_stem = Path(input_file).stem
        prefix = f"{input_stem}_{group_id}_"

        # Find matching checkpoint directories
        matches = [d for d in CHECKPOINTS_DIR.iterdir() if d.is_dir() and d.name.startswith(prefix)]

        if not matches:
            return None

        # Return the most recent one (by directory name, which includes run_id)
        latest = sorted(matches, key=lambda d: d.stat().st_mtime, reverse=True)[0]

        # Load the checkpoint
        return cls._load_from_dir(latest)

    @classmethod
    def _load_from_dir(cls, checkpoint_dir: Path) -> Optional["CheckpointManager"]:
        """Load a CheckpointManager from an existing directory."""
        metadata_file = checkpoint_dir / "metadata.json"
        if not metadata_file.exists():
            return None

        with open(metadata_file) as f:
            meta_dict = json.load(f)

        # Create instance without generating new run_id
        instance = cls.__new__(cls)
        instance.checkpoint_dir = checkpoint_dir
        instance.run_id = meta_dict["run_id"]
        instance.input_file = meta_dict["input_file"]
        instance.group_id = meta_dict["group_id"]
        instance.cli_args = meta_dict.get("cli_args", {})
        instance.metadata = CheckpointMetadata(**meta_dict)

        return instance

    def _ensure_dir(self):
        """Ensure checkpoint directory exists."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _save_metadata(self):
        """Save metadata.json."""
        self._ensure_dir()
        with open(self.checkpoint_dir / "metadata.json", "w") as f:
            json.dump({
                "run_id": self.metadata.run_id,
                "input_file": self.metadata.input_file,
                "group_id": self.metadata.group_id,
                "started_at": self.metadata.started_at,
                "last_phase": self.metadata.last_phase,
                "cli_args": self.metadata.cli_args
            }, f, indent=2)

    def save_phase1(self, extractions: list, document_uuid: str, document_name: str,
                    document_date_str: Optional[str], chunks: list):
        """Save Phase 1 (extraction) results."""
        self._ensure_dir()
        data = {
            "extractions": extractions,
            "document_uuid": document_uuid,
            "document_name": document_name,
            "document_date_str": document_date_str,
            "chunks": chunks
        }
        with open(self.checkpoint_dir / "phase1_extraction.pkl", "wb") as f:
            pickle.dump(data, f)
        self.metadata.last_phase = 1
        self._save_metadata()

    def save_phase2(self, entity_lookup: dict, topic_lookup: dict,
                    dedup_canonical_map: dict, uuid_by_name: dict):
        """Save Phase 2 (resolution) results."""
        self._ensure_dir()
        data = {
            "entity_lookup": entity_lookup,
            "topic_lookup": topic_lookup,
            "dedup_canonical_map": dedup_canonical_map,
            "uuid_by_name": uuid_by_name
        }
        with open(self.checkpoint_dir / "phase2_resolution.pkl", "wb") as f:
            pickle.dump(data, f)
        self.metadata.last_phase = 2
        self._save_metadata()

    def save_phase3(self, buffer, embeddings_done: bool = False):
        """Save Phase 3 (buffer) state."""
        self._ensure_dir()
        data = {
            "buffer": buffer,
            "embeddings_done": embeddings_done
        }
        with open(self.checkpoint_dir / "phase3_buffer.pkl", "wb") as f:
            pickle.dump(data, f)
        self.metadata.last_phase = 3
        self._save_metadata()

    def load_phase1(self) -> Optional[Dict]:
        """Load Phase 1 data if it exists."""
        path = self.checkpoint_dir / "phase1_extraction.pkl"
        if not path.exists():
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def load_phase2(self) -> Optional[Dict]:
        """Load Phase 2 data if it exists."""
        path = self.checkpoint_dir / "phase2_resolution.pkl"
        if not path.exists():
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def load_phase3(self) -> Optional[Dict]:
        """Load Phase 3 data if it exists."""
        path = self.checkpoint_dir / "phase3_buffer.pkl"
        if not path.exists():
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def cleanup(self):
        """Remove checkpoint directory after successful completion."""
        if self.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_dir)

    def delete(self):
        """Alias for cleanup - used when --fresh is specified."""
        self.cleanup()

    def get_resume_phase(self) -> int:
        """Return the phase to resume from (last_phase + 1, or 1 if starting fresh)."""
        return self.metadata.last_phase + 1

    def print_status(self):
        """Print checkpoint status for user."""
        phase_names = {0: "Not started", 1: "Extraction", 2: "Resolution", 3: "Buffer ready"}
        print(f"  Found checkpoint: {self.checkpoint_dir}")
        print(f"    Input: {self.metadata.input_file}")
        print(f"    Last completed: Phase {self.metadata.last_phase} ({phase_names.get(self.metadata.last_phase, 'Unknown')})")
        print(f"    Started: {self.metadata.started_at}")
```

**Step 2: Verify file was created correctly**

Run: `python -c "from src.util.checkpoint import CheckpointManager; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add src/util/checkpoint.py
git commit -m "feat: add CheckpointManager for pipeline resume capability"
```

---

### Task 2: Add Batched Writes to bulk_write_all

**Files:**
- Modify: `src/pipeline.py:365-473` (bulk_write_all function)

**Step 1: Modify bulk_write_all to accept batch_size and batch all writes**

Replace the `bulk_write_all` function (lines 365-473) with:

```python
def bulk_write_all(buffer: BulkWriteBuffer, neo4j, embeddings, llm, batch_size: int = 250) -> Dict[str, int]:
    """Execute all buffered operations in batched queries.

    Args:
        buffer: The BulkWriteBuffer containing all operations
        neo4j: Neo4j client
        embeddings: Embeddings client
        llm: LLM client for summary merging
        batch_size: Maximum items per batch (default 250)
    """
    counts = {"entities": 0, "facts": 0, "relationships": 0, "topics": 0}

    # 1. Create DocumentNode (single MERGE)
    neo4j.query("""
        MERGE (d:DocumentNode {uuid: $uuid, group_id: $group_id})
        ON CREATE SET
            d.name = $name,
            d.document_date = $document_date,
            d.created_at = datetime()
    """, {
        "uuid": buffer.document_uuid,
        "name": buffer.document_name,
        "group_id": buffer.group_id,
        "document_date": buffer.document_date
    })

    # 2. Bulk create EpisodicNodes (batched)
    if buffer.episodic_nodes:
        total = len(buffer.episodic_nodes)
        for i in range(0, total, batch_size):
            batch = buffer.episodic_nodes[i:i + batch_size]
            neo4j.query("""
                UNWIND $nodes AS n
                MATCH (d:DocumentNode {uuid: $doc_uuid, group_id: $group_id})
                MERGE (e:EpisodicNode {uuid: n.uuid, group_id: $group_id})
                ON CREATE SET
                    e.content = n.content,
                    e.header_path = n.header_path,
                    e.document_date = $document_date,
                    e.created_at = datetime()
                ON MATCH SET
                    e.content = n.content,
                    e.header_path = n.header_path
                MERGE (d)-[:CONTAINS_CHUNK]->(e)
            """, {
                "nodes": batch,
                "doc_uuid": buffer.document_uuid,
                "group_id": buffer.group_id,
                "document_date": buffer.document_date
            })
        log(f"  Episodic: {total}/{total} ({(total + batch_size - 1) // batch_size} batches)")

    # 3. Bulk create new EntityNodes (batched)
    if buffer.entity_nodes:
        total = len(buffer.entity_nodes)
        for i in range(0, total, batch_size):
            batch = buffer.entity_nodes[i:i + batch_size]
            neo4j.query("""
                UNWIND $nodes AS n
                CREATE (e:EntityNode {
                    uuid: n.uuid,
                    name: n.name,
                    summary: n.summary,
                    group_id: n.group_id,
                    name_embedding: n.embedding,
                    name_only_embedding: n.name_only_embedding,
                    created_at: datetime()
                })
            """, {"nodes": batch})
        counts["entities"] = total
        log(f"  Entities: {total}/{total} ({(total + batch_size - 1) // batch_size} batches)")

    # 4. Update existing entity summaries (requires LLM merge) - batched
    if buffer.entity_updates:
        _bulk_update_entity_summaries(buffer.entity_updates, buffer.group_id, neo4j, llm, batch_size)

    # 5. Bulk create FactNodes (batched)
    if buffer.fact_nodes:
        total = len(buffer.fact_nodes)
        for i in range(0, total, batch_size):
            batch = buffer.fact_nodes[i:i + batch_size]
            neo4j.query("""
                UNWIND $nodes AS n
                MERGE (f:FactNode {uuid: n.uuid, group_id: n.group_id})
                ON CREATE SET
                    f.content = n.content,
                    f.embedding = n.embedding,
                    f.created_at = datetime()
                ON MATCH SET
                    f.content = n.content,
                    f.embedding = n.embedding
            """, {"nodes": batch})
        counts["facts"] = total
        log(f"  Facts: {total}/{total} ({(total + batch_size - 1) // batch_size} batches)")

        # Also index facts to Qdrant for semantic search (in batches)
        fact_store = get_fact_store()
        qdrant_facts = [
            {
                "fact_id": f["uuid"],
                "embedding": f["embedding"],
                "group_id": f["group_id"],
                "subject": f["subject"],
                "object": f["object"],
                "edge_type": f["edge_type"],
                "content": f["content"]
            }
            for f in buffer.fact_nodes
        ]
        fact_store.index_facts_batch(qdrant_facts)
        log(f"  Indexed {len(qdrant_facts)} facts to Qdrant")

    # 6. Bulk create TopicNodes (batched)
    if buffer.topic_nodes:
        total = len(buffer.topic_nodes)
        for i in range(0, total, batch_size):
            batch = buffer.topic_nodes[i:i + batch_size]
            neo4j.query("""
                UNWIND $nodes AS n
                MERGE (t:TopicNode {name: n.name, group_id: n.group_id})
                ON CREATE SET
                    t.uuid = n.uuid,
                    t.embedding = n.embedding,
                    t.created_at = datetime()
            """, {"nodes": batch})
        counts["topics"] = total
        log(f"  Topics: {total}/{total} ({(total + batch_size - 1) // batch_size} batches)")

    # 7. Bulk create relationships (grouped by type, batched)
    if buffer.relationships:
        counts["relationships"] = _bulk_create_relationships(buffer.relationships, neo4j, batch_size)

    return counts
```

**Step 2: Update _bulk_update_entity_summaries to accept batch_size**

Modify `_bulk_update_entity_summaries` function signature (line 476) to add batch_size parameter:

```python
def _bulk_update_entity_summaries(updates: List[Dict], group_id: str, neo4j, llm, batch_size: int = 250) -> None:
```

**Step 3: Update _bulk_create_relationships to batch writes**

Replace `_bulk_create_relationships` (lines 512-559) with:

```python
def _bulk_create_relationships(relationships: List[Dict], neo4j, batch_size: int = 250) -> int:
    """Create relationships in batches grouped by type."""
    # Group by relationship type
    by_type = defaultdict(list)
    for rel in relationships:
        by_type[rel["rel_type"]].append(rel)

    total = 0
    for rel_type, rels in by_type.items():
        # Separate rels with properties vs without
        with_props = [r for r in rels if r.get("properties")]
        without_props = [r for r in rels if not r.get("properties")]

        # Simple relationships (no properties) - batched
        if without_props:
            for i in range(0, len(without_props), batch_size):
                batch = without_props[i:i + batch_size]
                neo4j.query(f"""
                    UNWIND $rels AS r
                    MATCH (a {{uuid: r.from_uuid}}), (b {{uuid: r.to_uuid}})
                    MERGE (a)-[:{rel_type}]->(b)
                """, {"rels": batch})
            total += len(without_props)

        # Relationships with properties - batched
        if with_props:
            # If fact_id exists, MERGE on it to avoid duplicates on reruns
            with_fact_id = [r for r in with_props if r.get("properties", {}).get("fact_id")]
            without_fact_id = [r for r in with_props if r not in with_fact_id]

            if with_fact_id:
                for i in range(0, len(with_fact_id), batch_size):
                    batch = with_fact_id[i:i + batch_size]
                    neo4j.query(f"""
                        UNWIND $rels AS r
                        MATCH (a {{uuid: r.from_uuid}}), (b {{uuid: r.to_uuid}})
                        MERGE (a)-[rel:{rel_type} {{fact_id: r.properties.fact_id}}]->(b)
                        SET rel += r.properties
                    """, {"rels": batch})
                total += len(with_fact_id)

            if without_fact_id:
                for i in range(0, len(without_fact_id), batch_size):
                    batch = without_fact_id[i:i + batch_size]
                    neo4j.query(f"""
                        UNWIND $rels AS r
                        MATCH (a {{uuid: r.from_uuid}}), (b {{uuid: r.to_uuid}})
                        CREATE (a)-[rel:{rel_type}]->(b)
                        SET rel = r.properties
                    """, {"rels": batch})
                total += len(without_fact_id)

    total_batches = sum((len(rels) + batch_size - 1) // batch_size for rels in by_type.values())
    log(f"  Relationships: {total}/{total} ({total_batches} batches)")
    return total
```

**Step 4: Verify syntax**

Run: `python -m py_compile src/pipeline.py`
Expected: No output (success)

**Step 5: Commit**

```bash
git add src/pipeline.py
git commit -m "feat: add batched writes to bulk_write_all for connection stability"
```

---

### Task 3: Add CLI Flags and Checkpoint Integration

**Files:**
- Modify: `src/pipeline.py:1613-1680` (main function and CLI args)

**Step 1: Add import for CheckpointManager at top of file (after line 43)**

Add after the existing imports:

```python
from src.util.checkpoint import CheckpointManager
```

**Step 2: Add new CLI arguments (after line 1622)**

Add these arguments after the existing ones in the argument parser:

```python
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint (fail if none exists)")
    parser.add_argument("--fresh", action="store_true", help="Force fresh start (ignore existing checkpoint)")
    parser.add_argument("--batch-size", "-b", type=int, default=250, help="Neo4j write batch size (default: 250)")
```

**Step 3: Update process_file signature**

Update the `process_file` function signature (around line 1150) to add `batch_size` and `checkpoint_mgr` parameters:

```python
async def process_file(
    filename: str,
    group_id: str,
    limit: Optional[int],
    concurrency: int,
    similarity_threshold: float,
    resolve_concurrency: int,
    dedup_concurrency: int,
    neo4j,
    embeddings,
    dedup_embeddings,
    llm,
    entity_registry,
    topic_librarian,
    batch_size: int = 250,
    checkpoint_mgr: Optional[CheckpointManager] = None
) -> Dict[str, Any]:
```

**Step 4: Add checkpoint logic to main function**

This is a larger change. The main function needs to:
1. Check for existing checkpoints
2. Handle --resume and --fresh flags
3. Create CheckpointManager for each file
4. Pass batch_size to process_file

I'll provide the full updated main function in the implementation.

**Step 5: Add checkpoint save calls in process_file**

Add checkpoint saves after each phase completes:
- After Phase 1 (line ~1232): `checkpoint_mgr.save_phase1(...)`
- After Phase 2 (line ~1515): `checkpoint_mgr.save_phase2(...)`
- After Phase 3a (line ~1567): `checkpoint_mgr.save_phase3(buffer, embeddings_done=False)`
- After Phase 3b (line ~1574): `checkpoint_mgr.save_phase3(buffer, embeddings_done=True)`

**Step 6: Add checkpoint resume logic at start of process_file**

Check if resuming and load appropriate phase data.

**Step 7: Verify syntax**

Run: `python -m py_compile src/pipeline.py`
Expected: No output (success)

**Step 8: Commit**

```bash
git add src/pipeline.py
git commit -m "feat: add checkpoint save/resume and CLI flags for pipeline reliability"
```

---

### Task 4: Integration Testing

**Files:**
- Test manually with: `src/pipeline.py`

**Step 1: Test fresh run with small limit**

Run: `uv run src/pipeline.py --filter beige --limit 3 --batch-size 50`
Expected: Pipeline completes, checkpoint directory created then cleaned up

**Step 2: Test checkpoint creation (interrupt mid-run)**

Run: `uv run src/pipeline.py --filter beige --limit 10` then Ctrl+C after Phase 1
Expected: Checkpoint directory exists with `phase1_extraction.pkl` and `metadata.json`

**Step 3: Test resume**

Run: `uv run src/pipeline.py --filter beige --resume`
Expected: "Resuming from Phase 2..." message, pipeline continues

**Step 4: Test --fresh flag**

Run: `uv run src/pipeline.py --filter beige --fresh --limit 3`
Expected: "Removing existing checkpoint..." message, fresh start

**Step 5: Commit any fixes**

```bash
git add -A
git commit -m "fix: checkpoint integration fixes from testing"
```

---

### Task 5: Update CLAUDE.md with New CLI Options

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Add new CLI options to Commands section**

Add to the Commands section:

```markdown
# With checkpoint/resume options
uv run src/pipeline.py --resume                # Resume from last checkpoint
uv run src/pipeline.py --fresh                 # Force fresh start
uv run src/pipeline.py --batch-size 100        # Smaller batches for unstable connections
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add checkpoint and batching CLI options to CLAUDE.md"
```
