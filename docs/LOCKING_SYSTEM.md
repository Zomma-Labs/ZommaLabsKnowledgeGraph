# Similarity Locking System

## Overview
The **Similarity Locking System** is a critical component of the ZommaLabs Knowledge Graph pipeline. It prevents **Race Conditions** during parallel ingestion by ensuring that semantically identical entities/topics are processed sequentially, even if they have slightly different names (e.g., "Minneapolis" vs. "Minneapolis District").

## The Problem: Parallel Duplication
When multiple chunks are processed in parallel:
1.  **Thread A** sees "Inflation" -> Checks DB (Not Found) -> Starts Creating.
2.  **Thread B** sees "Inflation Rate" -> Checks DB (Not Found) -> Starts Creating.
3.  **Result**: Two duplicate nodes are created simultaneously.

## The Solution: Vector-Based Locking
Instead of locking on strict string names (which fails for "Inflation" vs "Inflation Rate"), we lock on **Semantic Embeddings**.

### Mechanism
1.  **Acquire Lock (`acquire_lock`)**:
    - Generates an embedding vector for the term.
    - Compares this vector against *all currently active locks* using **Cosine Similarity**.
    - If `Similarity > 0.90`: The new process **WAITS** (`asyncio.Event.wait()`) until the conflicting process releases its lock.
    - If No Conflict: The process registers its vector and proceeds.

2.  **Critical Section**:
    - The process identifies the entity, resolves it against FIBO/Graph, and performs an **Atomic Write** (`MERGE`) to Neo4j.
    - This ensures the node exists in the database *before* the lock is released.

3.  **Release Lock (`release_lock`)**:
    - The process releases the lock using a unique `lock_id` (UUID).
    - This wakes up any waiting processes.

### Key Components

-   **`src/util/similarity_lock.py`**: The singleton manager.
    -   Attributes: `_active_operations` (List ofDicts containing vector, event, id).
    -   Methods: `acquire_lock(vector, term)`, `release_lock(lock_id)`.

-   **`src/workflows/main_pipeline.py`**:
    -   Generates embedding for the entity name.
    -   Calls `SimilarityLockManager.acquire_lock`.
    -   Enters `try/finally` block to ensure `release_lock` is always called.

## Example Scenario

| Time | Thread A ("Minneapolis") | Thread B ("Minneapolis District") |
| :--- | :--- | :--- |
| T0 | **Acquires Lock** (Vector A) | Calculates Vector B |
| T1 | Processing... | Checks Lock. `Sim(A, B) > 0.9`. **WAITS**. |
| T2 | atomic `MERGE` to Neo4j | ... Waiting ... |
| T3 | **Releases Lock** | **Wakes Up** |
| T4 | Finished. | **Acquires Lock**. Checks DB -> **Finds Node!** |
| T5 | | **Merges** into existing node. |

## Implementation Details

```python
# From src/workflows/main_pipeline.py

# 1. Generate Embedding
name_embedding = await embeddings.embed_query(name)

# 2. Acquire Lock (Waits if similar exists)
lock_id = await SimilarityLockManager.acquire_lock(name_embedding, name)

try:
    # 3. Critical Section (Read & Write to DB)
    result = await resolve_single_item(...) 
finally:
    # 4. Release Lock (Always)
    await SimilarityLockManager.release_lock(lock_id)
```
