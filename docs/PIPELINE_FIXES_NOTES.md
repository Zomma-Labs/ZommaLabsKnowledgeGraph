# Pipeline Fixes Notes (Context Preservation)

Date: 2026-01-13

This file captures the key fixes and rationale for items (1) and (3):
- (1) Deep Research runtime issues
- (3) Ingestion idempotency + document date fallback

## 1) Deep Research runtime issues

### 1A. Embedding mismatch in `search_facts`
- **Problem**: `search_facts` used `LLMClient.get_embeddings(model="voyage-3-large")`, but the fact index is built with OpenAI `text-embedding-3-large` (3072 dims). This mismatches and can fail at runtime.
- **Fix**: Use the same embedding service as the fact index (`services.embeddings`) in `src/querying_system/mcp_server.py`.
- **Result**: `search_facts` queries align with the index, avoiding dimension/model mismatch.

### 1B. Missing `plan_research` in hybrid mode
- **Problem**: `HybridDeepResearchPipeline` calls `plan_research`, but no function existed.
- **Fix**: Added a lightweight planner in `src/querying_system/deep_research/supervisor.py`:
  - `plan_research(question)` returns a list of topic dicts `{topic, hints}` using a structured LLM response.
  - Fallback: if planning fails, return a single-topic plan.
- **Result**: Hybrid mode no longer throws runtime error.

## 3) Ingestion idempotency + document date fallback

### 3A. Stable IDs for document, chunk, and fact
- **Document UUID**: `document_uuid = _stable_uuid(group_id, document_name)`
- **Chunk UUID**:
  - If `chunk_id` exists in JSONL: `_stable_uuid(group_id, document_name, chunk_id)`
  - Else: `_stable_uuid(group_id, document_name, f"idx:{i}")`
- **Fact UUID** (in bulk path):
  - `_stable_uuid(group_id, episodic_uuid, subject.canonical_name, rel_type, object.canonical_name, fact.fact, fact.date_context)`
- **Note**: Entity UUIDs still come from resolution (existing graph + LLM). Idempotency relies on entity resolution matching existing nodes.

### 3B. MERGE instead of CREATE where needed
- `EpisodicNode`: `MERGE` on `(uuid, group_id)` + update content/header
- `FactNode`: `MERGE` on `(uuid, group_id)` + update content/embedding
- Relationships with properties:
  - If `fact_id` present, `MERGE` on `fact_id` to prevent duplicates.
  - Otherwise, `CREATE` (unchanged behavior).

### 3C. Document date fallback tightened
- `TemporalExtractor.extract_date` now returns **None** when no valid document date is found (no fallback to today's date).
- `process_file` now:
  - Falls back to `metadata.document_date` if LLM date is missing.
  - Uses `"Unknown"` only for **prompt context**, not for storage.
  - Stores `document_date` as `None` if truly unknown.

### 3D. Prompt date handling
- Extraction prompts use `document_date_for_prompt = document_date_str or "Unknown"` so facts still get a date context even if unknown.

## Files changed for (1) and (3)
- `src/querying_system/mcp_server.py` (search_facts embeddings)
- `src/querying_system/deep_research/supervisor.py` (plan_research)
- `src/agents/temporal_extractor.py` (date fallback)
- `src/pipeline.py` (stable UUIDs, MERGE writes, date handling)

## Remaining caveats
- If chunk ordering or chunk IDs change, stable chunk UUIDs will also change (expected).
- Entity UUID stability depends on resolution accuracy; if resolution fails, new entities can still be created.
- Re-indexing facts into Qdrant uses `upsert` by `fact_id` (stable).
