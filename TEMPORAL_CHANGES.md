# Temporal Support Changes

## Summary

Implemented date-aware search results approach (the "better" scalable approach) instead of relying only on document-based filtering.

## Changes Made

### 1. Pipeline: Propagate document_date to EpisodicNode

**File:** `src/pipeline.py`

**What changed:**
- Added `document_date` field to EpisodicNode creation (both single and bulk)
- Now chunks inherit the document date from their parent DocumentNode

**Status:** ✅ Complete

---

### 2. Schema: Added document_date to EpisodicNode

**File:** `src/schemas/nodes.py`

**What changed:**
- Added `document_date: Optional[str]` field to EpisodicNode class

**Status:** ✅ Complete

---

### 3. MCP Server: Date-aware search_relationships

**File:** `src/querying_system/mcp_server.py`

**What changed:**
- `_search_relationships_logic()` now:
  - Includes `document_date` in every result
  - Sorts results by date (newest first) by default
  - Accepts optional `date_from` and `date_to` parameters for filtering
- `search_relationships` MCP tool updated with date_from/date_to params

**New output format:**
```python
{
    "fact": "...",
    "score": 0.89,
    "subject": "...",
    "edge_type": "...",
    "object": "...",
    "chunk_id": "...",
    "header": "Cleveland > Labor Markets",
    "document_date": "2025-11-01"  # NEW
}
```

**Status:** ✅ Complete

---

### 4. Researcher: Date-aware search_facts

**File:** `src/querying_system/deep_research/researcher.py`

**What changed:**
- `search_facts` tool now:
  - Includes date in output prefix: `[DATE | Header]`
  - Accepts optional `date_from` and `date_to` parameters
  - Results sorted newest-first

**New output format:**
```
- [2025-11-01 | Cleveland > Labor Markets] Employment levels were flat... (score: 0.89)
- [2025-10-15 | Cleveland > Labor Markets] Employment levels increased... (score: 0.87)
```

**Status:** ✅ Complete

---

### 5. Prompts: Temporal guidance

**File:** `src/querying_system/deep_research/prompts.py`

**What changed:**
- RESEARCHER_SYSTEM_PROMPT: Added "Temporal Awareness" section explaining:
  - Search results include dates sorted newest-first
  - How to interpret dates in results
  - When to use date_from/date_to filtering

- SYNTHESIZER_SYSTEM_PROMPT: Added "Handle temporal information correctly" principle:
  - Use most recent facts when no time period specified
  - Recognize conflicting facts from different dates as change over time
  - How to handle "how did X change" questions

**Status:** ✅ Complete

---

## How the New Approach Works

### Agent sees dates naturally

When an agent calls `search_facts("Cleveland employment")`:

```
- [2025-11-01 | Cleveland > Labor] Employment flat... (score: 0.92)
- [2025-10-15 | Cleveland > Labor] Employment up... (score: 0.88)
```

The agent can:
1. See that there are two time periods
2. Notice the information differs (flat vs up)
3. Infer this represents a change over time
4. Use the appropriate one based on the question

### Optional date filtering

For questions about specific time periods:

```python
search_facts("Cleveland employment", date_from="2025-11-01")  # Only November
search_facts("Cleveland employment", date_to="2025-10-31")    # Only October and before
```

### Benefits

1. **Scales to thousands of documents** - no need to list them all
2. **Agent sees dates naturally** - in every result
3. **Fewer tool calls** - no need to call list_documents first
4. **Optional filtering** - date_from/date_to when needed
5. **Latest-first by default** - answers "current state" questions automatically

---

## Files Modified

| File | Change |
|------|--------|
| `src/pipeline.py` | document_date propagation |
| `src/schemas/nodes.py` | document_date field |
| `src/querying_system/mcp_server.py` | date in results, date_from/date_to filtering |
| `src/querying_system/deep_research/researcher.py` | date in output, date_from/date_to filtering |
| `src/querying_system/deep_research/prompts.py` | temporal guidance |
