#!/usr/bin/env python3
"""
MODULE: Pipeline
DESCRIPTION: Knowledge graph ingestion pipeline.

Transforms financial documents into a typed knowledge graph stored in Neo4j
using V2 Chain-of-Thought extraction.

Pipeline Phases:
    Phase 1: Parallel Extraction - Extract facts from all chunks concurrently
    Phase 2: Resolution - Resolve entities and topics against graph/ontology
    Phase 3: Assembly - Write nodes and relationships to Neo4j

Usage:
    uv run src/pipeline.py                          # Process all JSONL files
    uv run src/pipeline.py --input file.jsonl       # Process specific file
    uv run src/pipeline.py --limit 5                # Limit chunks for testing
    uv run src/pipeline.py --concurrency 10         # Adjust parallelism
    VERBOSE=true uv run src/pipeline.py             # Verbose mode
"""

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4, uuid5, NAMESPACE_DNS

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.extractor_v2 import ExtractorV2
from src.agents.entity_registry import EntityRegistry
from src.agents.topic_librarian import TopicLibrarian
from src.agents.temporal_extractor import TemporalExtractor
from src.util.entity_dedup import DeferredDeduplicationManager
from src.util.fact_vector_store import get_fact_store
from src.util.checkpoint import CheckpointManager
from src.schemas.extraction import ChainOfThoughtResult, TopicResolution, EnumeratedEntity

# =============================================================================
# Configuration
# =============================================================================

CHUNKS_DIR = "src/chunker/SAVED"
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"
DEFAULT_CONCURRENCY = int(os.getenv("EXTRACTION_CONCURRENCY", "100"))


def log(msg: str):
    """Print only if VERBOSE mode is enabled."""
    if VERBOSE:
        print(f"[Pipeline] {msg}")


def _stable_uuid(*parts: str) -> str:
    """Create a deterministic UUID from input parts."""
    key = "|".join([p for p in parts if p])
    return str(uuid5(NAMESPACE_DNS, key))


def _prepend_header_if_missing(text: str, header_path: str) -> str:
    """Prepend header levels to text if not already present.

    Some chunks have headers in the body (e.g., "New York\nEconomic activity...")
    while others only have them in metadata. This ensures consistency by
    prepending any header levels that aren't already in the text.
    """
    if not header_path:
        return text

    # Generic headers to skip (chunker artifacts)
    skip_headers = {"body", "text", "content", "main", "section"}

    # Split into header levels
    headers = [h.strip() for h in header_path.split(">") if h.strip()]

    if not headers:
        return text

    text_lower = text.lower()

    # Find headers NOT in text, excluding generic ones
    missing_headers = [
        h for h in headers
        if h.lower() not in text_lower and h.lower() not in skip_headers
    ]

    if not missing_headers:
        return text

    # Prepend missing headers
    prefix = "\n".join(missing_headers)
    return f"{prefix}\n{text}"


def _ensure_vector_indexes(neo4j) -> None:
    """Ensure required Neo4j vector indexes exist. Creates them if missing."""
    # Check for entity_name_embeddings index
    try:
        result = neo4j.query("SHOW INDEXES WHERE name = 'entity_name_embeddings'")
        if not result:
            print("    Creating 'entity_name_embeddings' index...")
            neo4j.query("""
                CREATE VECTOR INDEX entity_name_embeddings IF NOT EXISTS
                FOR (n:EntityNode)
                ON (n.name_embedding)
                OPTIONS {indexConfig: {
                  `vector.dimensions`: 3072,
                  `vector.similarity_function`: 'cosine'
                }}
            """)
            print("    ✓ Created entity_name_embeddings")
        else:
            print("    ✓ entity_name_embeddings exists")
    except Exception as e:
        print(f"    ⚠ Warning: Could not verify entity_name_embeddings index: {e}")

    # Check for entity_name_only_embeddings index (for direct name lookup)
    try:
        result = neo4j.query("SHOW INDEXES WHERE name = 'entity_name_only_embeddings'")
        if not result:
            print("    Creating 'entity_name_only_embeddings' index...")
            neo4j.query("""
                CREATE VECTOR INDEX entity_name_only_embeddings IF NOT EXISTS
                FOR (n:EntityNode)
                ON (n.name_only_embedding)
                OPTIONS {indexConfig: {
                  `vector.dimensions`: 3072,
                  `vector.similarity_function`: 'cosine'
                }}
            """)
            print("    ✓ Created entity_name_only_embeddings")
        else:
            print("    ✓ entity_name_only_embeddings exists")
    except Exception as e:
        print(f"    ⚠ Warning: Could not verify entity_name_only_embeddings index: {e}")

    # Check for topic_embeddings index
    try:
        result = neo4j.query("SHOW INDEXES WHERE name = 'topic_embeddings'")
        if not result:
            print("    Creating 'topic_embeddings' index...")
            neo4j.query("""
                CREATE VECTOR INDEX topic_embeddings IF NOT EXISTS
                FOR (n:TopicNode)
                ON (n.embedding)
                OPTIONS {indexConfig: {
                  `vector.dimensions`: 3072,
                  `vector.similarity_function`: 'cosine'
                }}
            """)
            print("    ✓ Created topic_embeddings")
        else:
            print("    ✓ topic_embeddings exists")
    except Exception as e:
        print(f"    ⚠ Warning: Could not verify topic_embeddings index: {e}")


# =============================================================================
# Neo4j Operations
# =============================================================================

def create_document_node(neo4j, uuid: str, name: str, group_id: str, document_date: Optional[datetime]) -> None:
    """Create DocumentNode if it doesn't exist."""
    neo4j.query("""
        MERGE (d:DocumentNode {uuid: $uuid, group_id: $group_id})
        ON CREATE SET
            d.name = $name,
            d.document_date = $document_date,
            d.created_at = datetime()
    """, {
        "uuid": uuid,
        "name": name,
        "group_id": group_id,
        "document_date": document_date.isoformat() if document_date else None
    })


def create_episodic_node(neo4j, uuid: str, document_uuid: str, content: str, header_path: str, group_id: str, document_date: Optional[str] = None) -> None:
    """Create EpisodicNode and link to DocumentNode."""
    neo4j.query("""
        MATCH (d:DocumentNode {uuid: $document_uuid, group_id: $group_id})
        CREATE (e:EpisodicNode {
            uuid: $uuid,
            content: $content,
            header_path: $header_path,
            group_id: $group_id,
            document_date: $document_date,
            created_at: datetime()
        })
        CREATE (d)-[:CONTAINS_CHUNK]->(e)
    """, {
        "uuid": uuid,
        "document_uuid": document_uuid,
        "content": content,
        "header_path": header_path,
        "group_id": group_id,
        "document_date": document_date
    })


def create_entity_node(neo4j, uuid: str, name: str, summary: str, group_id: str, embedding: List[float]) -> None:
    """Create new EntityNode."""
    neo4j.query("""
        CREATE (e:EntityNode {
            uuid: $uuid,
            name: $name,
            summary: $summary,
            group_id: $group_id,
            name_embedding: $embedding,
            created_at: datetime()
        })
    """, {
        "uuid": uuid,
        "name": name,
        "summary": summary,
        "group_id": group_id,
        "embedding": embedding
    })


def update_entity_summary(neo4j, uuid: str, group_id: str, new_summary: str, llm) -> None:
    """Update existing EntityNode summary with LLM merge."""
    # Get existing - include group_id for safety
    existing = neo4j.query(
        "MATCH (e:EntityNode {uuid: $uuid, group_id: $group_id}) RETURN e.summary as summary",
        {"uuid": uuid, "group_id": group_id}
    )

    if not existing:
        log(f"Warning: EntityNode {uuid} not found in group {group_id}")
        return

    old_summary = existing[0].get("summary", "")

    # Skip if new info already in summary
    if not new_summary or new_summary in old_summary:
        return

    # LLM merge
    merged = _merge_summaries(old_summary, new_summary, llm)

    neo4j.query("""
        MATCH (e:EntityNode {uuid: $uuid, group_id: $group_id})
        SET e.summary = $summary
    """, {"uuid": uuid, "group_id": group_id, "summary": merged})


def create_fact_node(neo4j, uuid: str, content: str, group_id: str, embedding: List[float]) -> None:
    """Create FactNode."""
    neo4j.query("""
        CREATE (f:FactNode {
            uuid: $uuid,
            content: $content,
            group_id: $group_id,
            embedding: $embedding,
            created_at: datetime()
        })
    """, {
        "uuid": uuid,
        "content": content,
        "group_id": group_id,
        "embedding": embedding
    })


def create_relationship(neo4j, from_uuid: str, to_uuid: str, rel_type: str, properties: Optional[Dict] = None) -> None:
    """Create relationship between nodes."""
    props_str = ""
    params = {"from_uuid": from_uuid, "to_uuid": to_uuid}

    if properties:
        props_str = " {" + ", ".join([f"{k}: ${k}" for k in properties.keys()]) + "}"
        params.update(properties)

    neo4j.query(f"""
        MATCH (a {{uuid: $from_uuid}}), (b {{uuid: $to_uuid}})
        MERGE (a)-[r:{rel_type}{props_str}]->(b)
    """, params)


def get_or_create_topic_node(neo4j, label: str, group_id: str, embeddings) -> str:
    """Get existing TopicNode or create new one. Returns UUID."""
    result = neo4j.query(
        "MATCH (t:TopicNode {name: $label, group_id: $group_id}) RETURN t.uuid as uuid",
        {"label": label, "group_id": group_id}
    )

    if result:
        return result[0]["uuid"]

    topic_uuid = str(uuid4())
    embedding = embeddings.embed_query(label)

    neo4j.query("""
        CREATE (t:TopicNode {
            uuid: $uuid,
            name: $label,
            group_id: $group_id,
            embedding: $embedding,
            created_at: datetime()
        })
    """, {"uuid": topic_uuid, "label": label, "group_id": group_id, "embedding": embedding})

    return topic_uuid


def _normalize_rel_type(description: str) -> str:
    """Convert free-form relationship to UPPER_SNAKE_CASE."""
    words = description.upper().split()[:8]
    normalized = "_".join(words)
    normalized = "".join(c if c.isalnum() or c == "_" else "_" for c in normalized)
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_") or "RELATED_TO"


def _merge_summaries(old: str, new: str, llm) -> str:
    """Use LLM to merge entity summaries."""
    if not old.strip():
        return new
    if not new.strip():
        return old

    try:
        result = llm.invoke(f"""Merge into one concise summary (3-5 sentences max):

EXISTING: {old}
NEW: {new}

Combined:""")
        return result.content.strip()
    except Exception:
        return f"{old}\n{new}"


# =============================================================================
# Bulk Write Buffer
# =============================================================================

@dataclass
class BulkWriteBuffer:
    """Collects all Neo4j operations for batch execution."""
    document_uuid: str
    document_name: str
    document_date: Optional[str]
    group_id: str

    episodic_nodes: List[Dict] = field(default_factory=list)
    entity_nodes: List[Dict] = field(default_factory=list)
    entity_updates: List[Dict] = field(default_factory=list)  # {uuid, new_summary}
    fact_nodes: List[Dict] = field(default_factory=list)
    topic_nodes: List[Dict] = field(default_factory=list)
    relationships: List[Dict] = field(default_factory=list)

    # Track created topics to avoid duplicates
    _created_topics: Dict[str, str] = field(default_factory=dict)  # label -> uuid


def bulk_write_all(buffer: BulkWriteBuffer, neo4j, embeddings, llm, batch_size: int = 250) -> Dict[str, int]:
    """Execute all buffered operations in batched queries."""
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
        num_batches = (total + batch_size - 1) // batch_size
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
        log(f"  Episodic nodes: {total}/{total} ({num_batches} batches)")

    # 3. Bulk create new EntityNodes (batched)
    if buffer.entity_nodes:
        total = len(buffer.entity_nodes)
        num_batches = (total + batch_size - 1) // batch_size
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
        log(f"  Entities: {total}/{total} ({num_batches} batches)")

    # 4. Update existing entity summaries (requires LLM merge)
    if buffer.entity_updates:
        _bulk_update_entity_summaries(buffer.entity_updates, buffer.group_id, neo4j, llm, batch_size=batch_size)

    # 5. Bulk create FactNodes (batched)
    if buffer.fact_nodes:
        total = len(buffer.fact_nodes)
        num_batches = (total + batch_size - 1) // batch_size
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
        log(f"  Facts: {total}/{total} ({num_batches} batches)")

        # Also index facts to Qdrant for semantic search
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
        num_batches = (total + batch_size - 1) // batch_size
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
        log(f"  Topics: {total}/{total} ({num_batches} batches)")

    # 7. Bulk create relationships (grouped by type)
    if buffer.relationships:
        counts["relationships"] = _bulk_create_relationships(buffer.relationships, neo4j, batch_size=batch_size)

    return counts


def _bulk_update_entity_summaries(updates: List[Dict], group_id: str, neo4j, llm, batch_size: int = 250) -> None:
    """Batch read existing summaries, LLM merge, batch write."""
    if not updates:
        return

    # Batch read all existing summaries
    uuids = [u["uuid"] for u in updates]
    existing = neo4j.query("""
        UNWIND $uuids AS uuid
        MATCH (e:EntityNode {uuid: uuid, group_id: $group_id})
        RETURN e.uuid AS uuid, e.summary AS summary
    """, {"uuids": uuids, "group_id": group_id})

    existing_map = {r["uuid"]: r["summary"] or "" for r in existing}

    # Merge summaries with LLM
    merged_updates = []
    for update in updates:
        old_summary = existing_map.get(update["uuid"], "")
        new_summary = update["new_summary"]

        if not new_summary or new_summary in old_summary:
            continue

        merged = _merge_summaries(old_summary, new_summary, llm)
        merged_updates.append({"uuid": update["uuid"], "summary": merged})

    # Batch write merged summaries
    if merged_updates:
        total = len(merged_updates)
        num_batches = (total + batch_size - 1) // batch_size
        for i in range(0, total, batch_size):
            batch = merged_updates[i:i + batch_size]
            neo4j.query("""
                UNWIND $updates AS u
                MATCH (e:EntityNode {uuid: u.uuid, group_id: $group_id})
                SET e.summary = u.summary
            """, {"updates": batch, "group_id": group_id})
        log(f"  Entity summary updates: {total}/{total} ({num_batches} batches)")


def _bulk_create_relationships(relationships: List[Dict], neo4j, batch_size: int = 250) -> int:
    """Create relationships in batches grouped by type."""
    # Group by relationship type
    by_type = defaultdict(list)
    for rel in relationships:
        by_type[rel["rel_type"]].append(rel)

    total = 0
    total_batches = 0
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
                total_batches += 1
            total += len(without_props)

        # Relationships with properties - use CREATE to ensure each fact gets its own edge
        # (MERGE would dedupe on (a, rel_type, b) and overwrite properties including fact_id)
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
                    total_batches += 1
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
                    total_batches += 1
                total += len(without_fact_id)

    log(f"  Relationships: {total}/{total} ({total_batches} batches)")
    return total


# =============================================================================
# Phase 1: Parallel Extraction
# =============================================================================

async def extract_chunk(
    extractor: ExtractorV2,
    chunk_idx: int,
    chunk_text: str,
    header_path: str,
    document_date: str,
    chunk_uuid: str,
    semaphore: asyncio.Semaphore
) -> Dict[str, Any]:
    """Extract facts from a single chunk using V2 chain-of-thought."""
    async with semaphore:
        try:
            result = await asyncio.to_thread(
                extractor.extract,
                chunk_text=chunk_text,
                header_path=header_path,
                document_date=document_date
            )
            log(f"  [Extract] Chunk {chunk_idx}: {len(result.entities)} entities, {len(result.facts)} facts")
            return {
                "success": True,
                "chunk_idx": chunk_idx,
                "extraction": result,
                "chunk_text": chunk_text,
                "header_path": header_path,
                "chunk_uuid": chunk_uuid
            }
        except Exception as e:
            print(f"  [Extract] Chunk {chunk_idx}: ERROR - {e}")
            return {
                "success": False,
                "chunk_idx": chunk_idx,
                "error": str(e),
                "extraction": ChainOfThoughtResult(entities=[], facts=[]),
                "chunk_text": chunk_text,
                "header_path": header_path,
                "chunk_uuid": chunk_uuid
            }


# =============================================================================
# Phase 2: Resolution
# =============================================================================

def resolve_entities(extraction: ChainOfThoughtResult, episodic_uuid: str, entity_registry: EntityRegistry) -> Dict[str, Any]:
    """Resolve all entities from an extraction against the graph."""
    lookup = {}
    for entity in extraction.entities:
        if entity.entity_type.lower() == "topic":
            continue
        resolution = entity_registry.resolve(
            entity_name=entity.name,
            entity_type=entity.entity_type,
            entity_summary=entity.summary,
            chunk_uuid=episodic_uuid
        )
        lookup[entity.name] = resolution
    return lookup


def resolve_topics(extraction: ChainOfThoughtResult, chunk_text: str, topic_librarian: TopicLibrarian) -> Dict[str, Dict]:
    """Resolve all topics from an extraction against the ontology.

    This includes:
    - Topics from fact.topics list
    - Subjects/objects typed as 'topic' (e.g., 'Economic activity')
    """
    all_topics = []
    for fact in extraction.facts:
        # Add topics from the topics list
        all_topics.extend(fact.topics)
        # Also add subjects/objects that are typed as Topic
        if fact.subject_type.lower() == "topic":
            all_topics.append(fact.subject)
        if fact.object_type.lower() == "topic":
            all_topics.append(fact.object)

    unique_topics = list(set(all_topics))

    if not unique_topics:
        return {}

    definitions = topic_librarian.batch_define_topics(unique_topics, chunk_text)

    lookup = {}
    for topic in unique_topics:
        enriched = definitions.get(topic, topic)
        match = topic_librarian.resolve(topic, enriched_text=enriched, context=chunk_text)
        if match:
            # Store with lowercase key for case-insensitive lookup
            lookup[topic.lower()] = match

    return lookup


# =============================================================================
# Phase 3: Assembly (Bulk Write)
# =============================================================================

def collect_chunk(
    extraction: ChainOfThoughtResult,
    entity_lookup: Dict[str, Any],
    topic_lookup: Dict[str, TopicResolution],
    episodic_uuid: str,
    chunk_text: str,
    header_path: str,
    buffer: BulkWriteBuffer,
) -> Dict[str, int]:
    """Collect all nodes and relationships into buffer for bulk write.

    Note: Both EntityResolution and TopicResolution have .canonical_name and .uuid
    properties, so they can be used interchangeably in the fact pattern.

    IMPORTANT: This function does NOT generate embeddings. Embeddings are generated
    in batch after all chunks are collected (see batch_generate_embeddings).
    """
    entity_count = 0

    # Track which entities we've already added (for this chunk)
    seen_entities = set()

    # Collect EpisodicNode
    buffer.episodic_nodes.append({
        "uuid": episodic_uuid,
        "content": chunk_text,
        "header_path": header_path
    })

    # Collect EntityNodes (without embeddings - added later in batch)
    for entity_name, resolution in entity_lookup.items():
        if resolution.uuid in seen_entities:
            continue
        seen_entities.add(resolution.uuid)

        if resolution.is_new:
            # Check if we've already added this entity to buffer
            existing_uuids = {e["uuid"] for e in buffer.entity_nodes}
            if resolution.uuid not in existing_uuids:
                buffer.entity_nodes.append({
                    "uuid": resolution.uuid,
                    "name": resolution.canonical_name,
                    "summary": resolution.updated_summary,
                    "group_id": buffer.group_id,
                    "embedding": None,  # Will be filled by batch_generate_embeddings
                    "name_only_embedding": None  # Will be filled by batch_generate_embeddings
                })
                entity_count += 1
        else:
            # Queue summary update (will be deduplicated later)
            buffer.entity_updates.append({
                "uuid": resolution.uuid,
                "new_summary": resolution.updated_summary
            })

    # Collect TopicNodes (without embeddings - added later in batch)
    # Topic resolution ensures consistent UUIDs per canonical_label (checked against Neo4j)
    for topic_name, topic_res in topic_lookup.items():
        topic_label = topic_res.canonical_label

        # Only create TopicNode if not already in buffer (handles dedup across chunks)
        # Also skip if topic already exists in graph (is_new=False means it was found in Neo4j)
        if topic_label not in buffer._created_topics and topic_res.is_new:
            buffer._created_topics[topic_label] = topic_res.uuid
            buffer.topic_nodes.append({
                "uuid": topic_res.uuid,
                "name": topic_label,
                "group_id": buffer.group_id,
                "embedding": None  # Will be filled by batch_generate_embeddings
            })

    # Collect FactNodes and relationships (without embeddings - added later in batch)
    fact_count = 0
    rel_count = 0

    for fact in extraction.facts:
        # Subject -> Episodic -> Object
        # Lookups use lowercase keys for case-insensitive matching
        if fact.subject_type.lower() == "topic":
            subject = topic_lookup.get(fact.subject.lower())
        else:
            subject = entity_lookup.get(fact.subject.lower())

        if fact.object_type.lower() == "topic":
            obj = topic_lookup.get(fact.object.lower())
        else:
            obj = entity_lookup.get(fact.object.lower())

        # Skip fact entirely if subject or object can't be resolved
        # This ensures we don't have orphaned facts or incomplete relationship patterns
        if not subject or not obj:
            missing = []
            if not subject:
                missing.append(f"subject '{fact.subject}' ({fact.subject_type})")
            if not obj:
                missing.append(f"object '{fact.object}' ({fact.object_type})")
            log(f"Skipping fact due to unresolved {' and '.join(missing)}: {fact.fact[:60]}...")
            continue

        rel_type = _normalize_rel_type(fact.relationship)
        fact_uuid = _stable_uuid(
            buffer.group_id,
            episodic_uuid,
            subject.canonical_name,
            rel_type,
            obj.canonical_name,
            fact.fact,
            fact.date_context or "",
        )

        buffer.fact_nodes.append({
            "uuid": fact_uuid,
            "content": fact.fact,
            "group_id": buffer.group_id,
            "embedding": None,  # Will be filled by batch_generate_embeddings
            # Metadata for Qdrant indexing
            "subject": subject.canonical_name,
            "object": obj.canonical_name,
            "edge_type": rel_type
        })
        fact_count += 1

        # Fact -> Episodic
        buffer.relationships.append({
            "from_uuid": episodic_uuid,
            "to_uuid": fact_uuid,
            "rel_type": "CONTAINS_FACT",
            "properties": None
        })
        rel_count += 1

        # Both subject and obj are guaranteed to exist at this point
        props = {
            "fact_id": fact_uuid,
            "description": fact.relationship,
            "date_context": fact.date_context or ""
        }

        buffer.relationships.append({
            "from_uuid": subject.uuid,
            "to_uuid": episodic_uuid,
            "rel_type": rel_type,
            "properties": props
        })
        buffer.relationships.append({
            "from_uuid": episodic_uuid,
            "to_uuid": obj.uuid,
            "rel_type": f"{rel_type}_TARGET",
            "properties": props
        })
        rel_count += 2

        # Topics (from fact.topics list - creates DISCUSSES relationships)
        for topic_name in fact.topics:
            topic_res = topic_lookup.get(topic_name.lower())
            if topic_res:
                topic_label = topic_res.canonical_label

                # Get or create topic UUID (use resolved uuid)
                if topic_label not in buffer._created_topics:
                    topic_uuid = topic_res.uuid
                    buffer._created_topics[topic_label] = topic_uuid
                    buffer.topic_nodes.append({
                        "uuid": topic_uuid,
                        "name": topic_label,
                        "group_id": buffer.group_id,
                        "embedding": None  # Will be filled by batch_generate_embeddings
                    })
                else:
                    topic_uuid = buffer._created_topics[topic_label]

                buffer.relationships.append({
                    "from_uuid": episodic_uuid,
                    "to_uuid": topic_uuid,
                    "rel_type": "DISCUSSES",
                    "properties": None
                })
                rel_count += 1

    return {"entities": entity_count, "facts": fact_count, "relationships": rel_count}


async def batch_generate_embeddings(buffer: BulkWriteBuffer, embeddings, dedup_embeddings, batch_size: int = 128, concurrency: int = 10) -> None:
    """Generate all embeddings in parallel batches.

    Args:
        buffer: The bulk write buffer containing nodes that need embeddings
        embeddings: voyage-finance-2 embeddings client (for entities/topics)
        dedup_embeddings: voyage-3-large embeddings client (for facts)
        batch_size: Max texts per embedding API call (Voyage limit is ~128)
        concurrency: Max concurrent embedding API calls per embedding type
    """
    import time

    # Collect all texts that need embeddings (with indices for mapping back)
    entity_texts = []  # name + summary
    entity_name_texts = []  # name only
    entity_indices = []  # which entities have valid texts
    topic_texts = []
    topic_indices = []
    fact_texts = []
    fact_indices = []

    for i, entity in enumerate(buffer.entity_nodes):
        text = f"{entity['name']}: {entity['summary']}"
        name_text = entity['name'] or ""
        if text.strip() and name_text.strip():
            entity_texts.append(text)
            entity_name_texts.append(name_text)
            entity_indices.append(i)

    for i, topic in enumerate(buffer.topic_nodes):
        text = topic['name'] or ""
        if text.strip():
            topic_texts.append(text)
            topic_indices.append(i)

    for i, fact in enumerate(buffer.fact_nodes):
        text = fact['content'] or ""
        if text.strip():
            fact_texts.append(text)
            fact_indices.append(i)

    log(f"Generating embeddings: {len(entity_texts)} entities, {len(topic_texts)} topics, {len(fact_texts)} facts")

    async def embed_in_batches_async(texts: List[str], embed_fn, batch_sz: int, sem: asyncio.Semaphore) -> List[List[float]]:
        """Embed texts in parallel batches with concurrency control."""
        if not texts:
            return []

        async def embed_batch(batch: List[str], batch_idx: int) -> tuple:
            async with sem:
                try:
                    result = await asyncio.to_thread(embed_fn.embed_documents, batch)
                    return batch_idx, result
                except Exception as e:
                    if "rate limit" in str(e).lower():
                        log(f"Rate limited on batch {batch_idx}, waiting 30s...")
                        await asyncio.sleep(30)
                        result = await asyncio.to_thread(embed_fn.embed_documents, batch)
                        return batch_idx, result
                    raise

        # Create all batch tasks
        batches = [(texts[i:i + batch_sz], i // batch_sz) for i in range(0, len(texts), batch_sz)]
        tasks = [embed_batch(batch, idx) for batch, idx in batches]

        # Run all batches concurrently (limited by semaphore)
        results = await asyncio.gather(*tasks)

        # Sort by batch index and flatten
        results.sort(key=lambda x: x[0])
        all_embeddings = []
        for _, batch_embeddings in results:
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    # Create semaphores for concurrency control (separate per embedding client to avoid conflicts)
    sem_main = asyncio.Semaphore(concurrency)
    sem_dedup = asyncio.Semaphore(concurrency)

    # Run all 4 embedding types in parallel
    async def embed_entities():
        if not entity_texts:
            return [], []
        # Run entity text and entity name embeddings in parallel
        entity_emb, entity_name_emb = await asyncio.gather(
            embed_in_batches_async(entity_texts, embeddings, batch_size, sem_main),
            embed_in_batches_async(entity_name_texts, embeddings, batch_size, sem_main)
        )
        return entity_emb, entity_name_emb

    async def embed_topics():
        if not topic_texts:
            return []
        return await embed_in_batches_async(topic_texts, embeddings, batch_size, sem_main)

    async def embed_facts():
        if not fact_texts:
            return []
        return await embed_in_batches_async(fact_texts, dedup_embeddings, batch_size, sem_dedup)

    # Execute all embedding tasks in parallel
    (entity_embeddings, entity_name_embeddings), topic_embeddings, fact_embeddings = await asyncio.gather(
        embed_entities(),
        embed_topics(),
        embed_facts()
    )

    # Map embeddings back to buffer
    for j, idx in enumerate(entity_indices):
        buffer.entity_nodes[idx]['embedding'] = entity_embeddings[j]
        buffer.entity_nodes[idx]['name_only_embedding'] = entity_name_embeddings[j]

    for j, idx in enumerate(topic_indices):
        buffer.topic_nodes[idx]['embedding'] = topic_embeddings[j]

    for j, idx in enumerate(fact_indices):
        buffer.fact_nodes[idx]['embedding'] = fact_embeddings[j]

    log(f"Embeddings generated successfully")


# =============================================================================
# Phase 3: Assembly (Legacy - Individual Writes)
# =============================================================================

def assemble_chunk(
    extraction: ChainOfThoughtResult,
    entity_lookup: Dict[str, Any],
    topic_lookup: Dict[str, Dict],
    document_uuid: str,
    episodic_uuid: str,
    chunk_text: str,
    header_path: str,
    group_id: str,
    neo4j,
    embeddings,
    llm
) -> Dict[str, int]:
    """Write all nodes and relationships to Neo4j for a chunk."""
    entity_count = 0

    # Create EntityNodes
    for entity_name, resolution in entity_lookup.items():
        if resolution.is_new:
            embed_text = f"{resolution.canonical_name}: {resolution.updated_summary}"
            embedding = embeddings.embed_query(embed_text)
            create_entity_node(
                neo4j, resolution.uuid, resolution.canonical_name,
                resolution.updated_summary, group_id, embedding
            )
            entity_count += 1
        else:
            update_entity_summary(neo4j, resolution.uuid, group_id, resolution.updated_summary, llm)

    # Create FactNodes and relationships
    fact_count = 0
    rel_count = 0

    for fact in extraction.facts:
        rel_type = _normalize_rel_type(fact.relationship)
        fact_uuid = _stable_uuid(
            group_id,
            episodic_uuid,
            fact.subject,
            rel_type,
            fact.object,
            fact.fact,
            fact.date_context or "",
        )
        fact_embedding = embeddings.embed_query(fact.fact)
        create_fact_node(neo4j, fact_uuid, fact.fact, group_id, fact_embedding)
        fact_count += 1

        # Fact -> Episodic
        create_relationship(neo4j, episodic_uuid, fact_uuid, "CONTAINS_FACT")
        rel_count += 1

        # Subject -> Episodic -> Object
        subject = entity_lookup.get(fact.subject)
        obj = entity_lookup.get(fact.object)

        if subject and obj:
            props = {"fact_id": fact_uuid, "description": fact.relationship, "date_context": fact.date_context or ""}

            create_relationship(neo4j, subject.uuid, episodic_uuid, rel_type, props)
            create_relationship(neo4j, episodic_uuid, obj.uuid, f"{rel_type}_TARGET", props)
            rel_count += 2

        # Topics
        for topic_name in fact.topics:
            if topic_name in topic_lookup:
                topic_uuid = get_or_create_topic_node(
                    neo4j, topic_lookup[topic_name]["label"], group_id, embeddings
                )
                create_relationship(neo4j, episodic_uuid, topic_uuid, "DISCUSSES")
                rel_count += 1

    return {"entities": entity_count, "facts": fact_count, "relationships": rel_count}


# =============================================================================
# Chunk Processing
# =============================================================================

def process_chunk(
    extraction_result: Dict[str, Any],
    document_uuid: str,
    document_name: str,
    document_date: Optional[datetime],
    group_id: str,
    neo4j, embeddings, llm,
    entity_registry: EntityRegistry,
    topic_librarian: TopicLibrarian
) -> Dict[str, Any]:
    """Process a single extraction through resolution and assembly."""
    chunk_idx = extraction_result["chunk_idx"]
    extraction = extraction_result["extraction"]
    chunk_text = extraction_result["chunk_text"]
    header_path = extraction_result["header_path"]

    if not extraction_result["success"]:
        return {"success": False, "chunk_idx": chunk_idx, "error": extraction_result.get("error")}

    if not extraction.facts:
        return {"success": True, "chunk_idx": chunk_idx, "facts": 0, "entities": 0, "relationships": 0}

    try:
        episodic_uuid = extraction_result.get("chunk_uuid") or str(uuid4())

        # Resolution
        entity_lookup = resolve_entities(extraction, episodic_uuid, entity_registry)
        topic_lookup = resolve_topics(extraction, chunk_text, topic_librarian)

        # Create document and episodic nodes
        create_document_node(neo4j, document_uuid, document_name, group_id, document_date)
        create_episodic_node(neo4j, episodic_uuid, document_uuid, chunk_text, header_path, group_id)

        # Assembly
        counts = assemble_chunk(
            extraction, entity_lookup, topic_lookup,
            document_uuid, episodic_uuid, chunk_text, header_path,
            group_id, neo4j, embeddings, llm
        )

        log(f"  [Chunk {chunk_idx}] {counts['facts']} facts, {counts['entities']} entities, {counts['relationships']} rels")
        return {"success": True, "chunk_idx": chunk_idx, **counts}

    except Exception as e:
        print(f"  [Chunk {chunk_idx}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "chunk_idx": chunk_idx, "error": str(e)}


def process_chunk_with_lookup(
    extraction_result: Dict[str, Any],
    document_uuid: str,
    document_name: str,
    document_date: Optional[datetime],
    group_id: str,
    entity_lookup_global: Dict[str, Any],
    neo4j, embeddings, llm,
    topic_librarian: TopicLibrarian
) -> Dict[str, Any]:
    """Process a single extraction using pre-resolved entity lookup."""
    chunk_idx = extraction_result["chunk_idx"]
    extraction = extraction_result["extraction"]
    chunk_text = extraction_result["chunk_text"]
    header_path = extraction_result["header_path"]

    if not extraction_result["success"]:
        return {"success": False, "chunk_idx": chunk_idx, "error": extraction_result.get("error")}

    if not extraction.facts:
        return {"success": True, "chunk_idx": chunk_idx, "facts": 0, "entities": 0, "relationships": 0}

    try:
        episodic_uuid = extraction_result.get("chunk_uuid") or str(uuid4())

        # Build chunk-specific entity lookup from global pre-resolved lookup
        entity_lookup = {}
        for entity in extraction.entities:
            if entity.entity_type.lower() != "topic" and entity.name in entity_lookup_global:
                entity_lookup[entity.name] = entity_lookup_global[entity.name]

        # Topic resolution (still per-chunk for now)
        topic_lookup = resolve_topics(extraction, chunk_text, topic_librarian)

        # Create document and episodic nodes
        create_document_node(neo4j, document_uuid, document_name, group_id, document_date)
        create_episodic_node(neo4j, episodic_uuid, document_uuid, chunk_text, header_path, group_id)

        # Assembly
        counts = assemble_chunk(
            extraction, entity_lookup, topic_lookup,
            document_uuid, episodic_uuid, chunk_text, header_path,
            group_id, neo4j, embeddings, llm
        )

        log(f"  [Chunk {chunk_idx}] {counts['facts']} facts, {counts['entities']} entities, {counts['relationships']} rels")
        return {"success": True, "chunk_idx": chunk_idx, **counts}

    except Exception as e:
        print(f"  [Chunk {chunk_idx}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "chunk_idx": chunk_idx, "error": str(e)}


# =============================================================================
# File Processing
# =============================================================================

async def process_file(
    filepath: str,
    group_id: str,
    limit: Optional[int],
    concurrency: int,
    similarity_threshold: float,
    resolve_concurrency: int,
    dedup_concurrency: int,
    neo4j, embeddings, dedup_embeddings, llm,
    entity_registry: EntityRegistry,
    topic_librarian: TopicLibrarian,
    batch_size: int = 250,
    checkpoint_mgr: Optional[CheckpointManager] = None
) -> Dict[str, Any]:
    """Process all chunks in a JSONL file."""
    filename = os.path.basename(filepath)
    print(f"\nProcessing: {filename}")

    document_name = filename.replace(".jsonl", "")
    document_uuid = _stable_uuid(group_id, document_name)

    # Load chunks
    chunks = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = data.get("body", "")
            if not text:
                continue

            metadata = data.get("metadata", {})
            breadcrumbs = data.get("breadcrumbs") or metadata.get("headings", [])
            header_path = " > ".join(breadcrumbs) if breadcrumbs else ""

            # Prepend header to text if not already present (ensures consistent extraction)
            text = _prepend_header_if_missing(text, header_path)

            doc_date = None
            if "document_date" in metadata:
                try:
                    doc_date = datetime.fromisoformat(metadata["document_date"])
                except (ValueError, TypeError):
                    pass

            chunk_id = data.get("chunk_id") or metadata.get("chunk_id")
            if chunk_id:
                chunk_uuid = _stable_uuid(group_id, document_name, chunk_id)
            else:
                chunk_uuid = _stable_uuid(group_id, document_name, f"idx:{i}")

            chunks.append({
                "idx": i,
                "text": text,
                "header_path": header_path,
                "document_date": doc_date,
                "chunk_id": chunk_id,
                "chunk_uuid": chunk_uuid,
            })

    if not chunks:
        print(f"  No valid chunks")
        return {"file": filename, "chunks_processed": 0, "chunks_successful": 0, "total_facts": 0, "total_entities": 0, "total_relationships": 0}

    print(f"  {len(chunks)} chunks loaded")

    # ===== CHECK FOR RESUME =====
    resume_phase = 1  # Default: start from Phase 1
    extractions = None
    entity_lookup_global = None
    topic_lookup_global = None
    uuid_by_name = {}
    buffer = None
    embeddings_done = False
    document_date_str = None

    if checkpoint_mgr:
        resume_phase = checkpoint_mgr.get_resume_phase()

        if resume_phase > 1:
            # Load Phase 1 data
            phase1_data = checkpoint_mgr.load_phase1()
            if phase1_data:
                extractions = phase1_data["extractions"]
                document_date_str = phase1_data.get("document_date_str")
                # Note: document_uuid, document_name, chunks were saved
                # but we already loaded them from the file, so just use extractions
                print(f"  Resuming: Loaded Phase 1 ({len(extractions)} extractions)")
            else:
                print("  Warning: Could not load Phase 1 checkpoint, starting from scratch")
                resume_phase = 1

        if resume_phase > 2:
            # Load Phase 2 data
            phase2_data = checkpoint_mgr.load_phase2()
            if phase2_data:
                entity_lookup_global = phase2_data["entity_lookup"]
                topic_lookup_global = phase2_data["topic_lookup"]
                uuid_by_name = phase2_data["uuid_by_name"]
                print(f"  Resuming: Loaded Phase 2 ({len(entity_lookup_global)} entities, {len(topic_lookup_global)} topics)")
            else:
                print("  Warning: Could not load Phase 2 checkpoint, restarting from Phase 2")
                resume_phase = 2

        if resume_phase > 3:
            # Load Phase 3 data
            phase3_data = checkpoint_mgr.load_phase3()
            if phase3_data:
                buffer = phase3_data["buffer"]
                embeddings_done = phase3_data["embeddings_done"]
                print(f"  Resuming: Loaded Phase 3 (embeddings_done={embeddings_done})")
            else:
                print("  Warning: Could not load Phase 3 checkpoint, restarting from Phase 3")
                resume_phase = 3

    # ===== EXTRACT DOCUMENT DATE =====
    if document_date_str is None:
        print("  Extracting document date...")
        temporal_extractor = TemporalExtractor()
        first_chunks = [c["text"] for c in chunks[:6]]
        last_chunks = [c["text"] for c in chunks[-6:]] if len(chunks) > 6 else []
        document_date_str = temporal_extractor.extract_date(first_chunks, last_chunks, document_name)
        if not document_date_str:
            # Fall back to metadata date if available
            for c in chunks:
                if c["document_date"]:
                    document_date_str = c["document_date"].date().isoformat()
                    break
        print(f"    Document date: {document_date_str or 'Unknown'}")

    # ===== PHASE 1: PARALLEL EXTRACTION =====
    if resume_phase <= 1:
        print(f"  Phase 1: Extracting (concurrency={concurrency})...")
        t1 = time.time()

        extractor = ExtractorV2()
        sem = asyncio.Semaphore(concurrency)

        document_date_for_prompt = document_date_str or "Unknown"
        tasks = [
            extract_chunk(
                extractor,
                c["idx"],
                c["text"],
                c["header_path"],
                document_date_for_prompt,
                c["chunk_uuid"],
                sem,
            )
            for c in chunks
        ]
        extractions = await asyncio.gather(*tasks)

        phase1_time = time.time() - t1
        ok = sum(1 for e in extractions if e["success"])
        print(f"  Phase 1: {ok}/{len(chunks)} extracted ({phase1_time:.1f}s)")
        if checkpoint_mgr:
            checkpoint_mgr.save_phase1(extractions, document_uuid, document_name, document_date_str, chunks)
    else:
        phase1_time = 0.0

    # ===== PHASE 2a: COLLECT ALL ENTITIES =====
    if resume_phase <= 2:
        # Derive entities from fact subjects/objects (not from entity list)
        # The entity list was for CoT extraction help; relationships define what matters
        print("  Phase 2a: Collecting entities...")
        entities_by_name: Dict[str, List[Dict]] = {}
        for ext in extractions:
            if not ext["success"]:
                continue
            for fact in ext["extraction"].facts:
                # Collect subjects that are entities (not topics)
                if fact.subject_type.lower() != "topic":
                    entities_by_name.setdefault(fact.subject, []).append({
                        "chunk_idx": ext["chunk_idx"],
                        "entity": EnumeratedEntity(
                            name=fact.subject,
                            entity_type=fact.subject_type,
                            summary=fact.subject_summary
                        )
                    })
                # Collect objects that are entities (not topics)
                if fact.object_type.lower() != "topic":
                    entities_by_name.setdefault(fact.object, []).append({
                        "chunk_idx": ext["chunk_idx"],
                        "entity": EnumeratedEntity(
                            name=fact.object,
                            entity_type=fact.object_type,
                            summary=fact.object_summary
                        )
                    })
        print(f"    Found {len(entities_by_name)} unique entity names")

        # ===== PHASE 2b: HYBRID DEDUPLICATION (Embedding + LLM) =====
        t2 = time.time()
        dedup_manager = DeferredDeduplicationManager.get_instance()
        dedup_manager.reset()  # Clear any previous state

        if entities_by_name:
            print(f"  Phase 2b: Generating embeddings and registering entities...")

            # Generate embeddings for all entities
            entity_names = list(entities_by_name.keys())
            entity_texts = []
            for name in entity_names:
                first_entity = entities_by_name[name][0]["entity"]
                summary = first_entity.summary if hasattr(first_entity, 'summary') else ""
                entity_texts.append(f"{name}: {summary}" if summary else name)

            entity_embeddings = dedup_embeddings.embed_documents(entity_texts)

            # Register entities with dedup manager
            uuid_by_name = {}
            for i, name in enumerate(entity_names):
                first_entity = entities_by_name[name][0]["entity"]
                entity_uuid = str(uuid4())
                uuid_by_name[name] = entity_uuid

                dedup_manager.register_entity(
                    uuid=entity_uuid,
                    name=name,
                    node_type="Entity",
                    summary=first_entity.summary if hasattr(first_entity, 'summary') else "",
                    embedding=entity_embeddings[i],
                    group_id=group_id
                )

            print(f"  Phase 2c: Clustering and deduplicating (threshold={similarity_threshold}, concurrency={dedup_concurrency})...")
            dedup_stats = await dedup_manager.cluster_and_remap_async(
                similarity_threshold=similarity_threshold,
                concurrency=dedup_concurrency
            )
            print(f"    {len(entity_names)} entities → {dedup_stats['distinct_entities']} canonical ({dedup_stats['duplicates_merged']} merged)")
        else:
            uuid_by_name = {}
            dedup_stats = {"distinct_entities": 0, "duplicates_merged": 0}

        dedup_time = time.time() - t2

        # ===== PHASE 2d-e: RESOLVE ENTITIES AND TOPICS (Parallel) =====
        # Both run concurrently for maximum throughput
        t3 = time.time()

        # Collect topics from extractions (needed before parallel resolution)
        # Also collect definitions from extracted topic entities for better resolution
        topic_names_from_facts: set[str] = set()
        topic_definitions: Dict[str, str] = {}  # topic_name -> definition/summary
        for ext in extractions:
            if not ext["success"]:
                continue
            # Collect topic definitions from entities
            for entity in ext["extraction"].entities:
                if entity.entity_type.lower() == "topic" and entity.summary:
                    topic_definitions[entity.name] = entity.summary
            # Collect topic names from facts
            for fact in ext["extraction"].facts:
                if fact.subject_type.lower() == "topic":
                    topic_names_from_facts.add(fact.subject)
                if fact.object_type.lower() == "topic":
                    topic_names_from_facts.add(fact.object)
                for t in fact.topics:
                    topic_names_from_facts.add(t)

        # Get unique canonical entity UUIDs
        canonical_uuids = list(set(
            dedup_manager.get_remapped_uuid(uuid_by_name[name])
            for name in uuid_by_name
        ))

        print(f"  Phase 2d: Pre-computing embeddings for resolution...")

        # Pre-compute entity embeddings in batch (avoids rate limits during resolution)
        entity_embeddings_map: Dict[str, List[float]] = {}
        if canonical_uuids:
            entity_texts = []
            valid_uuids = []
            for canonical_uuid in canonical_uuids:
                canonical_entity = dedup_manager._pending_entities[canonical_uuid]
                text = f"{canonical_entity.name}: {canonical_entity.summary}"
                if text.strip():  # Skip empty texts
                    entity_texts.append(text)
                    valid_uuids.append(canonical_uuid)
            # Batch embed all entities
            if entity_texts:
                entity_embeddings_list = dedup_embeddings.embed_documents(entity_texts)
                for i, canonical_uuid in enumerate(valid_uuids):
                    entity_embeddings_map[canonical_uuid] = entity_embeddings_list[i]
            print(f"    Pre-computed {len(entity_embeddings_map)} entity embeddings")

        # Pre-compute topic embeddings in batch using enriched text (topic: definition)
        topic_embeddings_map: Dict[str, List[float]] = {}
        topic_enriched_texts: Dict[str, str] = {}  # topic_name -> enriched text for resolution
        topic_names_list = [t for t in topic_names_from_facts if t and t.strip()]
        if topic_names_list:
            # Build enriched texts for embedding (use definition if available)
            texts_to_embed = []
            for topic_name in topic_names_list:
                definition = topic_definitions.get(topic_name, "")
                if definition:
                    enriched = f"{topic_name}: {definition}"
                else:
                    enriched = topic_name
                topic_enriched_texts[topic_name] = enriched
                texts_to_embed.append(enriched)
            topic_embeddings_list = embeddings.embed_documents(texts_to_embed)
            for i, topic_name in enumerate(topic_names_list):
                topic_embeddings_map[topic_name] = topic_embeddings_list[i]
            print(f"    Pre-computed {len(topic_embeddings_map)} topic embeddings (with definitions)")

        print(f"  Phase 2e: Resolving {len(canonical_uuids)} entities + {len(topic_names_from_facts)} topics in parallel...")

        # Semaphore for concurrency control (shared across both)
        resolve_sem = asyncio.Semaphore(resolve_concurrency)

        # ----- Entity Resolution (parallel) -----
        async def resolve_all_entities() -> Dict[str, Any]:
            """Resolve all canonical entities against Neo4j."""
            if not canonical_uuids:
                return {}

            async def resolve_canonical(canonical_uuid: str):
                async with resolve_sem:
                    canonical_entity = dedup_manager._pending_entities[canonical_uuid]
                    chunk_uuid = str(uuid4())
                    # Pass pre-computed embedding to avoid API call
                    precomputed_emb = entity_embeddings_map.get(canonical_uuid)
                    resolution = await asyncio.to_thread(
                        entity_registry.resolve,
                        entity_name=canonical_entity.name,
                        entity_type="Entity",
                        entity_summary=canonical_entity.summary,
                        chunk_uuid=chunk_uuid,
                        precomputed_embedding=precomputed_emb
                    )
                    return canonical_uuid, resolution

            tasks = [resolve_canonical(c) for c in canonical_uuids]
            results = dict(await asyncio.gather(*tasks))

            # Build entity_lookup: original_name -> resolution
            lookup = {}
            for original_name, entity_uuid in uuid_by_name.items():
                canonical_uuid = dedup_manager.get_remapped_uuid(entity_uuid)
                lookup[original_name] = results[canonical_uuid]
            return lookup

        # ----- Topic Resolution (parallel) -----
        async def resolve_all_topics() -> Dict[str, TopicResolution]:
            """Resolve all topics against ontology, reusing existing UUIDs from Neo4j."""
            if not topic_names_from_facts:
                return {}

            # Step 1: Resolve all topic names to canonical labels (using definitions for better matching)
            async def resolve_topic_name(topic_name: str):
                async with resolve_sem:
                    try:
                        enriched_text = topic_enriched_texts.get(topic_name, topic_name)
                        resolution = await asyncio.to_thread(
                            topic_librarian.resolve,
                            text=topic_name,
                            enriched_text=enriched_text,
                            context="",  # Context not available in batch mode
                            top_k=10,
                            candidate_threshold=0.4
                        )
                        if resolution:
                            return topic_name, resolution["label"], resolution.get("definition", "")
                        else:
                            log(f"Topic '{topic_name}' not in ontology - will be skipped")
                            return topic_name, None, None
                    except Exception as e:
                        log(f"Warning: Failed to resolve topic '{topic_name}': {e}")
                        return topic_name, None, None

            tasks = [resolve_topic_name(t) for t in topic_names_from_facts]
            raw_results = await asyncio.gather(*tasks)

            # Step 2: Group by canonical label and collect definitions
            canonical_to_names: Dict[str, List[str]] = {}
            canonical_definitions: Dict[str, str] = {}
            for topic_name, canonical_label, definition in raw_results:
                if canonical_label is None:
                    continue
                if canonical_label not in canonical_to_names:
                    canonical_to_names[canonical_label] = []
                    canonical_definitions[canonical_label] = definition or ""
                canonical_to_names[canonical_label].append(topic_name)

            if not canonical_to_names:
                return {}

            # Step 3: Check Neo4j for existing TopicNodes with these canonical labels
            existing_topics = neo4j.query('''
                MATCH (t:TopicNode {group_id: $gid})
                WHERE t.name IN $labels
                RETURN t.name as name, t.uuid as uuid
            ''', {"gid": group_id, "labels": list(canonical_to_names.keys())})

            existing_uuid_map = {t['name']: t['uuid'] for t in existing_topics}
            log(f"Found {len(existing_uuid_map)} existing TopicNodes in graph")

            # Step 4: Create TopicResolution for each canonical label (reuse UUID if exists)
            canonical_resolutions: Dict[str, TopicResolution] = {}
            for canonical_label in canonical_to_names:
                if canonical_label in existing_uuid_map:
                    # Reuse existing UUID from graph
                    topic_uuid = existing_uuid_map[canonical_label]
                    is_new = False
                else:
                    # New topic - create UUID
                    topic_uuid = str(uuid4())
                    is_new = True

                canonical_resolutions[canonical_label] = TopicResolution(
                    uuid=topic_uuid,
                    canonical_label=canonical_label,
                    is_new=is_new,
                    definition=canonical_definitions[canonical_label]
                )

            # Step 5: Map each original topic name to its canonical resolution
            result: Dict[str, TopicResolution] = {}
            for canonical_label, topic_names in canonical_to_names.items():
                resolution = canonical_resolutions[canonical_label]
                for topic_name in topic_names:
                    result[topic_name] = resolution

            return result

        # Run BOTH resolutions concurrently
        entity_lookup_global, topic_lookup_global = await asyncio.gather(
            resolve_all_entities(),
            resolve_all_topics()
        )

        resolve_time = time.time() - t3
        # Count topics that didn't match ontology
        skipped_topics = len(topic_names_from_facts) - len(topic_lookup_global)
        print(f"    Resolved {len(entity_lookup_global)} entities + {len(topic_lookup_global)} topics ({skipped_topics} skipped - not in ontology) in {resolve_time:.1f}s")
        if checkpoint_mgr:
            checkpoint_mgr.save_phase2(entity_lookup_global, topic_lookup_global, {}, uuid_by_name)
    else:
        dedup_time = 0.0
        resolve_time = 0.0

    # Build case-insensitive lookup maps (lowercase key -> original resolution)
    # This fixes issues where "Inflation Expectations" vs "Inflation expectations" fail to match
    entity_lookup_lower: Dict[str, Any] = {k.lower(): v for k, v in entity_lookup_global.items()}
    topic_lookup_lower: Dict[str, TopicResolution] = {k.lower(): v for k, v in topic_lookup_global.items()}

    # ===== PHASE 3: ASSEMBLY (Bulk Write) =====
    # Phase 3 collection (if buffer is None)
    if buffer is None:
        print("  Phase 3: Collecting operations...")
        t4 = time.time()

        # Use the LLM-extracted document date
        # Initialize bulk write buffer
        buffer = BulkWriteBuffer(
            document_uuid=document_uuid,
            document_name=document_name,
            document_date=document_date_str,  # LLM/metadata date (or None)
            group_id=group_id
        )

        # Collect all operations into buffer
        results = []
        for i, ext in enumerate(extractions):
            chunk_idx = ext["chunk_idx"]
            extraction = ext["extraction"]
            chunk_text = ext["chunk_text"]
            header_path = ext["header_path"]

            if not ext["success"] or not extraction.facts:
                results.append({
                    "success": ext["success"],
                    "chunk_idx": chunk_idx,
                    "facts": 0, "entities": 0, "relationships": 0,
                    "error": ext.get("error")
                })
                continue

            try:
                episodic_uuid = ext.get("chunk_uuid") or str(uuid4())

                # Use global lookups directly with lowercase keys for case-insensitive matching
                # Collect into buffer (no Neo4j writes or embeddings yet)
                counts = collect_chunk(
                    extraction, entity_lookup_lower, topic_lookup_lower,
                    episodic_uuid, chunk_text, header_path,
                    buffer
                )
                results.append({"success": True, "chunk_idx": chunk_idx, **counts})

            except Exception as e:
                print(f"  [Chunk {chunk_idx}] ERROR: {e}")
                results.append({"success": False, "chunk_idx": chunk_idx, "error": str(e)})

            if (i + 1) % 10 == 0:
                log(f"    Collected {i+1}/{len(extractions)} chunks...")

        collect_time = time.time() - t4
        print(f"    Collected {len(buffer.episodic_nodes)} episodic, {len(buffer.entity_nodes)} entities, {len(buffer.fact_nodes)} facts, {len(buffer.relationships)} rels ({collect_time:.1f}s)")
        if checkpoint_mgr:
            checkpoint_mgr.save_phase3(buffer, embeddings_done=False)
    else:
        collect_time = 0.0
        # Build results from resumed buffer for summary stats
        results = [{"success": True, "chunk_idx": i, "facts": 0, "entities": 0, "relationships": 0} for i in range(len(chunks))]

    # Phase 3b embeddings (if not done)
    if not embeddings_done:
        print("  Phase 3b: Generating embeddings (parallel)...")
        t_embed = time.time()
        await batch_generate_embeddings(buffer, embeddings, dedup_embeddings, batch_size=128)
        embed_time = time.time() - t_embed
        print(f"    Generated embeddings in {embed_time:.1f}s")
        if checkpoint_mgr:
            checkpoint_mgr.save_phase3(buffer, embeddings_done=True)
    else:
        embed_time = 0.0

    # Phase 3c write (always runs if we got here)
    print("  Phase 3c: Writing to Neo4j...")
    # Warmup connection - Aura may have gone to sleep during embedding generation
    neo4j.warmup()
    t5 = time.time()
    write_counts = bulk_write_all(buffer, neo4j, embeddings, llm, batch_size)
    write_time = time.time() - t5
    print(f"    Wrote in {write_time:.1f}s")

    assembly_time = collect_time + embed_time + write_time

    # Summary
    successful = sum(1 for r in results if r["success"])
    facts = sum(r.get("facts", 0) for r in results if r["success"])
    entities = sum(r.get("entities", 0) for r in results if r["success"])
    rels = sum(r.get("relationships", 0) for r in results if r["success"])

    print(f"  Done: {successful}/{len(chunks)} chunks, {facts} facts, {entities} entities, {rels} rels")
    print(f"  Time: P1={phase1_time:.1f}s, P2={dedup_time + resolve_time:.1f}s (dedup={dedup_time:.1f}s, resolve={resolve_time:.1f}s), P3={assembly_time:.1f}s")

    return {
        "file": filename,
        "chunks_processed": len(chunks),
        "chunks_successful": successful,
        "total_facts": facts,
        "total_entities": entities,
        "total_relationships": rels,
        "phase1_time": phase1_time,
        "phase2_time": dedup_time + resolve_time,
        "phase3_time": assembly_time
    }


# =============================================================================
# CLI
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Knowledge Graph Ingestion Pipeline")
    parser.add_argument("--input", "-i", default=CHUNKS_DIR, help="Input JSONL file or directory")
    parser.add_argument("--limit", "-l", type=int, help="Limit chunks per file")
    parser.add_argument("--group-id", "-g", default="default", help="Group/tenant ID")
    parser.add_argument("--filter", "-f", help="Filter files by name substring")
    parser.add_argument("--concurrency", "-c", type=int, default=DEFAULT_CONCURRENCY, help="Max concurrent extractions")
    parser.add_argument("--similarity-threshold", "-s", type=float, default=0.70, help="Similarity threshold for entity deduplication (0.0-1.0)")
    parser.add_argument("--resolve-concurrency", "-r", type=int, default=50, help="Max concurrent entity resolutions")
    parser.add_argument("--dedup-concurrency", "-d", type=int, default=20, help="Max concurrent LLM deduplication calls")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint (fail if none exists)")
    parser.add_argument("--fresh", action="store_true", help="Force fresh start (ignore existing checkpoint)")
    parser.add_argument("--batch-size", "-b", type=int, default=250, help="Neo4j write batch size (default: 250)")
    args = parser.parse_args()

    # Validate arguments
    if args.fresh and args.resume:
        print("ERROR: --fresh and --resume cannot be used together")
        sys.exit(1)

    if args.batch_size <= 0:
        print("ERROR: --batch-size must be positive")
        sys.exit(1)

    print("=" * 60)
    print("KNOWLEDGE GRAPH PIPELINE")
    print("=" * 60)

    start = time.time()

    # Find files
    input_path = Path(args.input)
    files = []

    if input_path.is_file():
        files = [str(input_path)]
    elif input_path.is_dir():
        for f in sorted(input_path.glob("*.jsonl")):
            if not args.filter or args.filter.lower() in f.name.lower():
                files.append(str(f))
    else:
        print(f"ERROR: {args.input} not found")
        sys.exit(1)

    if not files:
        print(f"No JSONL files found")
        sys.exit(0)

    print(f"\n{len(files)} file(s) to process")
    if args.limit:
        print(f"Limiting to {args.limit} chunks/file")

    # Handle checkpoint flags
    checkpoint_managers = {}  # filepath -> CheckpointManager

    for filepath in files:
        existing = CheckpointManager.find_existing(filepath, args.group_id)

        if args.fresh and existing:
            print(f"  --fresh specified, removing checkpoint for {os.path.basename(filepath)}...")
            existing.delete()
            existing = None

        if args.resume:
            if not existing:
                print(f"ERROR: --resume specified but no checkpoint found for {os.path.basename(filepath)}")
                sys.exit(1)
            existing.print_status()
            checkpoint_managers[filepath] = existing
        elif existing:
            # Auto-resume
            print(f"\nFound existing checkpoint:")
            existing.print_status()
            checkpoint_managers[filepath] = existing
        else:
            # Fresh run - create new checkpoint manager
            checkpoint_managers[filepath] = CheckpointManager(
                filepath, args.group_id,
                cli_args={"limit": args.limit, "concurrency": args.concurrency}
            )

    # Init services
    print("\nInitializing services...")
    from src.util.services import get_services
    svc = get_services()

    # Ensure vector indexes exist (auto-create if missing)
    print("  Checking Neo4j vector indexes...")
    _ensure_vector_indexes(svc.neo4j)
    # EntityRegistry uses dedup_embeddings (voyage-3-large) and claude_llm
    # to match the dedup system in Phase 2b
    entity_registry = EntityRegistry(
        neo4j_client=svc.neo4j,
        embeddings=svc.dedup_embeddings,
        llm=svc.claude_llm,
        group_id=args.group_id
    )
    topic_librarian = TopicLibrarian()
    print("  Ready")

    # Process files
    results = []
    for f in files:
        r = await process_file(
            f, args.group_id, args.limit, args.concurrency, args.similarity_threshold,
            args.resolve_concurrency, args.dedup_concurrency,
            svc.neo4j, svc.embeddings, svc.dedup_embeddings, svc.llm, entity_registry, topic_librarian,
            batch_size=args.batch_size,
            checkpoint_mgr=checkpoint_managers.get(f)
        )
        results.append(r)
        # Cleanup checkpoint on success
        if checkpoint_managers.get(f):
            checkpoint_managers[f].cleanup()
            print(f"  Checkpoint cleaned up")

    # Summary
    elapsed = time.time() - start
    total_chunks = sum(r["chunks_processed"] for r in results)
    total_ok = sum(r["chunks_successful"] for r in results)
    total_facts = sum(r["total_facts"] for r in results)
    total_entities = sum(r["total_entities"] for r in results)
    total_rels = sum(r["total_relationships"] for r in results)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Files: {len(results)}")
    print(f"Chunks: {total_ok}/{total_chunks} successful")
    print(f"Created: {total_facts} facts, {total_entities} entities, {total_rels} relationships")
    print(f"Time: {elapsed:.1f}s ({elapsed/max(total_chunks,1):.2f}s/chunk)")


if __name__ == "__main__":
    asyncio.run(main())
