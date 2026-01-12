#!/usr/bin/env python3
"""
Diagnose Data Issues
====================

Checks for:
1. Truncated fact content (facts ending with "..." or cut off mid-sentence)
2. Relationships without proper evidence (orphaned edges)

Usage:
    uv run scripts/diagnose_data_issues.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.util.services import get_services


def check_truncated_facts(neo4j, group_id: str = "default"):
    """Find facts that appear to be truncated."""
    print("\n" + "=" * 60)
    print("CHECKING FOR TRUNCATED FACTS")
    print("=" * 60)

    # Get all facts
    results = neo4j.query("""
        MATCH (f:FactNode {group_id: $uid})
        RETURN f.uuid AS uuid, f.content AS content
    """, {"uid": group_id})

    print(f"Total facts: {len(results)}")

    truncated = []
    short_facts = []
    suspicious_endings = []

    for row in results:
        content = row.get("content", "") or ""
        uuid = row.get("uuid", "")

        # Check for explicit truncation markers
        if content.endswith("..."):
            truncated.append({"uuid": uuid, "content": content, "reason": "ends with ..."})
        elif content.endswith("…"):
            truncated.append({"uuid": uuid, "content": content, "reason": "ends with …"})

        # Check for suspiciously short facts (might be cut off)
        if len(content) < 20 and content:
            short_facts.append({"uuid": uuid, "content": content, "length": len(content)})

        # Check for facts that end mid-word or with incomplete patterns
        if content and not content.endswith(('.', '!', '?', '"', "'", ')', ']')):
            last_word = content.split()[-1] if content.split() else ""
            # Check if ends with common incomplete patterns
            if any(content.lower().endswith(p) for p in [' the', ' a', ' an', ' to', ' of', ' in', ' by', ' for', ' with', ' and', ' or']):
                suspicious_endings.append({"uuid": uuid, "content": content, "ends_with": last_word})

    print(f"\nFacts ending with '...' or '…': {len(truncated)}")
    for t in truncated[:10]:
        print(f"  - [{t['uuid'][:8]}...] {t['content'][:100]}...")
    if len(truncated) > 10:
        print(f"  ... and {len(truncated) - 10} more")

    print(f"\nSuspiciously short facts (<20 chars): {len(short_facts)}")
    for s in short_facts[:10]:
        print(f"  - [{s['uuid'][:8]}...] '{s['content']}' (len={s['length']})")
    if len(short_facts) > 10:
        print(f"  ... and {len(short_facts) - 10} more")

    print(f"\nFacts with suspicious endings: {len(suspicious_endings)}")
    for s in suspicious_endings[:10]:
        print(f"  - [{s['uuid'][:8]}...] ...{s['content'][-50:]} (ends: '{s['ends_with']}')")
    if len(suspicious_endings) > 10:
        print(f"  ... and {len(suspicious_endings) - 10} more")

    return {
        "total": len(results),
        "truncated": len(truncated),
        "short": len(short_facts),
        "suspicious": len(suspicious_endings)
    }


def check_orphaned_relationships(neo4j, group_id: str = "default"):
    """Find relationships that don't properly connect to chunks."""
    print("\n" + "=" * 60)
    print("CHECKING FOR ORPHANED RELATIONSHIPS")
    print("=" * 60)

    # Check 1: Relationships with fact_id that don't have corresponding chunks
    print("\n1. Checking for edges with fact_id but no chunk connection...")

    # Get all relationships with fact_id property
    rels_with_fact_id = neo4j.query("""
        MATCH (a)-[r]->(b)
        WHERE r.fact_id IS NOT NULL
          AND a.group_id = $uid
          AND NOT (a:EpisodicNode OR b:EpisodicNode)
        RETURN type(r) AS rel_type,
               r.fact_id AS fact_id,
               a.name AS from_name,
               b.name AS to_name,
               labels(a) AS from_labels,
               labels(b) AS to_labels
        LIMIT 100
    """, {"uid": group_id})

    print(f"   Edges with fact_id NOT connected to EpisodicNode: {len(rels_with_fact_id)}")
    for r in rels_with_fact_id[:5]:
        print(f"     - {r['from_name']} -[{r['rel_type']}]-> {r['to_name']} (fact_id: {r['fact_id'][:8]}...)")

    # Check 2: Get expected pattern (Entity -> Episodic -> Entity with matching fact_id)
    print("\n2. Checking Subject -> Chunk -> Object pattern...")

    # Find fact_ids that SHOULD have connections
    all_fact_ids = neo4j.query("""
        MATCH (f:FactNode {group_id: $uid})
        RETURN f.uuid AS fact_id
    """, {"uid": group_id})

    fact_id_set = {r["fact_id"] for r in all_fact_ids}
    print(f"   Total FactNodes: {len(fact_id_set)}")

    # Check how many have proper Subject->Chunk->Object pattern
    proper_pattern = neo4j.query("""
        MATCH (subj)-[r1]->(c:EpisodicNode {group_id: $uid})-[r2]->(obj)
        WHERE (subj:EntityNode OR subj:TopicNode)
          AND (obj:EntityNode OR obj:TopicNode)
          AND subj.group_id = $uid
          AND obj.group_id = $uid
          AND r1.fact_id IS NOT NULL
          AND r1.fact_id = r2.fact_id
        RETURN DISTINCT r1.fact_id AS fact_id
    """, {"uid": group_id})

    facts_with_pattern = {r["fact_id"] for r in proper_pattern}
    print(f"   Facts with proper Subject->Chunk->Object pattern: {len(facts_with_pattern)}")

    orphaned_facts = fact_id_set - facts_with_pattern
    print(f"   Orphaned facts (no pattern): {len(orphaned_facts)}")

    # Check 3: Get sample of orphaned facts
    if orphaned_facts:
        sample = list(orphaned_facts)[:5]
        print("\n   Sample orphaned facts:")
        for fid in sample:
            fact_info = neo4j.query("""
                MATCH (f:FactNode {uuid: $fid, group_id: $uid})
                RETURN f.content AS content
            """, {"fid": fid, "uid": group_id})
            if fact_info:
                content = fact_info[0]["content"][:80]
                print(f"     - [{fid[:8]}...] {content}...")

    # Check 4: Verify CONTAINS_FACT relationships exist
    print("\n3. Checking CONTAINS_FACT relationships...")
    contains_fact_count = neo4j.query("""
        MATCH (c:EpisodicNode {group_id: $uid})-[:CONTAINS_FACT]->(f:FactNode)
        RETURN count(*) AS cnt
    """, {"uid": group_id})
    print(f"   EpisodicNode -[:CONTAINS_FACT]-> FactNode: {contains_fact_count[0]['cnt']}")

    # Check 5: Sample some relationships and see if chunks exist
    print("\n4. Sampling edges to verify chunk retrievability...")
    sample_edges = neo4j.query("""
        MATCH (a)-[r]->(c:EpisodicNode {group_id: $uid})-[r2]->(b)
        WHERE r.fact_id IS NOT NULL
          AND r.fact_id = r2.fact_id
          AND (a:EntityNode OR a:TopicNode)
          AND (b:EntityNode OR b:TopicNode)
        RETURN a.name AS subject, type(r) AS edge_type, b.name AS object,
               c.uuid AS chunk_id, c.content AS chunk_content
        LIMIT 10
    """, {"uid": group_id})

    print(f"   Sampled {len(sample_edges)} edges with valid chunk connections:")
    for e in sample_edges[:3]:
        chunk_preview = (e["chunk_content"] or "")[:60]
        print(f"     - {e['subject']} -[{e['edge_type']}]-> {e['object']}")
        print(f"       Chunk [{e['chunk_id'][:8]}...]: {chunk_preview}...")

    # Check for edges that explore_neighbors would find but get_chunk would miss
    print("\n5. Testing get_chunk query pattern...")
    # This mimics what _get_chunk_logic does in mcp_server.py
    test_sample = neo4j.query("""
        MATCH (a)-[r1]->(c:EpisodicNode {group_id: $uid})-[r2]->(b)
        WHERE r1.fact_id IS NOT NULL
          AND r1.fact_id = r2.fact_id
          AND (a:EntityNode OR a:TopicNode)
          AND (b:EntityNode OR b:TopicNode)
          AND a.group_id = $uid
          AND b.group_id = $uid
        RETURN a.name AS e1, b.name AS e2, type(r1) AS edge_type
        LIMIT 5
    """, {"uid": group_id})

    success = 0
    fail = 0
    for sample in test_sample:
        e1, e2, edge_type = sample["e1"], sample["e2"], sample["edge_type"]
        # Test the exact query pattern from mcp_server
        result = neo4j.query("""
            MATCH (e1:EntityNode {name: $e1, group_id: $uid})-[r1]->(c:EpisodicNode {group_id: $uid})-[r2]->(e2:EntityNode {name: $e2, group_id: $uid})
            WHERE type(r1) = $edge_type AND r1.fact_id = r2.fact_id
            RETURN c
            LIMIT 1
        """, {"e1": e1, "e2": e2, "edge_type": edge_type, "uid": group_id})

        if result:
            success += 1
        else:
            fail += 1
            print(f"     FAILED: {e1} -[{edge_type}]-> {e2}")

    print(f"   get_chunk pattern success: {success}/{len(test_sample)}")

    return {
        "total_facts": len(fact_id_set),
        "facts_with_pattern": len(facts_with_pattern),
        "orphaned": len(orphaned_facts)
    }


def check_chunk_content_completeness(neo4j, group_id: str = "default"):
    """Check if chunk content is properly stored."""
    print("\n" + "=" * 60)
    print("CHECKING CHUNK CONTENT COMPLETENESS")
    print("=" * 60)

    results = neo4j.query("""
        MATCH (c:EpisodicNode {group_id: $uid})
        RETURN c.uuid AS uuid, c.content AS content, c.header_path AS header
    """, {"uid": group_id})

    print(f"Total chunks: {len(results)}")

    empty_chunks = []
    short_chunks = []

    for row in results:
        content = row.get("content", "") or ""
        uuid = row.get("uuid", "")

        if not content.strip():
            empty_chunks.append(uuid)
        elif len(content) < 50:
            short_chunks.append({"uuid": uuid, "content": content, "length": len(content)})

    print(f"Empty chunks: {len(empty_chunks)}")
    print(f"Short chunks (<50 chars): {len(short_chunks)}")

    for s in short_chunks[:5]:
        print(f"  - [{s['uuid'][:8]}...] '{s['content'][:40]}' (len={s['length']})")

    return {
        "total": len(results),
        "empty": len(empty_chunks),
        "short": len(short_chunks)
    }


def main():
    print("Initializing Neo4j connection...")
    svc = get_services()
    neo4j = svc.neo4j
    group_id = "default"

    # Run diagnostics
    fact_stats = check_truncated_facts(neo4j, group_id)
    rel_stats = check_orphaned_relationships(neo4j, group_id)
    chunk_stats = check_chunk_content_completeness(neo4j, group_id)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\nFacts: {fact_stats['total']} total")
    print(f"  - Truncated (ends with ...): {fact_stats['truncated']}")
    print(f"  - Suspiciously short: {fact_stats['short']}")
    print(f"  - Suspicious endings: {fact_stats['suspicious']}")

    print(f"\nRelationships: {rel_stats['total_facts']} facts")
    print(f"  - With proper Subject->Chunk->Object: {rel_stats['facts_with_pattern']}")
    print(f"  - Orphaned (no chunk connection): {rel_stats['orphaned']}")

    print(f"\nChunks: {chunk_stats['total']} total")
    print(f"  - Empty: {chunk_stats['empty']}")
    print(f"  - Short (<50 chars): {chunk_stats['short']}")

    # Issues detected
    issues = []
    if fact_stats['truncated'] > 0:
        issues.append(f"{fact_stats['truncated']} truncated facts")
    if rel_stats['orphaned'] > 0:
        issues.append(f"{rel_stats['orphaned']} orphaned relationships")
    if chunk_stats['empty'] > 0:
        issues.append(f"{chunk_stats['empty']} empty chunks")

    if issues:
        print(f"\n⚠️  Issues detected: {', '.join(issues)}")
    else:
        print("\n✓ No major issues detected")


if __name__ == "__main__":
    main()
