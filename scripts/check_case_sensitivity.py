#!/usr/bin/env python3
"""Check for case sensitivity issues in topic resolution."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.util.services import get_services

neo4j = get_services().neo4j

print("=" * 60)
print("CHECKING CASE SENSITIVITY ISSUES")
print("=" * 60)

# Get all TopicNode names
topics = neo4j.query('''
    MATCH (t:TopicNode {group_id: "default"})
    RETURN t.name as name
''')
topic_names = {t["name"] for t in topics}
topic_names_lower = {t["name"].lower(): t["name"] for t in topics}

print(f"Total TopicNodes: {len(topic_names)}")

# Get all EntityNode names
entities = neo4j.query('''
    MATCH (e:EntityNode {group_id: "default"})
    RETURN e.name as name
''')
entity_names = {e["name"] for e in entities}
entity_names_lower = {e["name"].lower(): e["name"] for e in entities}

print(f"Total EntityNodes: {len(entity_names)}")

# Check for near-duplicate names (same when lowercased)
print("\n" + "=" * 60)
print("CHECKING FOR CASE VARIANTS")
print("=" * 60)

# Topics with case variants
topic_lower_counts = {}
for t in topic_names:
    lower = t.lower()
    topic_lower_counts[lower] = topic_lower_counts.get(lower, []) + [t]

case_variants = {k: v for k, v in topic_lower_counts.items() if len(v) > 1}
print(f"\nTopics with case variants: {len(case_variants)}")
for lower, variants in list(case_variants.items())[:10]:
    print(f"  {variants}")

# Same for entities
entity_lower_counts = {}
for e in entity_names:
    lower = e.lower()
    entity_lower_counts[lower] = entity_lower_counts.get(lower, []) + [e]

entity_case_variants = {k: v for k, v in entity_lower_counts.items() if len(v) > 1}
print(f"\nEntities with case variants: {len(entity_case_variants)}")
for lower, variants in list(entity_case_variants.items())[:10]:
    print(f"  {variants}")

# Check if orphaned edges have case mismatches
print("\n" + "=" * 60)
print("ANALYZING ORPHANED EDGE PATTERNS")
print("=" * 60)

# Get orphaned subject edges and their relationship names
# The rel_type often contains hints about the expected object
orphaned = neo4j.query('''
    MATCH (subj)-[r1]->(c:EpisodicNode {group_id: "default"})
    WHERE r1.fact_id IS NOT NULL
      AND (subj:EntityNode OR subj:TopicNode)
      AND NOT EXISTS {
        MATCH (c)-[r2]->(obj)
        WHERE r2.fact_id = r1.fact_id
          AND (obj:EntityNode OR obj:TopicNode)
      }
    MATCH (f:FactNode {uuid: r1.fact_id})
    RETURN type(r1) as rel_type, f.content as fact_content
    LIMIT 30
''')

print(f"\nSample relationship types from orphaned edges:")
rel_types = {}
for o in orphaned:
    rt = o["rel_type"]
    rel_types[rt] = rel_types.get(rt, 0) + 1

for rt, cnt in sorted(rel_types.items(), key=lambda x: -x[1])[:15]:
    print(f"  {rt}: {cnt}")

# Try to extract likely object names from fact content
print("\n" + "=" * 60)
print("CHECKING FACT CONTENT FOR MISSING OBJECTS")
print("=" * 60)

print("\nAnalyzing fact content to identify likely missing objects:")
for o in orphaned[:5]:
    fact = o["fact_content"]
    rel = o["rel_type"]
    print(f"\n  Rel: {rel}")
    print(f"  Fact: {fact[:120]}...")

    # Look for common patterns that might indicate the object
    # E.g., "X is about Y" -> Y is the object
    words = fact.split()
    if "about" in words:
        idx = words.index("about")
        potential_object = " ".join(words[idx+1:idx+4])
        print(f"  Potential object after 'about': {potential_object}")
    elif "across" in words:
        idx = words.index("across")
        potential_object = " ".join(words[idx+1:idx+5])
        print(f"  Potential object after 'across': {potential_object}")
