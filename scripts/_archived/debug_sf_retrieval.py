"""Debug why San Francisco/Twelfth District is being missed in decline query."""

import asyncio
import os
import sys

sys.path.insert(0, "/home/rithv/Programming/Startups/ZommaLabsKG")
os.chdir("/home/rithv/Programming/Startups/ZommaLabsKG")

from src.querying_system.v5.graph_store import GraphStore


async def main():
    store = GraphStore(group_id="default")

    question = "Which districts reported a slight softening or decline in economic activity?"

    # Get embedding for the question
    from src.util.services import get_services
    services = get_services()
    embedding = await services.embeddings.aembed_query(question)

    print("=" * 60)
    print("GLOBAL VECTOR SEARCH")
    print("=" * 60)

    # Do a global vector search
    facts = await store.search_all_facts_vector(embedding, top_k=50)

    # Look for San Francisco / Twelfth District mentions
    sf_facts = []
    for f in facts:
        content_lower = f.content.lower() if f.content else ""
        subject_lower = f.subject.lower() if f.subject else ""
        object_lower = f.object.lower() if f.object else ""

        if any(term in content_lower or term in subject_lower or term in object_lower
               for term in ["san francisco", "twelfth", "12th"]):
            sf_facts.append(f)

    print(f"\nTotal facts retrieved: {len(facts)}")
    print(f"San Francisco/Twelfth District facts: {len(sf_facts)}")

    if sf_facts:
        print("\n--- SAN FRANCISCO FACTS ---")
        for f in sf_facts:
            print(f"\nSubject: {f.subject}")
            print(f"Edge: {f.edge_type}")
            print(f"Object: {f.object}")
            print(f"Content: {f.content}")
            print(f"Vector score: {f.vector_score:.3f}")
    else:
        print("\nNo SF facts in top 50. Let's check what we did get...")

    # Check for any "decline" or "softening" facts
    print("\n" + "=" * 60)
    print("FACTS MENTIONING DECLINE/SOFTENING/EDGED DOWN")
    print("=" * 60)

    decline_facts = []
    for f in facts:
        content_lower = f.content.lower() if f.content else ""
        if any(term in content_lower for term in ["decline", "softening", "edged down", "fell", "down"]):
            decline_facts.append(f)

    print(f"\nFacts mentioning decline-related terms: {len(decline_facts)}")
    for f in decline_facts[:10]:
        print(f"\n- {f.subject} -> {f.object}")
        print(f"  Content: {f.content[:200]}...")
        print(f"  Score: {f.vector_score:.3f}")

    # Now let's specifically search for "edged down"
    print("\n" + "=" * 60)
    print("KEYWORD SEARCH FOR 'edged down'")
    print("=" * 60)

    edged_facts = await store.search_all_facts_keyword(["edged", "down", "twelfth", "san francisco"], top_k=30)
    print(f"\nFacts found: {len(edged_facts)}")
    for f in edged_facts[:10]:
        print(f"\n- {f.subject} -> {f.object}")
        print(f"  Content: {f.content[:200]}...")

    # Check what's in the graph for Twelfth District
    print("\n" + "=" * 60)
    print("DIRECT ENTITY SEARCH: Twelfth District / San Francisco")
    print("=" * 60)

    # Try resolving "Twelfth District" and "San Francisco"
    from src.querying_system.v5.schemas import EntityHint

    hints = [
        EntityHint(name="Twelfth District", context="Federal Reserve district"),
        EntityHint(name="San Francisco", context="Federal Reserve district"),
        EntityHint(name="Federal Reserve Bank of San Francisco", context=""),
    ]

    resolved = await store.resolve_entities(hints, question)
    print(f"\nResolved entities: {len(resolved)}")
    for r in resolved:
        print(f"  {r.original_hint} -> {r.resolved_name} (conf: {r.confidence:.2f})")

        # Get facts for this entity
        entity_facts = await store.search_entity_facts(r.resolved_name, embedding)
        print(f"    Facts found: {len(entity_facts)}")
        for f in entity_facts[:3]:
            print(f"      - {f.content[:100]}...")


if __name__ == "__main__":
    asyncio.run(main())
