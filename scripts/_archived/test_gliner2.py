"""Test GLiNER2 on tricky Beige Book chunks - with relation extraction."""

from gliner import GLiNER

# Tricky chunks from the Beige Book
TRICKY_CHUNKS = [
    {
        "id": "chunk_0037",
        "description": "Many specific businesses with tariff/trade relationships",
        "text": """The pace of price increases was mostly unchanged; selling prices continued to rise moderately while input prices again rose strongly. Many businesses continued to strategize around how to structure prices in consideration of new tariffs. One major retailer reported adjusting price increases based on the demand elasticity for each item, and many reported working with vendors to find alternative inputs to mitigate the impact of tariffs on their costs. A Long Island area importer and wholesaler of auto parts noted that tariffs on goods imported from India presented particularly steep challenges to their business, while a coffee roaster and supplier noted that tariffs on imports from Brazil threatened their ability to remain profitable. A specialty appliance manufacturer from upstate New York hiked their prices for the second time this year to account for the impact of tariffs on their costs. An upstate brewing company reported that elevated ingredient and packaging material costs were getting passed along to the consumer. A pharmaceutical company reported that foreign manufacturers were absorbing tariff costs to maintain market share and production levels. Some restaurants noted escalating costs for food—particularly beef—and imported wine were hurting their businesses. A New York City-based specialty contractor reported that congestion pricing was pushing up the cost of doing business considerably. Looking ahead, firms expect significant pricing pressures to persist."""
    },
    {
        "id": "chunk_0031",
        "description": "Abstract socioeconomic concepts, policy programs",
        "text": """Contacts expressed increased anxiety and uncertainty regarding the economic situation of low- and moderate-income households in the First District. The prevalence of economic precarity increased in many communities, as evidenced by greater food pantry use and increased difficulty paying rent, utilities, and other bills. Consumers increasingly used savings to cover basic needs and took steps to reduce spending where possible. Planned cuts to SNAP and Medicaid, as well as further increases in consumer prices linked to tariffs, posed additional risks to low- and moderate-income households going forward. New England continued to face workforce housing shortages, and rising construction costs pushed against further development of such housing."""
    },
    {
        "id": "chunk_0018",
        "description": "Agricultural trade concerns, China soybean ban",
        "text": """District economic activity contracted slightly. Labor demand softened, according to firms and job seekers, though wage growth remained moderate. Price increases remained modest, but input price pressures increased. Manufacturing and commercial real estate were flat, but most other sectors contracted. Agricultural contacts were concerned about China's elimination of soybean purchases."""
    }
]

# Entity types to extract
ENTITY_TYPES = [
    "Organization",
    "Location",
    "Product",
    "Policy",
    "Economic Indicator",
    "Industry",
]

# Relation types for the relex model
RELATION_TYPES = [
    "imports from",
    "exports to",
    "affected by",
    "located in",
    "produces",
    "increases price of",
    "decreases price of",
    "eliminated",
    "banned",
    "imposed on",
]

print("=" * 80)
print("GLINER2 RELATION EXTRACTION TEST ON BEIGE BOOK CHUNKS")
print("=" * 80)

# Load the relation extraction model (it does both NER and relations)
print("\nLoading GLiNER relex model...")
relex_model = GLiNER.from_pretrained("knowledgator/gliner-relex-large-v0.5")
print("Relation extraction model loaded!")

for chunk in TRICKY_CHUNKS:
    print(f"\n{'='*80}")
    print(f"CHUNK: {chunk['id']}")
    print(f"DESCRIPTION: {chunk['description']}")
    print(f"{'='*80}")
    print(f"\nTEXT:\n{chunk['text'][:300]}...")

    # Extract entities AND relations together using inference
    print(f"\n--- JOINT ENTITY + RELATION EXTRACTION ---")
    try:
        # Use the inference method which takes both labels and relations
        entities, relations = relex_model.inference(
            texts=chunk['text'],
            labels=ENTITY_TYPES,
            relations=RELATION_TYPES,
            threshold=0.4,
            relation_threshold=0.3,
            return_relations=True,
        )

        # Display entities
        print(f"\n  ENTITIES:")
        if entities and entities[0]:
            by_label = {}
            for ent in entities[0]:
                label = ent['label']
                if label not in by_label:
                    by_label[label] = []
                by_label[label].append((ent['text'], round(ent['score'], 3)))

            for label, items in sorted(by_label.items()):
                print(f"\n    {label}:")
                for text, score in sorted(items, key=lambda x: -x[1])[:5]:
                    print(f"      - {text} ({score})")
        else:
            print("    (No entities found)")

        # Display relations
        print(f"\n  RELATIONS:")
        if relations and relations[0]:
            seen = set()
            for rel in sorted(relations[0], key=lambda x: -x['score']):
                source = rel.get('source', rel.get('head', '?'))
                target = rel.get('target', rel.get('tail', '?'))
                relation = rel.get('relation', rel.get('label', '?'))
                score = rel.get('score', 0)

                # Handle different formats
                if isinstance(source, dict):
                    source = source.get('text', str(source))
                if isinstance(target, dict):
                    target = target.get('text', str(target))

                key = (str(source), relation, str(target))
                if key not in seen:
                    seen.add(key)
                    print(f"    {source} --[{relation}]--> {target} (score: {score:.3f})")
        else:
            print("    (No relations found)")

    except Exception as e:
        import traceback
        print(f"  Error: {e}")
        traceback.print_exc()

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
