"""
MODULE: Topic Ontology Loader (FIBO + Custom)
DESCRIPTION:
    Loads the financial topic ontology from:
    1. FIBO's IND (Indices and Indicators) module - economic indicators, interest rates, etc.
    2. Custom topic definitions for concepts not in FIBO (AI, digital transformation, etc.)

    Indexes to Qdrant for topic resolution during entity extraction.

USAGE:
    python src/tools/topic_loader.py
"""

import rdflib
import json
import uuid
import os
import sys
import glob
from typing import List, Dict, Set
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from src.util.llm_client import get_embeddings

# --- Configuration ---
FIBO_DIR = "./data/fibo_source/fibo-master"
CUSTOM_TOPICS_DIR = "src/config/topics"
QDRANT_PATH = "./qdrant_topics"
COLLECTION_NAME = "topic_ontology"

# Limits
VECTOR_SIZE = 1024  # Voyage finance-2
BATCH_SIZE = 50

# FIBO modules to extract topics from
TOPIC_MODULES = [
    "/IND/",           # Indices and Indicators (economic indicators, interest rates)
    "/FBC/",           # Financial Business and Commerce
]

# Patterns to skip (too granular or meta)
SKIP_PATTERNS = [
    "/FND/Utilities/",
    "/FND/Relations/",
    "/MetadataIND",
    "/MetadataFBC",
    "Publisher",        # Skip publisher entities
    "Provider",         # Skip provider entities
    "Authority",        # Skip authority entities
]


def parse_fibo_topics() -> List[Dict]:
    """Parses FIBO for topic-relevant concepts from IND and FBC modules."""
    print(f"Parsing FIBO topic modules from {FIBO_DIR}...")

    if not os.path.exists(FIBO_DIR):
        print("   FIBO directory not found. Run fibo_loader.py first to download.")
        return []

    g = rdflib.Graph()

    # Find all RDF files in topic-relevant modules
    all_files = glob.glob(f"{FIBO_DIR}/**/*", recursive=True)
    files = []

    for f in all_files:
        if not (f.endswith('.ttl') or f.endswith('.rdf')):
            continue
        # Only include IND and relevant FBC modules
        if any(mod in f for mod in TOPIC_MODULES):
            # Skip metadata and noise
            if not any(skip in f for skip in SKIP_PATTERNS):
                files.append(f)

    print(f"   Found {len(files)} relevant ontology files")

    for file_path in tqdm(files, desc="Parsing FIBO files"):
        try:
            fmt = "turtle" if file_path.endswith('.ttl') else "xml"
            g.parse(file_path, format=fmt)
        except Exception:
            pass

    print(f"   Loaded {len(g)} triples")

    # Query for concepts with labels and definitions
    query = """
        SELECT DISTINCT ?entity ?label ?definition
        WHERE {
            ?entity a owl:Class .
            { ?entity rdfs:label ?label } UNION { ?entity skos:prefLabel ?label } .
            OPTIONAL { ?entity skos:definition ?definition } .
            FILTER NOT EXISTS { ?entity owl:deprecated "true"^^xsd:boolean }
        }
    """

    concepts_map = {}
    seen_labels: Set[str] = set()

    for row in g.query(query):
        uri = str(row["entity"])

        # Skip if matches skip patterns
        if any(skip in uri for skip in SKIP_PATTERNS):
            continue

        # Only keep IND and relevant concepts
        if not any(mod in uri for mod in TOPIC_MODULES):
            continue

        label = str(row["label"])
        definition = str(row["definition"]) if row["definition"] else ""

        # Skip if we've already seen this label (dedup)
        if label.lower() in seen_labels:
            continue
        seen_labels.add(label.lower())

        # Skip very short or numeric labels
        if len(label) < 3 or label.replace(" ", "").isdigit():
            continue

        # Skip duration/time period concepts (too granular)
        if any(x in label.lower() for x in ["months", "years", "days", "weeks", "overnight"]):
            continue

        search_text = f"Topic: {label}\nDefinition: {definition}"

        concepts_map[uri] = {
            "uri": uri,
            "label": label,
            "definition": definition,
            "search_text": search_text,
            "source": "fibo"
        }

    concepts = list(concepts_map.values())
    print(f"   Extracted {len(concepts)} FIBO topic concepts")
    return concepts


def load_custom_topics() -> List[Dict]:
    """Loads custom topic definitions from JSON files.

    Each synonym is indexed as a SEPARATE vector pointing to the parent topic.
    This ensures that searching for "Employees" directly matches "Labor Market".
    """
    print(f"Loading custom topics from {CUSTOM_TOPICS_DIR}...")

    topic_files = glob.glob(f"{CUSTOM_TOPICS_DIR}/*.json")

    if not topic_files:
        print("   No custom topic files found")
        return []

    all_topics = []
    synonym_count = 0

    for filepath in topic_files:
        try:
            with open(filepath, 'r') as f:
                topics = json.load(f)

                for topic in topics:
                    synonyms = topic.get("synonyms", [])
                    synonyms_str = ", ".join(synonyms)

                    # Index the main topic label in full format (with synonyms context)
                    search_text = f"Topic: {topic['label']}\nDefinition: {topic['definition']}"
                    if synonyms_str:
                        search_text += f"\nSynonyms: {synonyms_str}"

                    all_topics.append({
                        "uri": topic["uri"],
                        "label": topic["label"],
                        "definition": topic["definition"],
                        "synonyms": synonyms_str,
                        "search_text": search_text,
                        "source": "custom"
                    })

                    # ALSO index the main label as a standalone vector for exact matching
                    # Embed ONLY the label (no definition) for highest similarity
                    # Definition is stored in payload for LLM verification
                    all_topics.append({
                        "uri": f"{topic['uri']}/label",
                        "label": topic["label"],
                        "definition": topic["definition"],
                        "synonyms": synonyms_str,
                        "search_text": topic["label"],  # Just the label!
                        "source": "custom_label"
                    })

                    # Index each synonym as a SEPARATE vector pointing to the same topic
                    # Embed ONLY the synonym term for highest similarity
                    # Definition stored in payload for LLM verification
                    for synonym in synonyms:
                        all_topics.append({
                            "uri": f"{topic['uri']}/synonym/{synonym.replace(' ', '_').lower()}",
                            "label": topic["label"],  # Resolves to the PARENT label
                            "definition": topic["definition"],
                            "synonyms": synonyms_str,
                            "search_text": synonym,  # Just the synonym term!
                            "source": "custom_synonym"
                        })
                        synonym_count += 1

                print(f"   Loaded {len(topics)} topics from {os.path.basename(filepath)}")

        except Exception as e:
            print(f"   Failed to load {filepath}: {e}")

    label_count = len([t for t in all_topics if t.get("source") == "custom_label"])
    print(f"   Expanded to {len(all_topics)} vectors ({label_count} label + {synonym_count} synonym entries)")
    return all_topics


def deduplicate_topics(fibo_topics: List[Dict], custom_topics: List[Dict]) -> List[Dict]:
    """Deduplicates topics, preferring custom definitions over FIBO."""
    print("Deduplicating topics...")

    # Custom topics take precedence
    seen_labels = {t["label"].lower() for t in custom_topics}

    # Add FIBO topics that don't conflict
    deduped = list(custom_topics)
    for topic in fibo_topics:
        if topic["label"].lower() not in seen_labels:
            deduped.append(topic)
            seen_labels.add(topic["label"].lower())

    print(f"   Final: {len(deduped)} unique topics")
    return deduped


def batch_process(data, batch_size):
    """Yields successive n-sized chunks from data."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def index_to_qdrant(topics: List[Dict]):
    """Embeds topic text and pushes to Vector DB."""
    print(f"Generating Embeddings & Indexing ({len(topics)} topics)...")

    embeddings = get_embeddings()
    client = QdrantClient(path=QDRANT_PATH)

    # Recreate collection (fresh start)
    if client.collection_exists(COLLECTION_NAME):
        print(f"   Deleting existing collection '{COLLECTION_NAME}'...")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"   Created collection '{COLLECTION_NAME}'")

    # Process in batches
    total = len(topics)
    for i, batch in enumerate(batch_process(topics, BATCH_SIZE)):
        texts = [t["search_text"] for t in batch]

        try:
            vectors = embeddings.embed_documents(texts)

            points = []
            for j, topic in enumerate(batch):
                points.append(PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, topic["uri"])),
                    vector=vectors[j],
                    payload={
                        "uri": topic["uri"],
                        "label": topic["label"],
                        "definition": topic["definition"],
                        "synonyms": topic.get("synonyms", ""),
                        "source": topic.get("source", "unknown")
                    }
                ))

            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            print(f"   Indexed batch {i+1} ({min((i+1)*BATCH_SIZE, total)}/{total})")

        except Exception as e:
            print(f"   Batch failed: {e}")

    print("Indexing Complete.")


def get_topic_stats():
    """Returns stats about indexed topics."""
    client = QdrantClient(path=QDRANT_PATH)

    if not client.collection_exists(COLLECTION_NAME):
        return {"total": 0, "fibo": 0, "custom": 0, "custom_label": 0, "custom_synonym": 0}

    # Count by source
    fibo_count = 0
    custom_count = 0
    label_count = 0
    synonym_count = 0

    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            with_payload=True,
            with_vectors=False,
            offset=offset
        )

        for point in points:
            source = point.payload.get("source", "unknown")
            if source == "fibo":
                fibo_count += 1
            elif source == "custom":
                custom_count += 1
            elif source == "custom_label":
                label_count += 1
            elif source == "custom_synonym":
                synonym_count += 1

        if offset is None:
            break

    return {
        "total": fibo_count + custom_count + label_count + synonym_count,
        "fibo": fibo_count,
        "custom": custom_count,
        "custom_label": label_count,
        "custom_synonym": synonym_count
    }


if __name__ == "__main__":
    fibo_topics = parse_fibo_topics()
    custom_topics = load_custom_topics()

    all_topics = deduplicate_topics(fibo_topics, custom_topics)

    if all_topics:
        index_to_qdrant(all_topics)

        stats = get_topic_stats()
        print(f"\nTopic Ontology Stats:")
        print(f"   FIBO topics: {stats['fibo']}")
        print(f"   Custom topics: {stats['custom']}")
        print(f"   Custom labels: {stats['custom_label']}")
        print(f"   Custom synonyms: {stats['custom_synonym']}")
        print(f"   Total vectors: {stats['total']}")
    else:
        print("No topics found to index.")
