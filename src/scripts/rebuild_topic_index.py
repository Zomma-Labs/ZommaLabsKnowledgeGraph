#!/usr/bin/env python3
"""
Rebuilds the Qdrant topic index from custom topics only (no FIBO).

Usage:
    uv run python src/scripts/rebuild_topic_index.py
"""

import json
import uuid
import os
import sys
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from src.util.llm_client import get_embeddings

CUSTOM_TOPICS_DIR = "src/config/topics"
QDRANT_PATH = "./qdrant_topics"
COLLECTION_NAME = "topic_ontology"
VECTOR_SIZE = 1024  # Voyage finance-2
BATCH_SIZE = 50


def load_custom_topics():
    """Loads custom topic definitions from JSON files."""
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

                    # Main topic with full context
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

                    # Label-only vector for exact matching
                    all_topics.append({
                        "uri": f"{topic['uri']}/label",
                        "label": topic["label"],
                        "definition": topic["definition"],
                        "synonyms": synonyms_str,
                        "search_text": topic["label"],
                        "source": "custom_label"
                    })

                    # Each synonym as separate vector
                    for synonym in synonyms:
                        all_topics.append({
                            "uri": f"{topic['uri']}/synonym/{synonym.replace(' ', '_').lower()}",
                            "label": topic["label"],
                            "definition": topic["definition"],
                            "synonyms": synonyms_str,
                            "search_text": synonym,
                            "source": "custom_synonym"
                        })
                        synonym_count += 1

                print(f"   Loaded {len(topics)} topics from {os.path.basename(filepath)}")

        except Exception as e:
            print(f"   Failed to load {filepath}: {e}")

    label_count = len([t for t in all_topics if t.get("source") == "custom_label"])
    print(f"   Expanded to {len(all_topics)} vectors ({label_count} labels + {synonym_count} synonyms)")
    return all_topics


def index_to_qdrant(topics):
    """Embeds and indexes topics to Qdrant."""
    print(f"\nGenerating embeddings & indexing ({len(topics)} vectors)...")

    embeddings = get_embeddings()
    client = QdrantClient(path=QDRANT_PATH)

    # Recreate collection
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
    for i in range(0, total, BATCH_SIZE):
        batch = topics[i:i + BATCH_SIZE]
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

            client.upsert(collection_name=COLLECTION_NAME, points=points)
            print(f"   Indexed batch {i // BATCH_SIZE + 1} ({min(i + BATCH_SIZE, total)}/{total})")

        except Exception as e:
            print(f"   Batch failed: {e}")

    print("\n✅ Topic index rebuilt successfully!")


if __name__ == "__main__":
    topics = load_custom_topics()
    if topics:
        index_to_qdrant(topics)
    else:
        print("No topics found to index.")
