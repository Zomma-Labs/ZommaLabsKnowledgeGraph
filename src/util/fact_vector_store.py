"""
Fact Vector Store - Qdrant collection for semantic fact search.

Uses OpenAI text-embedding-3-large embeddings (3072 dimensions) for semantic matching.

Schema:
    - id: fact UUID
    - vector: embedding of fact content (voyage-3-large)
    - payload:
        - group_id: tenant isolation
        - subject: source entity name
        - object: target entity name
        - edge_type: relationship type (e.g., "SURPASSED", "REACHED")
        - content: the fact text (for display)

Usage:
    store = FactVectorStore()

    # Index a fact
    store.index_fact(
        fact_id="uuid",
        embedding=[0.1, ...],
        group_id="default",
        subject="alphabet inc.",
        object="$2 trillion",
        edge_type="SURPASSED",
        content="On April 26, 2024, Alphabet surpassed..."
    )

    # Search facts for an entity
    results = store.search_facts_for_entity(
        entity_name="alphabet inc.",
        query_embedding=embedding,
        group_id="default",
        top_k=10
    )
"""

import os
import fcntl
from pathlib import Path
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    VectorParams,
    Distance,
    Filter,
    FieldCondition,
    MatchValue,
)

QDRANT_PATH = "./qdrant_facts"
COLLECTION_NAME = "fact_vectors"
VECTOR_SIZE = 3072  # OpenAI text-embedding-3-large

VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"


def log(msg: str):
    if VERBOSE:
        print(f"[FactVectorStore] {msg}")


def _clear_stale_lock():
    """
    Check if the lock file is held by a dead process and clear it if so.
    This handles cases where the process was killed with SIGKILL.
    """
    lock_path = Path(QDRANT_PATH) / ".lock"
    if not lock_path.exists():
        return

    try:
        # Try to acquire an exclusive lock non-blocking
        fd = os.open(str(lock_path), os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # If we got here, the lock was stale - release it
            fcntl.flock(fd, fcntl.LOCK_UN)
            log("Cleared stale lock file")
        except BlockingIOError:
            # Lock is held by a live process - that's fine
            pass
        finally:
            os.close(fd)
    except (OSError, IOError):
        # Lock file issues - ignore
        pass


class FactVectorStore:
    _instance = None

    def __init__(self):
        # Clear stale locks from crashed processes
        _clear_stale_lock()
        self.client = QdrantClient(path=QDRANT_PATH)
        self._ensure_collection()

    @classmethod
    def get_instance(cls) -> "FactVectorStore":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = FactVectorStore()
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset singleton (for testing)."""
        if cls._instance is not None:
            cls._instance.close()
        cls._instance = None

    def close(self):
        """Close the Qdrant client to release file lock."""
        if hasattr(self, 'client') and self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        if not self.client.collection_exists(COLLECTION_NAME):
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            log(f"Created collection '{COLLECTION_NAME}'")

    def index_fact(
        self,
        fact_id: str,
        embedding: List[float],
        group_id: str,
        subject: str,
        object_entity: str,
        edge_type: str,
        content: str
    ):
        """Index a single fact with its metadata."""
        point = PointStruct(
            id=fact_id,
            vector=embedding,
            payload={
                "group_id": group_id,
                "subject": subject.lower(),  # Normalize for filtering
                "object": object_entity.lower(),
                "edge_type": edge_type,
                "content": content,
                "subject_original": subject,  # Keep original case for display
                "object_original": object_entity,
            }
        )
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=[point]
        )

    def index_facts_batch(self, facts: List[Dict]):
        """
        Batch index multiple facts.

        Each fact dict should have:
            - fact_id, embedding, group_id, subject, object, edge_type, content
        """
        if not facts:
            return

        points = []
        for f in facts:
            points.append(PointStruct(
                id=f["fact_id"],
                vector=f["embedding"],
                payload={
                    "group_id": f["group_id"],
                    "subject": f["subject"].lower(),
                    "object": f["object"].lower(),
                    "edge_type": f["edge_type"],
                    "content": f["content"],
                    "subject_original": f["subject"],
                    "object_original": f["object"],
                }
            ))

        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        log(f"Indexed {len(points)} facts")

    def search_facts_for_entity(
        self,
        entity_name: str,
        query_embedding: List[float],
        group_id: str,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Search facts where entity is either subject or object.

        Returns list of dicts with:
            - fact_id, subject, object, edge_type, content, score, direction
        """
        entity_lower = entity_name.lower()

        # Filter: group_id matches AND (subject = entity OR object = entity)
        # Qdrant doesn't support OR directly in Filter, so we do two queries

        # Query 1: entity is subject
        results_subject = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            query_filter=Filter(
                must=[
                    FieldCondition(key="group_id", match=MatchValue(value=group_id)),
                    FieldCondition(key="subject", match=MatchValue(value=entity_lower)),
                ]
            ),
            limit=top_k,
            with_payload=True
        ).points

        # Query 2: entity is object
        results_object = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            query_filter=Filter(
                must=[
                    FieldCondition(key="group_id", match=MatchValue(value=group_id)),
                    FieldCondition(key="object", match=MatchValue(value=entity_lower)),
                ]
            ),
            limit=top_k,
            with_payload=True
        ).points

        # Merge and dedupe by fact_id, keeping highest score
        seen = {}
        for hit in results_subject + results_object:
            fact_id = hit.id
            if fact_id not in seen or hit.score > seen[fact_id]["score"]:
                seen[fact_id] = {
                    "fact_id": fact_id,
                    "subject": hit.payload.get("subject_original", hit.payload["subject"]),
                    "object": hit.payload.get("object_original", hit.payload["object"]),
                    "edge_type": hit.payload["edge_type"],
                    "content": hit.payload["content"],
                    "score": hit.score,
                    "direction": "outgoing" if hit.payload["subject"].lower() == entity_lower else "incoming"
                }

        # Sort by score descending
        results = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_stats(self) -> Dict:
        """Get collection statistics."""
        if not self.client.collection_exists(COLLECTION_NAME):
            return {"exists": False, "count": 0}

        info = self.client.get_collection(COLLECTION_NAME)
        return {
            "exists": True,
            "count": info.points_count
        }

    def clear(self):
        """Delete and recreate the collection."""
        if self.client.collection_exists(COLLECTION_NAME):
            self.client.delete_collection(COLLECTION_NAME)
        self._ensure_collection()
        log("Collection cleared")


# Convenience function
def get_fact_store() -> FactVectorStore:
    return FactVectorStore.get_instance()
