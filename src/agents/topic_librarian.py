"""
MODULE: Topic Librarian
DESCRIPTION:
    Resolves extracted topic names to the curated topic ontology using hybrid retrieval:
    1. Vector Search (Semantic Match) via Qdrant + Voyage AI.
    2. Fuzzy Search (Typo/Exact Match) via RapidFuzz.

    If no match is found above threshold, returns None (topic should be discarded).
    This prevents garbage topics like "Transparency And Oversight" from entering the graph.
"""

import os
from typing import List, Dict, Optional, TYPE_CHECKING
from rapidfuzz import process, fuzz

if TYPE_CHECKING:
    from src.util.services import Services

# Control verbose output
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"

QDRANT_PATH = "./qdrant_topics"
COLLECTION_NAME = "topic_ontology"


def log(msg: str):
    """Print only if VERBOSE mode is enabled."""
    if VERBOSE:
        print(msg)


class TopicLibrarian:
    def __init__(self, services: Optional["Services"] = None):
        log("Initializing Topic Librarian...")

        if services is None:
            from src.util.services import get_services
            services = get_services()

        self.embeddings = services.embeddings

        # Initialize Qdrant client for topics
        from qdrant_client import QdrantClient
        self.client = QdrantClient(path=QDRANT_PATH)

        # Load labels for fuzzy matching
        self.label_map = self._load_all_labels()
        self.all_labels = list(self.label_map.keys())
        log(f"   Loaded {len(self.all_labels)} topics for fuzzy matching.")

    def _load_all_labels(self) -> Dict[str, Dict]:
        """
        Fetches all labels, synonyms, and URIs from Qdrant.
        Returns: {Label/Synonym: {"uri": ..., "canonical": ...}}
        """
        label_map = {}

        if not self.client.collection_exists(COLLECTION_NAME):
            log(f"   Collection '{COLLECTION_NAME}' not found. Run topic_loader.py first.")
            return {}

        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=None,
                limit=100,
                with_payload=True,
                with_vectors=False,
                offset=offset
            )

            for point in points:
                payload = point.payload
                if payload and "label" in payload and "uri" in payload:
                    canonical = payload["label"]
                    uri = payload["uri"]

                    # Add canonical label
                    label_map[canonical] = {"uri": uri, "canonical": canonical}

                    # Add synonyms
                    synonyms_str = payload.get("synonyms", "")
                    if synonyms_str:
                        for syn in synonyms_str.split(","):
                            syn = syn.strip()
                            if syn:
                                label_map[syn] = {"uri": uri, "canonical": canonical}

            if offset is None:
                break

        return label_map

    def resolve(self, text: str, top_k: int = 3, threshold: float = 0.70) -> Optional[Dict]:
        """
        Resolves a topic name to the best matching ontology concept.
        Returns the match dictionary or None if no good match.

        Threshold is lower than FIBO entities (0.70 vs 0.9) since topics are more abstract.
        """
        if not text or not text.strip():
            return None

        text = text.strip()

        # 1. Vector Search (Semantic)
        vector_candidates = self._vector_search(text, k=top_k)

        # 2. Fuzzy Search (Lexical)
        fuzzy_candidates = self._fuzzy_search(text, k=top_k)

        # 3. Merge & Rank
        combined = {}

        for cand in vector_candidates:
            uri = cand['uri']
            combined[uri] = {
                "uri": uri,
                "label": cand['label'],
                "definition": cand.get('definition', ''),
                "source": "vector",
                "score": cand['score']
            }

        for cand in fuzzy_candidates:
            uri = cand['uri']
            score = cand['score'] / 100.0  # Normalize to 0-1

            if uri in combined:
                existing_score = combined[uri]['score']
                new_score = (existing_score + score) / 2 + 0.1
                combined[uri]['score'] = min(new_score, 1.0)
                combined[uri]['source'] = "hybrid"
            else:
                combined[uri] = {
                    "uri": uri,
                    "label": cand['label'],
                    "definition": "",
                    "source": "fuzzy",
                    "score": score
                }

        sorted_candidates = sorted(combined.values(), key=lambda x: x['score'], reverse=True)

        if not sorted_candidates:
            log(f"   No topic match for '{text}'")
            return None

        best_match = sorted_candidates[0]

        if best_match['score'] < threshold:
            log(f"   Topic '{text}' best match '{best_match['label']}' score {best_match['score']:.2f} < {threshold}. Rejecting.")
            return None

        log(f"   Topic '{text}' -> '{best_match['label']}' ({best_match['score']:.2f})")
        return best_match

    def resolve_topics(self, topics: List[str], threshold: float = 0.70) -> List[str]:
        """
        Resolves a list of extracted topics to canonical names.
        Returns only valid topics that match the ontology.
        """
        resolved = []
        seen = set()

        for topic in topics:
            match = self.resolve(topic, threshold=threshold)
            if match:
                canonical = match['label']
                if canonical not in seen:
                    resolved.append(canonical)
                    seen.add(canonical)

        return resolved

    def _vector_search(self, text: str, k: int) -> List[Dict]:
        """Queries Qdrant for semantic matches."""
        if not self.client.collection_exists(COLLECTION_NAME):
            return []

        try:
            vector = self.embeddings.embed_query(text)

            results = self.client.query_points(
                collection_name=COLLECTION_NAME,
                query=vector,
                limit=k,
                with_payload=True
            ).points

            candidates = []
            for hit in results:
                candidates.append({
                    "uri": hit.payload["uri"],
                    "label": hit.payload["label"],
                    "definition": hit.payload.get("definition", ""),
                    "score": hit.score
                })
            return candidates

        except Exception as e:
            log(f"Topic vector search failed: {e}")
            return []

    def _fuzzy_search(self, text: str, k: int) -> List[Dict]:
        """Performs fuzzy matching on in-memory labels."""
        if not self.all_labels:
            return []

        results = process.extract(text, self.all_labels, scorer=fuzz.WRatio, limit=k)

        candidates = []
        for match, score, _ in results:
            data = self.label_map.get(match)
            if data:
                candidates.append({
                    "uri": data['uri'],
                    "label": data['canonical'],  # Use canonical name
                    "score": score
                })
        return candidates


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

    # Test the librarian
    librarian = TopicLibrarian()

    test_queries = [
        "Inflation",
        "Price Increases",
        "inflationary pressure",
        "Transparency And Oversight",  # Should NOT match
        "Consumer Confidence",
        "AI",
        "Artificial Intelligence",
        "market volatility",
        "Random Garbage Topic",  # Should NOT match
        "employment",
        "CPI",
        "GDP growth",
    ]

    print("\nTopic Resolution Tests:")
    print("-" * 50)

    for q in test_queries:
        result = librarian.resolve(q)
        if result:
            print(f"'{q}' -> '{result['label']}' ({result['score']:.2f})")
        else:
            print(f"'{q}' -> REJECTED (no match)")
