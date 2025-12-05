"""
MODULE: FIBO Librarian
DESCRIPTION: 
    Resolves entity names to FIBO URIs using a Hybrid Retrieval strategy:
    1. Vector Search (Semantic Match) via Qdrant + Voyage AI.
    2. Fuzzy Search (Typo/Exact Match) via RapidFuzz.
"""

from typing import List, Dict, Optional, TYPE_CHECKING
from rapidfuzz import process, fuzz

if TYPE_CHECKING:
    from src.util.services import Services

# Reuse configuration (should ideally be in settings.py)
COLLECTION_NAME = "fibo_entities"

class FIBOLibrarian:
    def __init__(self, services: Optional["Services"] = None):
        print("📚 Initializing FIBO Librarian...")
        if services is None:
            from src.util.services import get_services
            services = get_services()
        
        self.client = services.qdrant_fibo
        self.embeddings = services.embeddings
        
        # Load "Ground Truth" labels for Fuzzy Search
        self.label_map = self._load_all_labels()
        self.all_labels = list(self.label_map.keys())
        print(f"   ✅ Loaded {len(self.all_labels)} entities for fuzzy matching.")

    def _load_all_labels(self) -> Dict[str, str]:
        """
        Fetches all labels and URIs from Qdrant to build an in-memory index.
        Returns: {Label: URI}
        """
        label_map = {}
        
        # Check if collection exists
        if not self.client.collection_exists(COLLECTION_NAME):
            print(f"   ⚠️ Collection '{COLLECTION_NAME}' not found. Run fibo_loader.py first.")
            return {}

        # Scroll through all points
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
                    label_map[payload["label"]] = payload["uri"]
            
            if offset is None:
                break
                
        return label_map

    def resolve(self, text: str, top_k: int = 5, threshold: float = 0.9) -> Optional[Dict]:
        """
        Resolves an entity name to the best matching FIBO concept.
        Returns the top match dictionary or None.
        """
        if not text:
            return None

        # 1. Vector Search (Semantic)
        vector_candidates = self._vector_search(text, k=top_k)
        
        # 2. Fuzzy Search (Lexical)
        fuzzy_candidates = self._fuzzy_search(text, k=top_k)
        
        # 3. Merge & Rank
        # We'll use a simple strategy: Combine and deduplicate by URI.
        # If a candidate appears in both, we can boost confidence.
        # For now, we'll return the one with the highest normalized score.
        
        combined = {} # URI -> {data, score}
        
        # Process Vector Results (Score is Cosine Similarity 0-1)
        for cand in vector_candidates:
            uri = cand['uri']
            score = cand['score'] # 0.0 to 1.0
            combined[uri] = {
                "uri": uri,
                "label": cand['label'],
                "definition": cand['definition'],
                "source": "vector",
                "score": score
            }
            
        # Process Fuzzy Results (Score is 0-100)
        for cand in fuzzy_candidates:
            uri = cand['uri']
            score = cand['score'] / 100.0 # Normalize to 0-1
            
            if uri in combined:
                # Boost if found in both (simple average or max + boost)
                existing_score = combined[uri]['score']
                new_score = (existing_score + score) / 2 + 0.1 # Boost for agreement
                combined[uri]['score'] = min(new_score, 1.0)
                combined[uri]['source'] = "hybrid"
            else:
                combined[uri] = {
                    "uri": uri,
                    "label": cand['label'],
                    "definition": "N/A (Fuzzy Match)", # We might not have def in memory map
                    "source": "fuzzy",
                    "score": score
                }
                
        # Sort by score
        sorted_candidates = sorted(combined.values(), key=lambda x: x['score'], reverse=True)
        
        if not sorted_candidates:
            return None
            
        best_match = sorted_candidates[0]
        
        if best_match['score'] < threshold:
            print(f"   ⚠️ Best match '{best_match['label']}' score {best_match['score']:.2f} < {threshold}. Rejecting.")
            return None
            
        return best_match

    def _vector_search(self, text: str, k: int) -> List[Dict]:
        """Queries Qdrant for semantic matches."""
        try:
            vector = self.embeddings.embed_query(text)
            # Use query_points as search might be deprecated/missing in this version
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
            print(f"Vector search failed: {e}")
            return []

    def _fuzzy_search(self, text: str, k: int) -> List[Dict]:
        """Performs fuzzy matching on in-memory labels."""
        if not self.all_labels:
            return []
            
        # extract returns list of (match, score, index)
        results = process.extract(text, self.all_labels, scorer=fuzz.WRatio, limit=k)
        
        candidates = []
        for match, score, _ in results:
            uri = self.label_map.get(match)
            if uri:
                candidates.append({
                    "uri": uri,
                    "label": match,
                    "score": score # 0-100
                })
        return candidates

if __name__ == "__main__":
    # Simple test
    librarian = FIBOLibrarian()
    
    test_queries = ["Berkshire Hathway", "Apple Inc", "Investment Fund"]
    for q in test_queries:
        print(f"\nQuery: {q}")
        result = librarian.resolve(q)
        if result:
            print(f"   ✅ Match: {result['label']} ({result['score']:.2f})")
            print(f"      URI: {result['uri']}")
            print(f"      Source: {result['source']}")
        else:
            print("   ❌ No match found.")
