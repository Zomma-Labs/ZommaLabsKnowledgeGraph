"""
MODULE: FIBO Loader (The Librarian's Bookshelf)
DESCRIPTION: 
    Parses FIBO RDF/Turtle files and indexes them into Qdrant.
    It creates the 'Ground Truth' dictionary for the Entity Resolver.

USAGE:
    python src/tools/fibo_loader.py
"""

import rdflib
import uuid
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

# --- Configuration ---
# In production, download the full FIBO zip. For now, we stream key modules.
FIBO_URLS = [
    "https://spec.edmcouncil.org/fibo/ontology/master/latest/BE/LegalEntities/LegalPersons.ttl",
    "https://spec.edmcouncil.org/fibo/ontology/master/latest/BE/Corporations/Corporations.ttl",
    "https://spec.edmcouncil.org/fibo/ontology/master/latest/FND/AgentsAndPeople/Agents.ttl",
]

QDRANT_PATH = "./qdrant_data"  # Local persistence
COLLECTION_NAME = "fibo_entities"
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # Fast, cheap, effective

def fetch_and_parse_fibo() -> List[Dict]:
    """Parses RDF to extract Label, Definition, and URI."""
    print("📚 Reading FIBO Ontology files...")
    g = rdflib.Graph()
    
    for url in FIBO_URLS:
        print(f"   - Fetching {url}...")
        try:
            g.parse(url, format="turtle")
        except Exception as e:
            print(f"   ⚠️ Failed to load {url}: {e}")

    print(f"   ✅ Loaded {len(g)} triples.")

    # SPARQL Query to get standardized terms
    # We look for Classes (Concepts) and Named Individuals (Specific Entities if any)
    query = """
        SELECT DISTINCT ?entity ?label ?definition
        WHERE {
            ?entity a owl:Class .
            ?entity rdfs:label ?label .
            OPTIONAL { ?entity skos:definition ?definition } .
            FILTER(LANG(?label) = "en")
        }
    """
    
    concepts = []
    for row in g.query(query):
        uri = str(row["entity"])
        label = str(row["label"])
        definition = str(row["definition"]) if row["definition"] else "Financial entity defined in FIBO."
        
        # The "Rich Text" we will vector embed
        search_text = f"Term: {label}\nDefinition: {definition}"
        
        concepts.append({
            "uri": uri,
            "label": label,
            "definition": definition,
            "search_text": search_text
        })
    
    print(f"   found {len(concepts)} unique concepts.")
    return concepts

def index_to_qdrant(concepts: List[Dict]):
    """Embeds text and pushes to Vector DB."""
    print("🧠 Generating Embeddings (this may take a moment)...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Batch embedding
    texts = [c["search_text"] for c in concepts]
    vectors = model.encode(texts, show_progress_bar=True)
    
    print("💾 Indexing to Qdrant...")
    client = QdrantClient(path=QDRANT_PATH)
    
    # 1. Create Collection (if needed)
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
    
    # 2. Upload Points
    points = []
    for i, concept in enumerate(concepts):
        points.append(PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, concept["uri"])), # Deterministic ID based on URI
            vector=vectors[i].tolist(),
            payload={
                "uri": concept["uri"],
                "label": concept["label"],
                "definition": concept["definition"]
            }
        ))
    
    # Upsert in batches (Qdrant handles batching, but good practice to know)
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print(f"   ✅ Successfully indexed {len(points)} FIBO concepts.")

if __name__ == "__main__":
    data = fetch_and_parse_fibo()
    if data:
        index_to_qdrant(data)