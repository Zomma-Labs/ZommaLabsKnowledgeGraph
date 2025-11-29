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
import os
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from src.util.llm_client import get_embeddings

# --- Configuration ---
# --- Configuration ---
# We stream these specific modules to build a "Fed-Aware" Knowledge Graph.
FIBO_URLS = [
    # 1. THE ACTORS (Companies & Government)
    # Essential for "Green Dot Bank" or "Department of Labor"
    "https://spec.edmcouncil.org/fibo/ontology/master/latest/BE/LegalEntities/LegalPersons.ttl",
    "https://spec.edmcouncil.org/fibo/ontology/master/latest/BE/Corporations/Corporations.ttl",
    "https://spec.edmcouncil.org/fibo/ontology/master/latest/BE/GovernmentEntities/GovernmentEntities.ttl",
    
    # 2. THE REGULATORS (The Fed System)
    # Essential for "Board of Governors", "District 12", "Central Bank"
    # Note: USRegulatoryAgencies is critical for specific Fed Banks (NY Fed, SF Fed)
    "https://spec.edmcouncil.org/fibo/ontology/master/latest/FBC/FunctionalEntities/RegulatoryAgencies.ttl",
    "https://spec.edmcouncil.org/fibo/ontology/master/latest/FBC/FunctionalEntities/NorthAmericanEntities/USRegulatoryAgencies.ttl",

    # 3. THE ECONOMY (Macro Signals)
    # Essential for "Inflation", "CPI", "Unemployment", "GDP"
    # This is the single most important file for the Beige Book.
    "https://spec.edmcouncil.org/fibo/ontology/master/latest/IND/EconomicIndicators/EconomicIndicators.ttl",
    "https://spec.edmcouncil.org/fibo/ontology/master/latest/IND/InterestRates/InterestRates.ttl",

    # 4. THE GEOGRAPHY (Districts & Regions)
    # Essential for "New York Region", "Midwest", "Sunbelt"
    "https://spec.edmcouncil.org/fibo/ontology/master/latest/FND/Places/Locations.ttl",
    "https://spec.edmcouncil.org/fibo/ontology/master/latest/FND/Places/Addresses.ttl",

    # 5. THE PRODUCTS (What they sell/buy)
    # Essential for "Mortgage demand", "Commercial Loans", "Securities"
    "https://spec.edmcouncil.org/fibo/ontology/master/latest/FBC/FinancialInstruments/FinancialInstruments.ttl",
    "https://spec.edmcouncil.org/fibo/ontology/master/latest/LOAN/LoansGeneral/Loans.ttl", 
]

QDRANT_PATH = "./qdrant_data"  # Local persistence
COLLECTION_NAME = "fibo_entities"
# Voyage finance-2 dimension is 1024
VECTOR_SIZE = 1024 

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
            { ?entity rdfs:label ?label } UNION { ?entity skos:prefLabel ?label } .
            OPTIONAL { ?entity skos:definition ?definition } .
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
    
    embeddings = get_embeddings()
    
    # Batch embedding
    texts = [c["search_text"] for c in concepts]
    # embed_documents returns List[List[float]]
    vectors = embeddings.embed_documents(texts)
    
    print("💾 Indexing to Qdrant...")
    client = QdrantClient(path=QDRANT_PATH)
    
    # 1. Create Collection (if needed)
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    
    # 2. Upload Points
    points = []
    for i, concept in enumerate(concepts):
        points.append(PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, concept["uri"])), # Deterministic ID based on URI
            vector=vectors[i],
            payload={
                "uri": concept["uri"],
                "label": concept["label"],
                "definition": concept["definition"]
            }
        ))
    
    # Upsert in batches (Qdrant handles batching, but good practice to know)
    # Qdrant client handles batching automatically for upsert usually, but explicit is fine.
    # For simplicity, we pass all points. If list is huge, chunk it.
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print(f"   ✅ Successfully indexed {len(points)} FIBO concepts.")

if __name__ == "__main__":
    data = fetch_and_parse_fibo()
    if data:
        index_to_qdrant(data)
