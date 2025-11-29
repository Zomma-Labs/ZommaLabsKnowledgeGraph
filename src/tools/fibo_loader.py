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
import sys
import json
import glob
from typing import List, Dict

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from src.util.llm_client import get_embeddings

# --- Configuration ---
FIBO_BASE = "https://spec.edmcouncil.org/fibo/ontology/master/latest"

FIBO_URLS = [
    # --- 1. BUSINESS ENTITIES (The Actors) ---
    f"{FIBO_BASE}/BE/LegalEntities/LegalPersons.ttl",
    f"{FIBO_BASE}/BE/LegalEntities/CorporateBodies.ttl",
    f"{FIBO_BASE}/BE/Corporations/Corporations.ttl",
    f"{FIBO_BASE}/BE/GovernmentEntities/GovernmentEntities.ttl",
    f"{FIBO_BASE}/BE/Partnerships/Partnerships.ttl",
    f"{FIBO_BASE}/BE/Trusts/Trusts.ttl",
    f"{FIBO_BASE}/BE/OwnershipAndControl/OwnershipParties.ttl",

    # --- 2. REGULATORS & AGENCIES (The Rules) ---
    f"{FIBO_BASE}/FBC/FunctionalEntities/RegulatoryAgencies.ttl",
    f"{FIBO_BASE}/FBC/FunctionalEntities/NorthAmericanEntities/USRegulatoryAgencies.ttl",
    f"{FIBO_BASE}/FBC/FunctionalEntities/RegistrationAuthorities.ttl",

    # --- 3. MACROECONOMICS (The Numbers) ---
    f"{FIBO_BASE}/IND/EconomicIndicators/EconomicIndicators.ttl",
    f"{FIBO_BASE}/IND/InterestRates/InterestRates.ttl",
    f"{FIBO_BASE}/IND/ForeignExchange/ForeignExchange.ttl",

    # --- 4. FINANCIAL INSTRUMENTS (The Assets) ---
    f"{FIBO_BASE}/FBC/FinancialInstruments/FinancialInstruments.ttl",
    f"{FIBO_BASE}/SEC/Equities/Equities.ttl",
    f"{FIBO_BASE}/SEC/Debt/Bonds.ttl",
    f"{FIBO_BASE}/LOAN/LoansGeneral/Loans.ttl",

    # --- 5. FOUNDATIONS (Time & Place) ---
    f"{FIBO_BASE}/FND/Places/Locations.ttl",
    f"{FIBO_BASE}/FND/Places/Addresses.ttl",
    f"{FIBO_BASE}/FND/AgentsAndPeople/Agents.ttl",
    f"{FIBO_BASE}/FND/Agreements/Contracts.ttl",

    # --- 6. DOCUMENTS & REPORTS (NEW - Crucial for "Beige Book") ---
    # Defines "Report", "Publication", "Record".
    # Without this, "Beige Book" is just a string. With this, it is a "Periodical".
    f"{FIBO_BASE}/FND/Arrangements/Documents.ttl",
    f"{FIBO_BASE}/FND/Arrangements/Reporting.ttl",

    # --- 7. MARKETS & ANALYTICS (NEW - Crucial for "Economic Conditions") ---
    # Defines "Market", "Situation", "Analysis", "Outlook".
    # This allows you to resolve "Labor Market" to a specific concept.
    f"{FIBO_BASE}/FBC/FunctionalEntities/Markets.ttl",
    f"{FIBO_BASE}/FND/Utilities/Analytics.ttl",
    f"{FIBO_BASE}/FND/DatesAndTimes/Occurrences.ttl", # Defines "Events" vs "Situations"
]

QDRANT_PATH = "./qdrant_fibo"  # Local persistence
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

def load_aliases() -> List[Dict]:
    """Loads manual aliases from src/config/aliases/*.json"""
    print("📂 Loading Manual Aliases...")
    alias_files = glob.glob("src/config/aliases/*.json")
    
    aliases = []
    for filepath in alias_files:
        print(f"   - Reading {filepath}...")
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                for item in data:
                    # Construct search text including synonyms
                    synonyms_str = ", ".join(item.get("synonyms", []))
                    search_text = f"Term: {item['label']}\nSynonyms: {synonyms_str}\nDefinition: {item['definition']}"
                    
                    aliases.append({
                        "uri": item["uri"],
                        "label": item["label"],
                        "definition": item["definition"],
                        "search_text": search_text
                    })
        except Exception as e:
            print(f"   ⚠️ Failed to load aliases from {filepath}: {e}")
            
    print(f"   ✅ Loaded {len(aliases)} manual aliases.")
    return aliases

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
    print(f"   ✅ Successfully indexed {len(points)} concepts (FIBO + Aliases).")

if __name__ == "__main__":
    fibo_data = fetch_and_parse_fibo()
    alias_data = load_aliases()
    
    # Combine datasets
    all_concepts = fibo_data + alias_data
    
    if all_concepts:
        index_to_qdrant(all_concepts)
