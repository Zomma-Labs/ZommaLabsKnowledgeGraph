"""
MODULE: FIBO Loader (The Librarian's Bookshelf - Complete Edition)
DESCRIPTION: 
    Downloads the full FIBO Ontology (Master/Latest), parses ALL modules, 
    and indexes them into Qdrant to create a comprehensive Ground Truth.

USAGE:
    python src/tools/fibo_loader.py
"""

import rdflib
import uuid
import os
import sys
import json
import glob
import shutil
import requests
import zipfile
import io
from typing import List, Dict
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from src.util.llm_client import get_embeddings

# --- Configuration ---
# The official "Master" zip contains the entire ontology
# GitHub Archive is more reliable than the spec site
FIBO_ZIP_URL = "https://github.com/edmcouncil/fibo/archive/refs/heads/master.zip"
DATA_DIR = "./data/fibo_source"
QDRANT_PATH = "./qdrant_fibo"
COLLECTION_NAME = "fibo_entities"

# Limits
VECTOR_SIZE = 1024  # Voyage finance-2
BATCH_SIZE = 100    # Batch size for Embedding API and Qdrant Upsert

def download_and_extract_fibo():
    """Downloads the latest FIBO zip and extracts it locally."""
    if os.path.exists(DATA_DIR):
        print(f"📂 FIBO directory found at {DATA_DIR}. Skipping download.")
        # Optional: Uncomment to force fresh download
        # shutil.rmtree(DATA_DIR)
        return

    print(f"⬇️  Downloading FIBO from {FIBO_ZIP_URL}...")
    try:
        response = requests.get(FIBO_ZIP_URL)
        response.raise_for_status()
        
        print("📦 Extracting archive...")
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(DATA_DIR)
        print(f"✅ Extracted to {DATA_DIR}")
    except Exception as e:
        print(f"❌ Failed to download/extract FIBO: {e}")
        sys.exit(1)

def parse_all_fibo_files() -> List[Dict]:
    """Recursively finds and parses all .ttl files in the data directory."""
    print("📚 Parsing RDF files with Domain Filtering...")
    g = rdflib.Graph()
    
    # Find all RDF files (Turtle .ttl or RDF/XML .rdf)
    # The GitHub archive uses .rdf (XML) mostly.
    all_files = glob.glob(f"{DATA_DIR}/**/*", recursive=True)
    files = [f for f in all_files if f.endswith('.ttl') or f.endswith('.rdf')]
    
    print(f"   - Found {len(files)} ontology modules.")

    for file_path in tqdm(files, desc="Parsing Ontology Files"):
        try:
            # Detect format
            fmt = "turtle" if file_path.endswith('.ttl') else "xml"
            g.parse(file_path, format=fmt)
        except Exception as e:
            # pass silent for robustness
            pass

    print(f"   ✅ Loaded {len(g)} total triples.")

    # --- THE UPDATED QUERY ---
    # We select the URI string so we can filter in Python (easier than complex SPARQL regex)
    # Added: ?example for skos:example
    query = """
        SELECT DISTINCT ?entity ?label ?definition ?example
        WHERE {
            ?entity a owl:Class .
            { ?entity rdfs:label ?label } UNION { ?entity skos:prefLabel ?label } .
            OPTIONAL { ?entity skos:definition ?definition } .
            OPTIONAL { ?entity skos:example ?example } .
            FILTER NOT EXISTS { ?entity owl:deprecated "true"^^xsd:boolean }
        }
    """
    
    print("🕵️  Querying for Concepts...")
    
    # Map to aggregate examples (URI -> {label, def, examples: set})
    concepts_map = {}
    
    # 🚫 BLOCKLIST: Namespaces that are too abstract for a Trader's KG
    # 'Arrangements', 'Utilities', 'Relations' usually contain meta-logic
    SKIP_PATTERNS = [
        "/FND/Utilities/", 
        "/FND/Relations/", 
        "/FND/GoalsAndObjectives/",
        # "/FND/Arrangements/" # Moved to ALLOWLIST
    ]

    # ✅ ALLOWLIST: FND modules we actually want
    # We want 'Agents' (People/Orgs), 'Places' (Addresses), 'Dates' (Tenors)
    # README implies we handle "Contracts" and "Legal" entities, so we keep Agreements.
    KEEP_FND_PATTERNS = [
        "/FND/AgentsAndPeople/",
        "/FND/Places/",
        "/FND/DatesAndTimes/",
        "/FND/Law/",
        "/FND/Organizations/",
        "/FND/Agreements/", # Critical for Contracts/Legal
        "/FND/Arrangements/" # Critical for Documents/Reporting (Beige Book)
    ]

    for row in g.query(query):
        uri = str(row["entity"])
        
        # --- FILTERING LOGIC ---
        
        # 1. If it's a core financial module, KEEP IT
        is_core_finance = any(x in uri for x in ["/SEC/", "/DER/", "/IND/", "/BE/", "/FBC/", "/LOAN/"])
        
        # 2. If it's Foundation (FND), only keep specific useful sub-modules
        is_useful_foundation = any(x in uri for x in KEEP_FND_PATTERNS)
        
        # 3. Explicitly skip known noise
        is_noise = any(x in uri for x in SKIP_PATTERNS)

        # DECISION:
        if is_noise:
            continue
        if not (is_core_finance or is_useful_foundation):
            # Skip obscure abstract ontologies
            continue
            
        label = str(row["label"])
        definition = str(row["definition"]) if row["definition"] else ""
        example = str(row["example"]) if row.get("example") else ""
        
        if uri not in concepts_map:
            concepts_map[uri] = {
                "label": label,
                "definition": definition,
                "examples": set()
            }
        
        if example:
            concepts_map[uri]["examples"].add(example)

    # Convert map to flat list for indexing
    concepts = []
    for uri, data in concepts_map.items():
        ex_str = "; ".join(sorted(list(data["examples"])))
        search_text = f"Term: {data['label']}\nDefinition: {data['definition']}"
        if ex_str:
            search_text += f"\nExamples: {ex_str}"
            
        concepts.append({
            "uri": uri,
            "label": data['label'],
            "definition": data['definition'],
            "examples": ex_str, # Store in payload too
            "search_text": search_text
        })

    print(f"   ✅ Extracted {len(concepts)} High-Value concepts (Filtered).")
    return concepts

def load_aliases() -> List[Dict]:
    """Loads manual aliases from src/config/aliases/*.json"""
    print("📂 Loading Manual Aliases...")
    alias_files = glob.glob("src/config/aliases/*.json")
    
    aliases = []
    for filepath in alias_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                for item in data:
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
            
    return aliases

def batch_process(data, batch_size):
    """Yields successive n-sized chunks from data."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def index_to_qdrant(concepts: List[Dict]):
    """Embeds text and pushes to Vector DB in batches."""
    print(f"🧠 Generating Embeddings & Indexing ({len(concepts)} items)...")
    
    embeddings = get_embeddings()
    client = QdrantClient(path=QDRANT_PATH)
    
    # 1. Ensure Collection Exists
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    
    # 2. Process in Batches
    total = len(concepts)
    for i, batch in enumerate(batch_process(concepts, BATCH_SIZE)):
        # Generate Embeddings for Batch
        texts = [c["search_text"] for c in batch]
        try:
            vectors = embeddings.embed_documents(texts)
            
            points = []
            for j, concept in enumerate(batch):
                points.append(PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, concept["uri"])),
                    vector=vectors[j],
                    payload={
                        "uri": concept["uri"],
                        "label": concept["label"],
                        "definition": concept["definition"],
                        "examples": concept.get("examples", "")
                    }
                ))
            
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            print(f"   Indexed batch {i+1} ({(i+1)*BATCH_SIZE}/{total})")
            
        except Exception as e:
            print(f"   ❌ Batch failed: {e}")

    print("✅ Indexing Complete.")

if __name__ == "__main__":
    download_and_extract_fibo()
    fibo_data = parse_all_fibo_files()
    alias_data = load_aliases()
    
    all_concepts = fibo_data + alias_data
    if all_concepts:
        index_to_qdrant(all_concepts)
    else:
        print("⚠️ No concepts found to index.")
