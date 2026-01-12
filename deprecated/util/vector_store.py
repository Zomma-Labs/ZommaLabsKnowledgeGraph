import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
import voyageai
from src.schemas.relationship import RelationshipDefinition, RelationshipType

load_dotenv()

class VectorStore:
    def __init__(self, collection_name: str = "relationships", path: str = "qdrant_relationships", client: Optional[QdrantClient] = None):
        """
        Initialize the VectorStore with Qdrant and VoyageAI.
        
        Args:
            collection_name: Name of the Qdrant collection.
            path: Path to the Qdrant data directory (for local mode).
            client: Optional existing QdrantClient instance.
        """
        if client:
            self.client = client
        else:
            self.client = QdrantClient(path=path)
        self.collection_name = collection_name
        # Ensure VOYAGE_API_KEY is set in environment
        self.voyage_client = voyageai.Client() 
        self.embedding_model = "voyage-finance-2"
        self.vector_size = 1024 # voyage-finance-2 dimension

        self._ensure_collection()

    def _ensure_collection(self):
        """Create the collection if it doesn't exist."""
        collections = self.client.get_collections()
        if self.collection_name not in [c.name for c in collections.collections]:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE),
            )

    def add_relationships(self, relationships: List[RelationshipDefinition]):
        """
        Embed and add relationships to the vector store.
        
        The text to embed is a combination of description and examples.
        """
        texts = []
        payloads = []
        ids = []

        for i, rel in enumerate(relationships):
            # Construct a rich text representation for embedding
            # "ACQUIRED: One entity purchasing another... Examples: Google acquired YouTube..."
            text = f"{rel.name}: {rel.description} Examples: {'; '.join(rel.examples)}"
            texts.append(text)
            payloads.append(rel.model_dump())
            ids.append(i) # Simple integer IDs for now

        if not texts:
            return

        # Get embeddings
        embeddings = self.voyage_client.embed(
            texts, 
            model=self.embedding_model, 
            input_type="document"
        ).embeddings

        # Upsert to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=models.Batch(
                ids=ids,
                vectors=embeddings,
                payloads=payloads
            )
        )
        print(f"Upserted {len(relationships)} relationships to {self.collection_name}.")

    def search_relationships(self, query: str, limit: int = 20) -> List[RelationshipDefinition]:
        """
        Search for relevant relationships based on a query (long-form description).
        """
        # Embed the query
        query_embedding = self.voyage_client.embed(
            [query], 
            model=self.embedding_model, 
            input_type="query"
        ).embeddings[0]

        # Search Qdrant
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit
        ).points

        results = []
        for hit in search_result:
            # Reconstruct RelationshipDefinition from payload
            # Note: payload is a dict, we need to convert it back to the object
            # We might need to handle potential missing fields if schema changes, but for now it's direct.
            try:
                rel = RelationshipDefinition(**hit.payload)
                results.append(rel)
            except Exception as e:
                print(f"Error parsing search result payload: {e}")
        
        return results
