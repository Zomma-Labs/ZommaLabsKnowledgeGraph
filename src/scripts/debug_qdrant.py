import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import qdrant_client
from qdrant_client import QdrantClient

# print(f"Qdrant Client Version: {qdrant_client.__version__}")

client = QdrantClient(path="qdrant_data")
print(f"Client type: {type(client)}")
print(f"Has search: {hasattr(client, 'search')}")
print(f"Dir: {dir(client)}")
