import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
import uuid

class SimilarityLockManager:
    """
    Manages locks based on vector similarity.
    Ensures that semantically similar terms are processed serially.
    """
    _active_operations: List[Dict[str, Any]] = [] # List of {'vector': np.array, 'event': asyncio.Event, 'term': str}
    _lock = asyncio.Lock() # Protects the _active_operations list itself

    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        """Calculates cosine similarity between two vectors."""
        if not v1 or not v2:
            return 0.0
        # Convert to numpy for speed if not already
        a = np.array(v1)
        b = np.array(v2)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    @classmethod
    async def acquire_lock(cls, vector: List[float], term: str) -> str:
        """
        Pauses if a semantically similar operation is already in progress.
        Returns a unique lock_id that MUST be used to release the lock.
        """
        while True:
            # Check for conflicts
            conflict_event = None
            
            async with cls._lock:
                for op in cls._active_operations:
                    sim = cls.cosine_similarity(vector, op['vector'])
                    if sim > 0.90:
                        # Found a conflict! We must wait.
                        conflict_event = op['event']
                        break
                
                if not conflict_event:
                    # No conflict found. Register ourselves.
                    my_event = asyncio.Event()
                    lock_id = str(uuid.uuid4())
                    cls._active_operations.append({
                        'id': lock_id,
                        'vector': vector,
                        'event': my_event,
                        'term': term
                    })
                    return lock_id # Return the handle

            # Wait outside lock
            if conflict_event:
                await conflict_event.wait()

    @classmethod
    async def release_lock(cls, lock_id: str) -> None:
        """
        Unregisters the operation by ID and triggers waiting tasks.
        """
        async with cls._lock:
            # Find and remove by ID
            idx_to_remove = -1
            for i, op in enumerate(cls._active_operations):
                if op['id'] == lock_id:
                    idx_to_remove = i
                    op['event'].set() # Wake up everyone waiting
                    break
            
            if idx_to_remove != -1:
                cls._active_operations.pop(idx_to_remove)
            else:
                print(f"CRITICAL WARNING: Attempted to release lock_id '{lock_id}' but it was not found active!")

