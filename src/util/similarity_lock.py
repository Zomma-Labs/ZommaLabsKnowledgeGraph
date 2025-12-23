import asyncio
from typing import List, Dict, Any, Optional, ClassVar
import uuid
from scipy.spatial.distance import cosine as scipy_cosine_distance

class SimilarityLockManager:
    """
    Manages locks based on vector similarity.
    Ensures that semantically similar terms are processed serially.
    
    NOTE: Class-level state is initialized lazily and thread-safely.
    """
    _active_operations: ClassVar[Optional[List[Dict[str, Any]]]] = None
    _lock: ClassVar[Optional[asyncio.Lock]] = None
    _initialized: ClassVar[bool] = False

    @classmethod
    def _ensure_initialized(cls) -> None:
        """Thread-safe lazy initialization of class state."""
        if not cls._initialized:
            cls._active_operations = []
            cls._lock = asyncio.Lock()
            cls._initialized = True

    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        """
        Calculates cosine similarity between two vectors using scipy's optimized C implementation.
        Returns value between -1 and 1 (1 = identical, 0 = orthogonal, -1 = opposite).
        """
        if not v1 or not v2:
            return 0.0
        try:
            # scipy.spatial.distance.cosine returns DISTANCE (1 - similarity)
            # So we compute: similarity = 1 - distance
            return 1.0 - scipy_cosine_distance(v1, v2)
        except (ValueError, ZeroDivisionError):
            # Handle zero-norm vectors
            return 0.0

    @classmethod
    async def acquire_lock(cls, vector: List[float], term: str) -> str:
        """
        Pauses if a semantically similar operation is already in progress.
        Returns a unique lock_id that MUST be used to release the lock.
        """
        cls._ensure_initialized()
        
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
        cls._ensure_initialized()
        
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

    @classmethod
    def reset(cls) -> None:
        """Reset class state. Useful for testing."""
        cls._active_operations = None
        cls._lock = None
        cls._initialized = False
