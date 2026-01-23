"""
Checkpoint management for pipeline resume capability.

Saves intermediate state after each phase so the pipeline can resume
from the last completed phase if it crashes.

Security: Pickle files are only loaded from checkpoints created by this
pipeline locally. Do not load checkpoints from untrusted sources.
"""

import json
import pickle
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4


CHECKPOINTS_DIR = Path("checkpoints")


@dataclass
class CheckpointMetadata:
    """Metadata about a checkpoint run."""
    run_id: str
    input_file: str
    group_id: str
    started_at: str
    last_phase: int  # 0 = not started, 1 = extraction, 2 = resolution, 3 = buffer ready
    cli_args: Dict[str, Any]


class CheckpointManager:
    """Manages checkpoint save/load/cleanup for pipeline runs."""

    def __init__(self, input_file: str, group_id: str, cli_args: Optional[Dict] = None):
        self.input_file = input_file
        self.group_id = group_id
        self.cli_args = cli_args or {}

        # Generate checkpoint directory name
        input_stem = Path(input_file).stem
        self.run_id = uuid4().hex[:8]
        self.checkpoint_dir = CHECKPOINTS_DIR / f"{input_stem}_{group_id}_{self.run_id}"

        self.metadata = CheckpointMetadata(
            run_id=self.run_id,
            input_file=input_file,
            group_id=group_id,
            started_at=datetime.now().isoformat(),
            last_phase=0,
            cli_args=self.cli_args
        )

    @classmethod
    def find_existing(cls, input_file: str, group_id: str) -> Optional["CheckpointManager"]:
        """Find an existing checkpoint for the given input file and group_id."""
        if not CHECKPOINTS_DIR.exists():
            return None

        input_stem = Path(input_file).stem
        prefix = f"{input_stem}_{group_id}_"

        # Find matching checkpoint directories
        matches = [d for d in CHECKPOINTS_DIR.iterdir() if d.is_dir() and d.name.startswith(prefix)]

        if not matches:
            return None

        # Return the most recent one (by directory name, which includes run_id)
        latest = sorted(matches, key=lambda d: d.stat().st_mtime, reverse=True)[0]

        # Load the checkpoint
        return cls._load_from_dir(latest)

    @classmethod
    def _load_from_dir(cls, checkpoint_dir: Path) -> Optional["CheckpointManager"]:
        """Load a CheckpointManager from an existing directory."""
        metadata_file = checkpoint_dir / "metadata.json"
        if not metadata_file.exists():
            return None

        with open(metadata_file) as f:
            meta_dict = json.load(f)

        # Create instance without generating new run_id
        instance = cls.__new__(cls)
        instance.checkpoint_dir = checkpoint_dir
        instance.run_id = meta_dict["run_id"]
        instance.input_file = meta_dict["input_file"]
        instance.group_id = meta_dict["group_id"]
        instance.cli_args = meta_dict.get("cli_args", {})
        instance.metadata = CheckpointMetadata(**meta_dict)

        return instance

    def _ensure_dir(self):
        """Ensure checkpoint directory exists."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _save_metadata(self):
        """Save metadata.json."""
        self._ensure_dir()
        with open(self.checkpoint_dir / "metadata.json", "w") as f:
            json.dump({
                "run_id": self.metadata.run_id,
                "input_file": self.metadata.input_file,
                "group_id": self.metadata.group_id,
                "started_at": self.metadata.started_at,
                "last_phase": self.metadata.last_phase,
                "cli_args": self.metadata.cli_args
            }, f, indent=2)

    def save_phase1(self, extractions: list, document_uuid: str, document_name: str,
                    document_date_str: Optional[str], chunks: list):
        """Save Phase 1 (extraction) results."""
        self._ensure_dir()
        data = {
            "extractions": extractions,
            "document_uuid": document_uuid,
            "document_name": document_name,
            "document_date_str": document_date_str,
            "chunks": chunks
        }
        with open(self.checkpoint_dir / "phase1_extraction.pkl", "wb") as f:
            pickle.dump(data, f)
        self.metadata.last_phase = 1
        self._save_metadata()

    def save_phase2(self, entity_lookup: dict, topic_lookup: dict,
                    dedup_canonical_map: dict, uuid_by_name: dict):
        """Save Phase 2 (resolution) results."""
        self._ensure_dir()
        data = {
            "entity_lookup": entity_lookup,
            "topic_lookup": topic_lookup,
            "dedup_canonical_map": dedup_canonical_map,
            "uuid_by_name": uuid_by_name
        }
        with open(self.checkpoint_dir / "phase2_resolution.pkl", "wb") as f:
            pickle.dump(data, f)
        self.metadata.last_phase = 2
        self._save_metadata()

    def save_phase3(self, buffer, embeddings_done: bool = False):
        """Save Phase 3 (buffer) state."""
        self._ensure_dir()
        data = {
            "buffer": buffer,
            "embeddings_done": embeddings_done
        }
        with open(self.checkpoint_dir / "phase3_buffer.pkl", "wb") as f:
            pickle.dump(data, f)
        self.metadata.last_phase = 3
        self._save_metadata()

    def load_phase1(self) -> Optional[Dict]:
        """Load Phase 1 data if it exists."""
        path = self.checkpoint_dir / "phase1_extraction.pkl"
        if not path.exists():
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def load_phase2(self) -> Optional[Dict]:
        """Load Phase 2 data if it exists."""
        path = self.checkpoint_dir / "phase2_resolution.pkl"
        if not path.exists():
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def load_phase3(self) -> Optional[Dict]:
        """Load Phase 3 data if it exists."""
        path = self.checkpoint_dir / "phase3_buffer.pkl"
        if not path.exists():
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def cleanup(self):
        """Remove checkpoint directory after successful completion."""
        if self.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_dir)

    def delete(self):
        """Alias for cleanup - used when --fresh is specified."""
        self.cleanup()

    def get_resume_phase(self) -> int:
        """Return the phase to resume from (last_phase + 1, or 1 if starting fresh)."""
        return self.metadata.last_phase + 1

    def print_status(self):
        """Print checkpoint status for user."""
        phase_names = {0: "Not started", 1: "Extraction", 2: "Resolution", 3: "Buffer ready"}
        print(f"  Found checkpoint: {self.checkpoint_dir}")
        print(f"    Input: {self.metadata.input_file}")
        print(f"    Last completed: Phase {self.metadata.last_phase} ({phase_names.get(self.metadata.last_phase, 'Unknown')})")
        print(f"    Started: {self.metadata.started_at}")
