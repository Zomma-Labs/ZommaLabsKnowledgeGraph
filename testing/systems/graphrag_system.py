"""
MODULE: GraphRAGSystem
DESCRIPTION: Wrapper for Microsoft's GraphRAG library for evaluation comparison.
             Uses GraphRAG Python API directly for fast, concurrent queries.
"""

import asyncio
import re
from pathlib import Path
from typing import Literal

import pandas as pd
from dotenv import load_dotenv

#import graphrag.api as graphrag_api
#from graphrag.config.load_config import load_config

load_dotenv()


class GraphRAGIndexNotFoundError(Exception):
    """Raised when the GraphRAG index directory does not exist or is invalid."""

    pass


class GraphRAGSystem:
    """Wrapper for Microsoft's GraphRAG library.

    This system uses the GraphRAG Python API directly to query a pre-built index.
    Data is loaded once at initialization and reused for all queries, making
    concurrent queries fast.

    Supports search modes:
    - global: Uses community summaries for broad, thematic questions
    - local: Uses specific entity context for focused questions

    Note: Requires a pre-built GraphRAG index. Run setup_graphrag.py first.
    """

    def __init__(
        self,
        index_path: str,
        search_type: Literal["global", "local"] = "local",
        community_level: int = 2,
    ):
        """Initialize GraphRAGSystem with a pre-built index.

        Loads all required dataframes into memory for fast querying.

        Args:
            index_path: Path to GraphRAG index directory (or root project dir).
            search_type: Search method - "global" or "local".
            community_level: Leiden hierarchy level for community reports.
        """
        self.index_path = Path(index_path)
        self.search_type = search_type
        self.community_level = community_level

        # Determine root directory (parent of output/ if pointing to output dir)
        if self.index_path.name == "output":
            self.root_path = self.index_path.parent
        else:
            self.root_path = self.index_path
            # Check if output subdirectory exists
            if (self.index_path / "output").exists():
                pass  # root_path is correct
            elif self.index_path.exists():
                # Pointing directly to output directory
                self.root_path = self.index_path.parent

        self._validate_index()
        self._load_data()

    def _validate_index(self) -> None:
        """Validate that the GraphRAG index exists and has required files."""
        if not self.root_path.exists():
            raise GraphRAGIndexNotFoundError(
                f"GraphRAG root path does not exist: {self.root_path}\n"
                "Please run: uv run testing/setup/setup_graphrag.py"
            )

        # Determine output directory
        output_dir = self.root_path / "output"
        if not output_dir.exists():
            output_dir = self.root_path

        # Check for required parquet files (GraphRAG 2.x naming)
        required_files = ["entities.parquet", "community_reports.parquet"]
        missing = [f for f in required_files if not (output_dir / f).exists()]

        if missing:
            raise GraphRAGIndexNotFoundError(
                f"GraphRAG index is incomplete. Missing: {missing}\n"
                "Please run: uv run testing/setup/setup_graphrag.py"
            )

    def _load_data(self) -> None:
        """Load all required dataframes into memory.

        Uses synchronous parquet loading to avoid nested event loop issues.
        """
        # Load config
        self.config = load_config(self.root_path.resolve())

        # Determine output directory
        output_dir = self.root_path / "output"
        if not output_dir.exists():
            output_dir = self.root_path

        # Load dataframes directly from parquet (synchronous, no event loop needed)
        self.entities = pd.read_parquet(output_dir / "entities.parquet")
        self.communities = pd.read_parquet(output_dir / "communities.parquet")
        self.community_reports = pd.read_parquet(
            output_dir / "community_reports.parquet"
        )

        # Local search needs additional tables
        if self.search_type == "local":
            self.text_units = pd.read_parquet(output_dir / "text_units.parquet")
            self.relationships = pd.read_parquet(output_dir / "relationships.parquet")

            # Covariates are optional
            covariates_path = output_dir / "covariates.parquet"
            if covariates_path.exists():
                self.covariates = pd.read_parquet(covariates_path)
            else:
                self.covariates = None

    @property
    def index_info(self) -> dict:
        """Return information about the loaded index."""
        info = {
            "root_path": str(self.root_path),
            "search_type": self.search_type,
            "community_level": self.community_level,
        }

        if hasattr(self, "entities"):
            info["entities"] = len(self.entities)
        if hasattr(self, "community_reports"):
            info["community_reports"] = len(self.community_reports)

        return info

    async def query(self, question: str, verbose: bool = False) -> tuple[str, list[dict]]:
        """Query the GraphRAG index using the Python API.

        Args:
            question: The question to answer.
            verbose: If True, print debug info.

        Returns:
            Tuple of (answer, evidence) where evidence contains community reports used.
        """
        try:
            if verbose:
                print(f"[GraphRAG] Starting query: {question[:50]}...")

            if self.search_type == "local":
                response, context_data = await graphrag_api.local_search(
                    config=self.config,
                    entities=self.entities,
                    communities=self.communities,
                    community_reports=self.community_reports,
                    text_units=self.text_units,
                    relationships=self.relationships,
                    covariates=self.covariates,
                    community_level=self.community_level,
                    response_type="Single paragraph",
                    query=question,
                )
            else:  # global search
                response, context_data = await graphrag_api.global_search(
                    config=self.config,
                    entities=self.entities,
                    communities=self.communities,
                    community_reports=self.community_reports,
                    community_level=self.community_level,
                    dynamic_community_selection=False,
                    response_type="Single paragraph",
                    query=question,
                )

            # Convert response to string if needed
            answer = str(response) if response else ""

            # Extract evidence from the response
            evidence = self._extract_evidence(answer)

            return answer, evidence

        except Exception as e:
            # Return error message as the answer
            return f"Error during query: {str(e)}", []

    def _extract_evidence(self, answer: str) -> list[dict]:
        """Extract evidence references from the GraphRAG response.

        GraphRAG includes [Data: Reports (X, Y, Z)] references in responses.
        """
        evidence = []

        # Find all Data: Reports references
        pattern = r"\[Data: Reports \(([^\)]+)\)\]"
        matches = re.findall(pattern, answer)

        report_ids = set()
        for match in matches:
            # Parse comma-separated report IDs
            ids = [x.strip() for x in match.split(",")]
            for rid in ids:
                if rid and rid != "+more":
                    report_ids.add(rid)

        # Create evidence entries
        for rid in sorted(report_ids, key=lambda x: int(x) if x.isdigit() else 0):
            evidence.append(
                {
                    "type": "community_report",
                    "report_id": rid,
                    "search_type": self.search_type,
                }
            )

        return evidence

    def query_sync(self, question: str) -> tuple[str, list[dict]]:
        """Synchronous wrapper for query().

        Args:
            question: The question to answer.

        Returns:
            Tuple of (answer, evidence).
        """
        return asyncio.run(self.query(question))
