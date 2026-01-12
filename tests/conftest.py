"""
Shared test fixtures for ZommaLabsKG test suite.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Test Embeddings (4D vectors for predictable similarity tests)
# =============================================================================

# Company embeddings (similar to each other)
EMBEDDING_APPLE_INC = [1.0, 0.0, 0.0, 0.0]
EMBEDDING_APPLE = [0.95, 0.05, 0.0, 0.0]
EMBEDDING_AAPL = [0.92, 0.08, 0.0, 0.0]

# Google/Alphabet cluster
EMBEDDING_GOOGLE = [0.0, 1.0, 0.0, 0.0]
EMBEDDING_ALPHABET = [0.05, 0.95, 0.0, 0.0]

# Person embedding (different from companies)
EMBEDDING_TIM_COOK = [0.0, 0.0, 1.0, 0.0]
EMBEDDING_SUNDAR_PICHAI = [0.0, 0.0, 0.95, 0.05]

# Other companies
EMBEDDING_MICROSOFT = [0.0, 0.0, 0.0, 1.0]
EMBEDDING_AMAZON = [0.0, 0.0, 0.05, 0.95]


# =============================================================================
# Sample Test Data
# =============================================================================

SAMPLE_FINANCIAL_TEXT = """
Apple Inc. announced today that CEO Tim Cook will present the company's new
iPhone 16 at the annual product launch event. The event will be held at
Apple Park in Cupertino, California. Analysts expect the new device to
boost Apple's revenue in Q4 2024.
"""

SAMPLE_HEADER_PATH = "Company News > Product Launches > Technology"


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_neo4j_client():
    """Create a mock Neo4j client."""
    mock = MagicMock()
    mock.query.return_value = []
    mock.vector_search.return_value = []
    return mock


@pytest.fixture
def mock_embeddings_client():
    """Create a mock embeddings client."""
    mock = MagicMock()
    mock.embed_query.return_value = [0.1] * 1024
    mock.embed_documents.return_value = [[0.1] * 1024]
    return mock


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content="Mocked response")
    mock.with_structured_output.return_value = MagicMock()
    return mock


@pytest.fixture
def mock_services(mock_neo4j_client, mock_embeddings_client, mock_llm_client):
    """Create mock services container."""
    mock = MagicMock()
    mock.neo4j = mock_neo4j_client
    mock.embeddings = mock_embeddings_client
    mock.llm = mock_llm_client
    return mock


# =============================================================================
# Test Entity Data
# =============================================================================

@pytest.fixture
def apple_entities():
    """Test entity data for Apple-related entities."""
    return {
        "Apple Inc.": {
            "embedding": EMBEDDING_APPLE_INC,
            "summary": "Apple Inc. is a multinational technology company headquartered in Cupertino, California.",
            "type": "Company"
        },
        "Apple": {
            "embedding": EMBEDDING_APPLE,
            "summary": "Apple is a technology company known for iPhones and Macs.",
            "type": "Company"
        },
        "AAPL": {
            "embedding": EMBEDDING_AAPL,
            "summary": "AAPL is the stock ticker symbol for Apple Inc.",
            "type": "Company"
        },
        "Tim Cook": {
            "embedding": EMBEDDING_TIM_COOK,
            "summary": "Tim Cook is the CEO of Apple Inc.",
            "type": "Person"
        }
    }


@pytest.fixture
def google_entities():
    """Test entity data for Google-related entities."""
    return {
        "Google": {
            "embedding": EMBEDDING_GOOGLE,
            "summary": "Google is a search engine and technology company.",
            "type": "Company"
        },
        "Alphabet": {
            "embedding": EMBEDDING_ALPHABET,
            "summary": "Alphabet is the parent company of Google.",
            "type": "Company"
        },
        "Sundar Pichai": {
            "embedding": EMBEDDING_SUNDAR_PICHAI,
            "summary": "Sundar Pichai is the CEO of Google and Alphabet.",
            "type": "Person"
        }
    }


# =============================================================================
# Test Fact Data
# =============================================================================

@pytest.fixture
def sample_facts():
    """Sample extracted facts for testing."""
    from src.schemas.extraction import ExtractedFact

    return [
        ExtractedFact(
            fact="Apple announced the iPhone 16 at their annual product launch event.",
            subject="Apple",
            subject_type="Company",
            object="iPhone 16",
            object_type="Product",
            relationship="announced",
            date_context="Q4 2024",
            topics=["Product Launch", "Technology"]
        ),
        ExtractedFact(
            fact="Tim Cook presented the new device at Apple Park.",
            subject="Tim Cook",
            subject_type="Person",
            object="Apple Park",
            object_type="Location",
            relationship="presented at",
            topics=["Corporate Events"]
        ),
        ExtractedFact(
            fact="Google acquired a cybersecurity startup for $500 million.",
            subject="Google",
            subject_type="Company",
            object="Cybersecurity Startup",
            object_type="Company",
            relationship="acquired",
            date_context="2024",
            topics=["M&A", "Cybersecurity"]
        )
    ]


# =============================================================================
# Ground Truth Data for Deduplication Testing
# =============================================================================

@pytest.fixture
def dedup_ground_truth():
    """Ground truth for entity deduplication testing."""
    return {
        # Apple cluster - should merge
        "Apple Inc.": "Apple Inc.",
        "Apple": "Apple Inc.",
        "AAPL": "Apple Inc.",

        # Google cluster - should merge
        "Google": "Google",
        "Alphabet": "Google",  # Debatable - could be separate

        # People - should NOT merge with companies
        "Tim Cook": "Tim Cook",
        "Sundar Pichai": "Sundar Pichai",

        # Different companies - should NOT merge
        "Microsoft": "Microsoft",
        "Amazon": "Amazon",
    }


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_llm: marks tests that require actual LLM calls"
    )
    config.addinivalue_line(
        "markers", "requires_neo4j: marks tests that require Neo4j connection"
    )
