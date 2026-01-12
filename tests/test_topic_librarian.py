#!/usr/bin/env python3
"""
Tests for Topic Librarian (src/agents/topic_librarian.py)

Tests topic resolution against the ontology using vector search + LLM verification.
"""

import os
import sys
import pytest
from unittest.mock import Mock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.topic_librarian import (
    TopicLibrarian,
    TopicResolutionResponse,
    TopicDefinition,
    BatchTopicDefinitions,
)


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_services():
    """Create mock services."""
    mock = MagicMock()
    mock.embeddings = MagicMock()
    mock.embeddings.embed_query.return_value = [0.1] * 1024
    return mock


@pytest.fixture
def mock_qdrant():
    """Create mock Qdrant client."""
    mock = MagicMock()
    mock.collection_exists.return_value = True
    mock.query_points.return_value = MagicMock(points=[])
    return mock


@pytest.fixture
def mock_llm():
    """Create mock LLM for topic resolution."""
    mock = MagicMock()
    mock.invoke.return_value = TopicResolutionResponse(selected_number=None)
    return mock


@pytest.fixture
def mock_definition_llm():
    """Create mock LLM for topic definitions."""
    mock = MagicMock()
    mock.invoke.return_value = BatchTopicDefinitions(definitions=[])
    return mock


@pytest.fixture
def librarian(mock_services, mock_qdrant, mock_llm, mock_definition_llm):
    """Create TopicLibrarian with mocked dependencies."""
    with patch('qdrant_client.QdrantClient', return_value=mock_qdrant):
        with patch('src.util.llm_client.get_nano_llm') as mock_get_llm:
            mock_get_llm.return_value.with_structured_output.side_effect = [
                mock_llm,
                mock_definition_llm
            ]
            lib = TopicLibrarian(services=mock_services)
            lib.llm = mock_llm
            lib.definition_llm = mock_definition_llm
            lib.client = mock_qdrant
            return lib


# =============================================================================
# Initialization Tests
# =============================================================================

class TestTopicLibrarianInit:
    """Test TopicLibrarian initialization."""

    def test_init_with_services(self, mock_services):
        """Test initialization with explicit services."""
        with patch('src.agents.topic_librarian.QdrantClient'):
            with patch('src.agents.topic_librarian.get_nano_llm') as mock_get_llm:
                mock_get_llm.return_value.with_structured_output.return_value = MagicMock()
                lib = TopicLibrarian(services=mock_services)

                assert lib.embeddings == mock_services.embeddings


# =============================================================================
# Vector Search Tests
# =============================================================================

class TestVectorSearch:
    """Test vector search functionality."""

    def test_vector_search_calls_qdrant(self, librarian, mock_qdrant):
        """Test that vector search queries Qdrant."""
        mock_qdrant.query_points.return_value = MagicMock(points=[])

        librarian._vector_search("Inflation", k=10)

        mock_qdrant.query_points.assert_called_once()
        call_kwargs = mock_qdrant.query_points.call_args[1]
        assert call_kwargs["collection_name"] == "topic_ontology"
        assert call_kwargs["limit"] == 10

    def test_vector_search_returns_candidates(self, librarian, mock_qdrant):
        """Test that vector search returns formatted candidates."""
        mock_point = MagicMock()
        mock_point.payload = {
            "uri": "http://example.com/inflation",
            "label": "Inflation",
            "definition": "Rise in prices",
            "synonyms": "price increase"
        }
        mock_point.score = 0.95

        mock_qdrant.query_points.return_value = MagicMock(points=[mock_point])

        candidates = librarian._vector_search("Inflation", k=10)

        assert len(candidates) == 1
        assert candidates[0]["label"] == "Inflation"
        assert candidates[0]["score"] == 0.95
        assert candidates[0]["definition"] == "Rise in prices"

    def test_vector_search_empty_collection(self, librarian, mock_qdrant):
        """Test handling of missing collection."""
        mock_qdrant.collection_exists.return_value = False

        candidates = librarian._vector_search("Inflation", k=10)

        assert candidates == []

    def test_vector_search_error_returns_empty(self, librarian, mock_qdrant):
        """Test that search errors return empty list."""
        mock_qdrant.query_points.side_effect = Exception("Search failed")

        candidates = librarian._vector_search("Inflation", k=10)

        assert candidates == []


# =============================================================================
# Topic Resolution Tests
# =============================================================================

class TestTopicResolution:
    """Test topic resolution functionality."""

    def test_resolve_empty_text(self, librarian):
        """Test resolving empty text returns None."""
        result = librarian.resolve("")
        assert result is None

        result = librarian.resolve("   ")
        assert result is None

    def test_resolve_no_candidates_below_threshold(self, librarian, mock_qdrant):
        """Test resolution when candidates below threshold."""
        mock_point = MagicMock()
        mock_point.payload = {"uri": "x", "label": "Test", "definition": "", "synonyms": ""}
        mock_point.score = 0.30  # Below default threshold of 0.40

        mock_qdrant.query_points.return_value = MagicMock(points=[mock_point])

        result = librarian.resolve("Some topic")

        assert result is None

    def test_resolve_llm_selects_candidate(self, librarian, mock_qdrant, mock_llm):
        """Test resolution when LLM selects a candidate."""
        mock_point = MagicMock()
        mock_point.payload = {
            "uri": "http://example.com/inflation",
            "label": "Inflation",
            "definition": "Rise in general price levels",
            "synonyms": "price increase"
        }
        mock_point.score = 0.90

        mock_qdrant.query_points.return_value = MagicMock(points=[mock_point])
        mock_llm.invoke.return_value = TopicResolutionResponse(selected_number=1)

        result = librarian.resolve("Inflation", context="Prices are rising")

        assert result is not None
        assert result["label"] == "Inflation"

    def test_resolve_llm_rejects_all(self, librarian, mock_qdrant, mock_llm):
        """Test resolution when LLM rejects all candidates."""
        mock_point = MagicMock()
        mock_point.payload = {"uri": "x", "label": "Test", "definition": "", "synonyms": ""}
        mock_point.score = 0.50

        mock_qdrant.query_points.return_value = MagicMock(points=[mock_point])
        mock_llm.invoke.return_value = TopicResolutionResponse(selected_number=None)

        result = librarian.resolve("Some topic")

        assert result is None

    def test_resolve_uses_context(self, librarian, mock_qdrant, mock_llm):
        """Test that context is passed to LLM verification."""
        mock_point = MagicMock()
        mock_point.payload = {"uri": "x", "label": "Inflation", "definition": "Price rise", "synonyms": ""}
        mock_point.score = 0.80

        mock_qdrant.query_points.return_value = MagicMock(points=[mock_point])
        mock_llm.invoke.return_value = TopicResolutionResponse(selected_number=1)

        librarian.resolve("Inflation", context="CPI increased by 3%")

        # Check that context was included in prompt
        call_args = mock_llm.invoke.call_args[0][0]
        assert "CPI increased by 3%" in call_args


# =============================================================================
# Resolution with Definition Tests
# =============================================================================

class TestResolutionWithDefinition:
    """Test resolution with enriched definitions."""

    def test_resolve_with_definition_uses_enriched_text(self, librarian, mock_qdrant, mock_llm):
        """Test that enriched text is used for embedding."""
        mock_point = MagicMock()
        mock_point.payload = {"uri": "x", "label": "M&A", "definition": "Mergers and Acquisitions", "synonyms": ""}
        mock_point.score = 0.85

        mock_qdrant.query_points.return_value = MagicMock(points=[mock_point])
        mock_llm.invoke.return_value = TopicResolutionResponse(selected_number=1)

        result = librarian.resolve_with_definition(
            text="M&A",
            enriched_text="M&A: Corporate mergers and acquisitions activity",
            context="Company acquired a competitor"
        )

        assert result is not None
        assert result["label"] == "M&A"

    def test_resolve_with_definition_empty_text(self, librarian):
        """Test that empty text returns None."""
        result = librarian.resolve_with_definition(
            text="",
            enriched_text="Something: Definition",
            context=""
        )

        assert result is None


# =============================================================================
# Batch Topic Definition Tests
# =============================================================================

class TestBatchTopicDefinition:
    """Test batch topic definition functionality."""

    def test_batch_define_empty_list(self, librarian):
        """Test defining empty topic list."""
        result = librarian.batch_define_topics([], "Some context")

        assert result == {}

    def test_batch_define_returns_definitions(self, librarian, mock_definition_llm):
        """Test that batch definition returns enriched topics."""
        mock_definition_llm.invoke.return_value = BatchTopicDefinitions(
            definitions=[
                TopicDefinition(topic="Inflation", definition="Rise in general price levels"),
                TopicDefinition(topic="M&A", definition="Mergers and acquisitions activity"),
            ]
        )

        result = librarian.batch_define_topics(
            ["Inflation", "M&A"],
            "Economic news context"
        )

        assert len(result) == 2
        assert "Inflation" in result
        assert "Inflation:" in result["Inflation"]
        assert "Rise in general price levels" in result["Inflation"]

    def test_batch_define_deduplicates_topics(self, librarian, mock_definition_llm):
        """Test that duplicate topics are deduplicated."""
        mock_definition_llm.invoke.return_value = BatchTopicDefinitions(
            definitions=[
                TopicDefinition(topic="Inflation", definition="Price rise"),
            ]
        )

        # Pass duplicates
        result = librarian.batch_define_topics(
            ["Inflation", "Inflation", "Inflation"],
            "Context"
        )

        # Should only have one entry
        assert len(result) == 1

    def test_batch_define_fallback_on_error(self, librarian, mock_definition_llm):
        """Test fallback to raw topics on error."""
        mock_definition_llm.invoke.side_effect = Exception("LLM failed")

        result = librarian.batch_define_topics(
            ["Inflation", "M&A"],
            "Context"
        )

        # Should return raw topics
        assert result["Inflation"] == "Inflation"
        assert result["M&A"] == "M&A"

    def test_batch_define_handles_missing_definitions(self, librarian, mock_definition_llm):
        """Test handling when LLM doesn't define all topics."""
        mock_definition_llm.invoke.return_value = BatchTopicDefinitions(
            definitions=[
                TopicDefinition(topic="Inflation", definition="Price rise"),
                # M&A missing
            ]
        )

        result = librarian.batch_define_topics(
            ["Inflation", "M&A"],
            "Context"
        )

        # Inflation should have definition
        assert "Price rise" in result["Inflation"]
        # M&A should fallback to raw
        assert result["M&A"] == "M&A"


# =============================================================================
# LLM Verification Tests
# =============================================================================

class TestLLMVerification:
    """Test LLM topic verification."""

    def test_llm_verify_empty_candidates(self, librarian):
        """Test verification with empty candidates."""
        result = librarian._llm_verify_topic("Test", "Context", [])

        assert result is None

    def test_llm_verify_formats_candidates(self, librarian, mock_llm):
        """Test that candidates are formatted correctly."""
        candidates = [
            {"label": "Inflation", "definition": "Price rise", "synonyms": "CPI increase", "score": 0.9},
            {"label": "Deflation", "definition": "Price fall", "synonyms": "CPI decrease", "score": 0.8},
        ]

        mock_llm.invoke.return_value = TopicResolutionResponse(selected_number=1)

        librarian._llm_verify_topic("Inflation", "Prices rising", candidates)

        call_args = mock_llm.invoke.call_args[0][0]
        assert "Inflation" in call_args
        assert "Deflation" in call_args
        assert "Price rise" in call_args

    def test_llm_verify_returns_selected_candidate(self, librarian, mock_llm):
        """Test that correct candidate is returned."""
        candidates = [
            {"label": "Topic1", "definition": "", "synonyms": "", "score": 0.9},
            {"label": "Topic2", "definition": "", "synonyms": "", "score": 0.8},
        ]

        mock_llm.invoke.return_value = TopicResolutionResponse(selected_number=2)

        result = librarian._llm_verify_topic("Test", "", candidates)

        assert result["label"] == "Topic2"

    def test_llm_verify_invalid_index_returns_none(self, librarian, mock_llm):
        """Test that invalid index returns None."""
        candidates = [
            {"label": "Topic1", "definition": "", "synonyms": "", "score": 0.9},
        ]

        mock_llm.invoke.return_value = TopicResolutionResponse(selected_number=99)

        result = librarian._llm_verify_topic("Test", "", candidates)

        assert result is None

    def test_llm_verify_error_returns_none(self, librarian, mock_llm):
        """Test that LLM error returns None."""
        candidates = [
            {"label": "Topic1", "definition": "", "synonyms": "", "score": 0.9},
        ]

        mock_llm.invoke.side_effect = Exception("LLM failed")

        result = librarian._llm_verify_topic("Test", "", candidates)

        assert result is None


# =============================================================================
# Batch Resolution Tests
# =============================================================================

class TestBatchResolution:
    """Test batch topic resolution."""

    def test_resolve_topics_empty_list(self, librarian):
        """Test resolving empty topic list."""
        result = librarian.resolve_topics([])

        assert result == []

    def test_resolve_topics_filters_none(self, librarian, mock_qdrant, mock_llm):
        """Test that None results are filtered out."""
        # First topic matches, second doesn't
        mock_point1 = MagicMock()
        mock_point1.payload = {"uri": "x", "label": "Inflation", "definition": "", "synonyms": ""}
        mock_point1.score = 0.90

        mock_point2 = MagicMock()
        mock_point2.payload = {"uri": "y", "label": "Unrelated", "definition": "", "synonyms": ""}
        mock_point2.score = 0.20  # Below threshold

        def mock_query_points(**kwargs):
            query = librarian.embeddings.embed_query.call_args
            # Return different results based on call count
            return MagicMock(points=[mock_point1])

        mock_qdrant.query_points.return_value = MagicMock(points=[mock_point1])
        mock_llm.invoke.side_effect = [
            TopicResolutionResponse(selected_number=1),
            TopicResolutionResponse(selected_number=None),
        ]

        result = librarian.resolve_topics(["Inflation", "RandomTopic"], context="Economic news")

        # Only Inflation should be in result
        assert "Inflation" in result

    def test_resolve_topics_deduplicates(self, librarian, mock_qdrant, mock_llm):
        """Test that resolved topics are deduplicated."""
        mock_point = MagicMock()
        mock_point.payload = {"uri": "x", "label": "Inflation", "definition": "", "synonyms": ""}
        mock_point.score = 0.90

        mock_qdrant.query_points.return_value = MagicMock(points=[mock_point])
        mock_llm.invoke.return_value = TopicResolutionResponse(selected_number=1)

        # Both map to same canonical
        result = librarian.resolve_topics(
            ["Inflation", "Price Increase", "CPI"],  # All might map to Inflation
            context=""
        )

        # Should only have one Inflation
        assert result.count("Inflation") <= 1


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_whitespace_only_text(self, librarian):
        """Test handling of whitespace-only text."""
        result = librarian.resolve("   \t\n   ")
        assert result is None

    def test_special_characters_in_topic(self, librarian, mock_qdrant, mock_llm):
        """Test handling of special characters."""
        mock_point = MagicMock()
        mock_point.payload = {"uri": "x", "label": "M&A", "definition": "", "synonyms": ""}
        mock_point.score = 0.90

        mock_qdrant.query_points.return_value = MagicMock(points=[mock_point])
        mock_llm.invoke.return_value = TopicResolutionResponse(selected_number=1)

        result = librarian.resolve("M&A Activity", context="")

        assert result is not None

    def test_long_topic_name(self, librarian, mock_qdrant, mock_llm):
        """Test handling of very long topic names."""
        long_topic = "A" * 500  # Very long topic name

        mock_qdrant.query_points.return_value = MagicMock(points=[])

        # Should not crash
        result = librarian.resolve(long_topic)

        assert result is None

    def test_custom_threshold(self, librarian, mock_qdrant, mock_llm):
        """Test using custom candidate threshold."""
        mock_point = MagicMock()
        mock_point.payload = {"uri": "x", "label": "Test", "definition": "", "synonyms": ""}
        mock_point.score = 0.50  # Between default 0.40 and high 0.80

        mock_qdrant.query_points.return_value = MagicMock(points=[mock_point])
        mock_llm.invoke.return_value = TopicResolutionResponse(selected_number=1)

        # Should match with default threshold (0.40)
        result_default = librarian.resolve("Test", candidate_threshold=0.40)
        assert result_default is not None

        # Should not match with high threshold
        mock_llm.invoke.return_value = TopicResolutionResponse(selected_number=1)
        result_high = librarian.resolve("Test", candidate_threshold=0.80)
        assert result_high is None


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
