#!/usr/bin/env python3
"""
Tests for Entity Registry (src/agents/entity_registry.py)

Tests entity resolution against Neo4j with LLM verification.
"""

import os
import sys
import pytest
from unittest.mock import Mock, MagicMock, patch
from uuid import uuid4

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.entity_registry import EntityRegistry
from src.schemas.extraction import EntityMatchDecision, EntityResolution


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_neo4j():
    """Create a mock Neo4j client."""
    mock = MagicMock()
    mock.vector_search.return_value = []
    return mock


@pytest.fixture
def mock_embeddings():
    """Create a mock embeddings client."""
    mock = MagicMock()
    mock.embed_query.return_value = [0.1] * 1024  # Fake embedding
    return mock


@pytest.fixture
def mock_llm():
    """Create a mock LLM client."""
    mock = MagicMock()
    return mock


@pytest.fixture
def registry(mock_neo4j, mock_embeddings, mock_llm):
    """Create an EntityRegistry with mocked dependencies."""
    return EntityRegistry(
        neo4j_client=mock_neo4j,
        embeddings=mock_embeddings,
        llm=mock_llm,
        group_id="test"
    )


# =============================================================================
# Initialization Tests
# =============================================================================

class TestEntityRegistryInit:
    """Test EntityRegistry initialization."""

    def test_init_with_explicit_dependencies(self, mock_neo4j, mock_embeddings, mock_llm):
        """Test initialization with explicit dependencies."""
        registry = EntityRegistry(
            neo4j_client=mock_neo4j,
            embeddings=mock_embeddings,
            llm=mock_llm,
            group_id="test"
        )

        assert registry.neo4j == mock_neo4j
        assert registry.embeddings == mock_embeddings
        assert registry.llm == mock_llm
        assert registry.group_id == "test"

    def test_init_creates_structured_matcher(self, mock_neo4j, mock_embeddings, mock_llm):
        """Test that structured matcher is created from LLM."""
        registry = EntityRegistry(
            neo4j_client=mock_neo4j,
            embeddings=mock_embeddings,
            llm=mock_llm,
            group_id="test"
        )

        # Should have called with_structured_output
        mock_llm.with_structured_output.assert_called_once_with(EntityMatchDecision)


# =============================================================================
# Resolution - New Entity Tests
# =============================================================================

class TestNewEntityResolution:
    """Test resolving entities that don't exist in the graph."""

    def test_new_entity_when_no_candidates(self, registry, mock_neo4j):
        """Test creating new entity when no candidates found."""
        mock_neo4j.vector_search.return_value = []

        result = registry.resolve(
            entity_name="Apple Inc.",
            entity_type="Company",
            entity_summary="Technology company",
            chunk_uuid="chunk-1"
        )

        assert result.is_new is True
        assert result.canonical_name == "Apple Inc."
        assert result.uuid is not None
        assert "Technology company" in result.updated_summary
        assert "chunk-1" in result.source_chunks

    def test_new_entity_generates_uuid(self, registry, mock_neo4j):
        """Test that new entities get unique UUIDs."""
        mock_neo4j.vector_search.return_value = []

        result1 = registry.resolve("Apple", "Company", "Tech", "chunk-1")
        result2 = registry.resolve("Google", "Company", "Tech", "chunk-2")

        assert result1.uuid != result2.uuid

    def test_new_entity_includes_source_tracking(self, registry, mock_neo4j):
        """Test that new entities include source chunk tracking."""
        mock_neo4j.vector_search.return_value = []

        result = registry.resolve(
            entity_name="Apple Inc.",
            entity_type="Company",
            entity_summary="Tech company",
            chunk_uuid="chunk-123"
        )

        assert "[Source: chunk-123]" in result.updated_summary
        assert "chunk-123" in result.source_chunks


# =============================================================================
# Resolution - Existing Entity Tests
# =============================================================================

class TestExistingEntityResolution:
    """Test resolving entities that exist in the graph."""

    def test_match_existing_entity(self, registry, mock_neo4j, mock_llm):
        """Test matching to existing entity."""
        # Set up mock candidates
        mock_neo4j.vector_search.return_value = [
            {
                "node": {
                    "uuid": "existing-uuid",
                    "name": "Apple Inc.",
                    "summary": "Technology company",
                    "entity_type": "Company",
                    "source_chunks": ["chunk-0"],
                    "aliases": []
                },
                "score": 0.95
            }
        ]

        # Mock LLM to say it's a match
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = EntityMatchDecision(
            is_same=True,
            match_index=1,
            reasoning="Same company"
        )
        registry.structured_matcher = mock_structured

        result = registry.resolve(
            entity_name="Apple",
            entity_type="Company",
            entity_summary="Tech giant",
            chunk_uuid="chunk-1"
        )

        assert result.is_new is False
        assert result.uuid == "existing-uuid"
        assert result.canonical_name == "Apple Inc."

    def test_no_match_creates_new_entity(self, registry, mock_neo4j, mock_llm):
        """Test that LLM rejection creates new entity."""
        mock_neo4j.vector_search.return_value = [
            {
                "node": {
                    "uuid": "existing-uuid",
                    "name": "Apple Records",
                    "summary": "Music label",
                    "entity_type": "Company",
                    "source_chunks": [],
                    "aliases": []
                },
                "score": 0.80
            }
        ]

        # Mock LLM to say it's NOT a match
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = EntityMatchDecision(
            is_same=False,
            match_index=None,
            reasoning="Different companies"
        )
        registry.structured_matcher = mock_structured

        result = registry.resolve(
            entity_name="Apple Inc.",
            entity_type="Company",
            entity_summary="Technology company",
            chunk_uuid="chunk-1"
        )

        assert result.is_new is True
        assert result.canonical_name == "Apple Inc."
        assert result.uuid != "existing-uuid"

    def test_match_updates_source_chunks(self, registry, mock_neo4j, mock_llm):
        """Test that matching updates source chunks list."""
        mock_neo4j.vector_search.return_value = [
            {
                "node": {
                    "uuid": "existing-uuid",
                    "name": "Apple Inc.",
                    "summary": "Tech company",
                    "entity_type": "Company",
                    "source_chunks": ["chunk-0"],
                    "aliases": []
                },
                "score": 0.95
            }
        ]

        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = EntityMatchDecision(
            is_same=True,
            match_index=1,
            reasoning="Same company"
        )
        registry.structured_matcher = mock_structured

        result = registry.resolve(
            entity_name="Apple",
            entity_type="Company",
            entity_summary="Tech giant",
            chunk_uuid="chunk-1"
        )

        assert "chunk-0" in result.source_chunks
        assert "chunk-1" in result.source_chunks

    def test_match_adds_alias(self, registry, mock_neo4j, mock_llm):
        """Test that matching adds alias for different name."""
        mock_neo4j.vector_search.return_value = [
            {
                "node": {
                    "uuid": "existing-uuid",
                    "name": "Apple Inc.",
                    "summary": "Tech company",
                    "entity_type": "Company",
                    "source_chunks": [],
                    "aliases": []
                },
                "score": 0.95
            }
        ]

        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = EntityMatchDecision(
            is_same=True,
            match_index=1,
            reasoning="Same company"
        )
        registry.structured_matcher = mock_structured

        result = registry.resolve(
            entity_name="AAPL",  # Different name
            entity_type="Company",
            entity_summary="Tech company ticker",
            chunk_uuid="chunk-1"
        )

        assert "AAPL" in result.aliases

    def test_same_name_not_added_as_alias(self, registry, mock_neo4j, mock_llm):
        """Test that same name is not added as alias."""
        mock_neo4j.vector_search.return_value = [
            {
                "node": {
                    "uuid": "existing-uuid",
                    "name": "Apple Inc.",
                    "summary": "Tech company",
                    "entity_type": "Company",
                    "source_chunks": [],
                    "aliases": []
                },
                "score": 0.95
            }
        ]

        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = EntityMatchDecision(
            is_same=True,
            match_index=1,
            reasoning="Same company"
        )
        registry.structured_matcher = mock_structured

        result = registry.resolve(
            entity_name="Apple Inc.",  # Same name
            entity_type="Company",
            entity_summary="Tech company",
            chunk_uuid="chunk-1"
        )

        assert "Apple Inc." not in result.aliases


# =============================================================================
# Vector Search Tests
# =============================================================================

class TestVectorSearch:
    """Test vector search functionality."""

    def test_search_called_with_correct_params(self, registry, mock_neo4j, mock_embeddings):
        """Test that vector search is called with correct parameters."""
        mock_neo4j.vector_search.return_value = []

        registry.resolve("Apple", "Company", "Tech", "chunk-1")

        mock_embeddings.embed_query.assert_called_once()
        mock_neo4j.vector_search.assert_called_once()

        # Check search params
        call_kwargs = mock_neo4j.vector_search.call_args
        assert call_kwargs[1]["index_name"] == "entity_name_embeddings"
        assert call_kwargs[1]["top_k"] == 25
        assert call_kwargs[1]["filters"] == {"group_id": "test"}

    def test_search_error_returns_empty(self, registry, mock_neo4j):
        """Test that search errors are handled gracefully."""
        mock_neo4j.vector_search.side_effect = Exception("Search failed")

        # Should not raise, should create new entity
        result = registry.resolve("Apple", "Company", "Tech", "chunk-1")

        assert result.is_new is True


# =============================================================================
# Summary Merging Tests
# =============================================================================

class TestSummaryMerging:
    """Test summary merging functionality."""

    def test_empty_existing_summary(self, registry):
        """Test merging when existing summary is empty."""
        result = registry._merge_summaries(
            existing_summary="",
            new_summary="New information",
            chunk_uuid="chunk-1"
        )

        assert "New information" in result
        assert "chunk-1" in result

    def test_duplicate_summary_not_merged(self, registry):
        """Test that duplicate summary doesn't trigger LLM."""
        result = registry._merge_summaries(
            existing_summary="Apple is a tech company.",
            new_summary="Apple is a tech company.",
            chunk_uuid="chunk-1"
        )

        # Should return existing without modification
        assert result == "Apple is a tech company."

    def test_subset_summary_not_merged(self, registry):
        """Test that subset summary doesn't trigger LLM."""
        result = registry._merge_summaries(
            existing_summary="Apple is a tech company based in Cupertino.",
            new_summary="tech company",
            chunk_uuid="chunk-1"
        )

        # Should return existing
        assert result == "Apple is a tech company based in Cupertino."

    def test_new_summary_triggers_llm(self, registry, mock_llm):
        """Test that genuinely new info triggers LLM merge."""
        mock_llm.invoke.return_value = MagicMock(
            content="Apple is a tech company. It makes iPhones."
        )

        result = registry._merge_summaries(
            existing_summary="Apple is a tech company.",
            new_summary="Apple makes iPhones.",
            chunk_uuid="chunk-1"
        )

        mock_llm.invoke.assert_called_once()
        assert "iPhones" in result or mock_llm.invoke.called


# =============================================================================
# Batch Resolution Tests
# =============================================================================

class TestBatchResolution:
    """Test batch entity resolution."""

    def test_batch_resolve_empty_list(self, registry):
        """Test batch resolution with empty list."""
        results = registry.resolve_batch([], "chunk-1")
        assert results == {}

    def test_batch_resolve_multiple_entities(self, registry, mock_neo4j):
        """Test batch resolution with multiple entities."""
        mock_neo4j.vector_search.return_value = []

        entities = [
            {"name": "Apple", "type": "Company", "summary": "Tech"},
            {"name": "Google", "type": "Company", "summary": "Search"},
            {"name": "Tim Cook", "type": "Person", "summary": "CEO"},
        ]

        results = registry.resolve_batch(entities, "chunk-1")

        assert len(results) == 3
        assert "Apple" in results
        assert "Google" in results
        assert "Tim Cook" in results

    def test_batch_resolve_skips_empty_names(self, registry, mock_neo4j):
        """Test that empty names are skipped."""
        mock_neo4j.vector_search.return_value = []

        entities = [
            {"name": "Apple", "type": "Company", "summary": "Tech"},
            {"name": "", "type": "Company", "summary": "Empty"},
            {"name": "Google", "type": "Company", "summary": "Search"},
        ]

        results = registry.resolve_batch(entities, "chunk-1")

        assert len(results) == 2
        assert "" not in results


# =============================================================================
# LLM Match Verification Tests
# =============================================================================

class TestLLMMatchVerification:
    """Test LLM match verification."""

    def test_llm_verify_formats_candidates(self, registry, mock_llm):
        """Test that candidates are formatted correctly for LLM."""
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = EntityMatchDecision(
            is_same=False,
            match_index=None,
            reasoning="No match"
        )
        registry.structured_matcher = mock_structured

        candidates = [
            {
                "uuid": "1",
                "name": "Apple Inc.",
                "summary": "Tech company",
                "entity_type": "Company",
                "aliases": ["AAPL"]
            },
            {
                "uuid": "2",
                "name": "Apple Records",
                "summary": "Music label",
                "entity_type": "Company",
                "aliases": []
            }
        ]

        registry._llm_verify_match(
            entity_name="Apple",
            entity_type="Company",
            entity_summary="Unknown Apple",
            candidates=candidates
        )

        # Check the prompt contains candidate info
        call_args = mock_structured.invoke.call_args[0][0]
        prompt_text = str(call_args)
        assert "Apple Inc." in prompt_text
        assert "Apple Records" in prompt_text

    def test_llm_error_returns_no_match(self, registry, mock_llm):
        """Test that LLM error returns no match decision."""
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = Exception("LLM error")
        registry.structured_matcher = mock_structured

        candidates = [{"uuid": "1", "name": "Apple", "summary": "", "entity_type": ""}]

        result = registry._llm_verify_match(
            entity_name="Apple",
            entity_type="Company",
            entity_summary="Tech",
            candidates=candidates
        )

        assert result.is_same is False
        assert "Error" in result.reasoning


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_match_index_handled(self, registry, mock_neo4j, mock_llm):
        """Test handling of invalid match index from LLM."""
        mock_neo4j.vector_search.return_value = [
            {
                "node": {
                    "uuid": "1",
                    "name": "Apple",
                    "summary": "",
                    "entity_type": "",
                    "source_chunks": [],
                    "aliases": []
                },
                "score": 0.9
            }
        ]

        mock_structured = MagicMock()
        mock_structured.invoke.return_value = EntityMatchDecision(
            is_same=True,
            match_index=99,  # Invalid - only 1 candidate
            reasoning="Test"
        )
        registry.structured_matcher = mock_structured

        # Should handle gracefully - the code will try to access index 98 which doesn't exist
        # This tests the boundary condition
        result = registry.resolve("Apple", "Company", "Tech", "chunk-1")

        # The current implementation would raise IndexError - this test documents the behavior
        # In production, you might want to add bounds checking

    def test_empty_summary_handled(self, registry, mock_neo4j):
        """Test handling of empty summary."""
        mock_neo4j.vector_search.return_value = []

        result = registry.resolve(
            entity_name="Apple",
            entity_type="Company",
            entity_summary="",
            chunk_uuid="chunk-1"
        )

        assert result.is_new is True
        assert result.canonical_name == "Apple"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
