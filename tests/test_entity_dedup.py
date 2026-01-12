#!/usr/bin/env python3
"""
Tests for Entity Deduplication System (src/util/entity_dedup.py)

Tests the hybrid approach: embedding similarity + LLM verification
"""

import os
import sys
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.util.entity_dedup import (
    UnionFind,
    DeferredDeduplicationManager,
    PendingEntity,
    DistinctEntity,
    DeduplicationResult,
)


# =============================================================================
# Test Data
# =============================================================================

# Test embeddings - simple 4D vectors for predictable similarity
EMBEDDING_APPLE_INC = [1.0, 0.0, 0.0, 0.0]
EMBEDDING_APPLE = [0.95, 0.05, 0.0, 0.0]  # Similar to Apple Inc
EMBEDDING_AAPL = [0.92, 0.08, 0.0, 0.0]  # Similar to Apple Inc
EMBEDDING_GOOGLE = [0.0, 1.0, 0.0, 0.0]  # Different from Apple
EMBEDDING_ALPHABET = [0.05, 0.95, 0.0, 0.0]  # Similar to Google
EMBEDDING_TIM_COOK = [0.0, 0.0, 1.0, 0.0]  # Person - different from companies
EMBEDDING_MICROSOFT = [0.0, 0.0, 0.0, 1.0]  # Different company


# =============================================================================
# Union-Find Tests
# =============================================================================

class TestUnionFind:
    """Test the Union-Find (Disjoint Set Union) data structure."""

    def test_initialization(self):
        """Test that Union-Find initializes correctly."""
        uf = UnionFind(5)
        assert len(uf.parent) == 5
        assert len(uf.rank) == 5
        # Each element is its own parent initially
        for i in range(5):
            assert uf.parent[i] == i
            assert uf.rank[i] == 0

    def test_find_without_union(self):
        """Test find returns self when no unions performed."""
        uf = UnionFind(5)
        for i in range(5):
            assert uf.find(i) == i

    def test_simple_union(self):
        """Test basic union of two elements."""
        uf = UnionFind(5)
        uf.union(0, 1)
        # After union, both should have same root
        assert uf.find(0) == uf.find(1)

    def test_transitive_union(self):
        """Test that unions are transitive: A-B, B-C => A-C."""
        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(1, 2)
        # All three should be in same component
        assert uf.find(0) == uf.find(1) == uf.find(2)

    def test_multiple_components(self):
        """Test maintaining separate components."""
        uf = UnionFind(6)
        # Component 1: 0, 1, 2
        uf.union(0, 1)
        uf.union(1, 2)
        # Component 2: 3, 4
        uf.union(3, 4)
        # 5 stays alone

        # Check components
        assert uf.find(0) == uf.find(1) == uf.find(2)
        assert uf.find(3) == uf.find(4)
        assert uf.find(5) == 5
        # Different components have different roots
        assert uf.find(0) != uf.find(3)
        assert uf.find(0) != uf.find(5)

    def test_get_components(self):
        """Test getting all components as a dict."""
        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(2, 3)

        components = uf.get_components()

        # Should have 3 components: {0,1}, {2,3}, {4}
        assert len(components) == 3

        # Check component sizes
        sizes = sorted([len(v) for v in components.values()])
        assert sizes == [1, 2, 2]

    def test_union_idempotent(self):
        """Test that unioning already-connected elements is safe."""
        uf = UnionFind(3)
        uf.union(0, 1)
        uf.union(0, 1)  # Repeat
        uf.union(1, 0)  # Reverse order

        components = uf.get_components()
        assert len(components) == 2  # {0,1} and {2}

    def test_path_compression(self):
        """Test that path compression works (find updates parent)."""
        uf = UnionFind(4)
        uf.union(0, 1)
        uf.union(1, 2)
        uf.union(2, 3)

        # After find, all should point closer to root
        root = uf.find(3)
        # Path compression should have updated parents
        assert uf.find(0) == root
        assert uf.find(1) == root
        assert uf.find(2) == root


# =============================================================================
# Similarity Matrix Tests
# =============================================================================

class TestSimilarityComputation:
    """Test cosine similarity matrix computation."""

    def setup_method(self):
        """Set up manager for each test."""
        DeferredDeduplicationManager.reset()
        self.manager = DeferredDeduplicationManager.get_instance()

    def teardown_method(self):
        """Clean up after each test."""
        DeferredDeduplicationManager.reset()

    def test_identical_vectors(self):
        """Test that identical vectors have similarity 1.0."""
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        similarity = self.manager._compute_similarity_matrix(embeddings)

        assert similarity[0, 0] == pytest.approx(1.0)
        assert similarity[1, 1] == pytest.approx(1.0)
        assert similarity[0, 1] == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """Test that orthogonal vectors have similarity 0.0."""
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        similarity = self.manager._compute_similarity_matrix(embeddings)

        # Diagonal should be 1.0
        assert similarity[0, 0] == pytest.approx(1.0)
        assert similarity[1, 1] == pytest.approx(1.0)
        assert similarity[2, 2] == pytest.approx(1.0)

        # Off-diagonal should be 0.0
        assert similarity[0, 1] == pytest.approx(0.0)
        assert similarity[0, 2] == pytest.approx(0.0)
        assert similarity[1, 2] == pytest.approx(0.0)

    def test_similar_vectors(self):
        """Test that similar vectors have high similarity."""
        embeddings = np.array([
            EMBEDDING_APPLE_INC,
            EMBEDDING_APPLE,
        ])
        similarity = self.manager._compute_similarity_matrix(embeddings)

        # Should be very similar (> 0.9)
        assert similarity[0, 1] > 0.9

    def test_different_vectors(self):
        """Test that different vectors have low similarity."""
        embeddings = np.array([
            EMBEDDING_APPLE_INC,
            EMBEDDING_GOOGLE,
        ])
        similarity = self.manager._compute_similarity_matrix(embeddings)

        # Should be very different (< 0.1)
        assert similarity[0, 1] < 0.1

    def test_symmetry(self):
        """Test that similarity matrix is symmetric."""
        embeddings = np.array([
            [1.0, 0.5, 0.3],
            [0.2, 0.8, 0.1],
            [0.5, 0.5, 0.5],
        ])
        similarity = self.manager._compute_similarity_matrix(embeddings)

        assert similarity[0, 1] == pytest.approx(similarity[1, 0])
        assert similarity[0, 2] == pytest.approx(similarity[2, 0])
        assert similarity[1, 2] == pytest.approx(similarity[2, 1])


# =============================================================================
# Connected Components Tests
# =============================================================================

class TestConnectedComponents:
    """Test building connected components from similarity graph."""

    def setup_method(self):
        """Set up manager for each test."""
        DeferredDeduplicationManager.reset()
        self.manager = DeferredDeduplicationManager.get_instance()

    def teardown_method(self):
        """Clean up after each test."""
        DeferredDeduplicationManager.reset()

    def test_single_entity(self):
        """Test with single entity returns single component."""
        entities = [
            PendingEntity("1", "Apple Inc.", "Entity", "Tech company", EMBEDDING_APPLE_INC, "test")
        ]
        components = self.manager._build_connected_components(entities, 0.7)

        assert len(components) == 1
        assert list(components.values())[0] == [0]

    def test_two_similar_entities(self):
        """Test that similar entities form one component."""
        entities = [
            PendingEntity("1", "Apple Inc.", "Entity", "Tech company", EMBEDDING_APPLE_INC, "test"),
            PendingEntity("2", "Apple", "Entity", "Tech company", EMBEDDING_APPLE, "test"),
        ]
        components = self.manager._build_connected_components(entities, 0.7)

        # Should be one component with both entities
        assert len(components) == 1
        component = list(components.values())[0]
        assert len(component) == 2

    def test_two_different_entities(self):
        """Test that different entities stay separate."""
        entities = [
            PendingEntity("1", "Apple Inc.", "Entity", "Tech company", EMBEDDING_APPLE_INC, "test"),
            PendingEntity("2", "Google", "Entity", "Tech company", EMBEDDING_GOOGLE, "test"),
        ]
        components = self.manager._build_connected_components(entities, 0.7)

        # Should be two separate components
        assert len(components) == 2

    def test_transitive_similarity(self):
        """Test that transitive similar entities form one component."""
        # A similar to B, B similar to C => A, B, C in one component
        entities = [
            PendingEntity("1", "Apple Inc.", "Entity", "Tech company", EMBEDDING_APPLE_INC, "test"),
            PendingEntity("2", "Apple", "Entity", "Tech company", EMBEDDING_APPLE, "test"),
            PendingEntity("3", "AAPL", "Entity", "Tech company", EMBEDDING_AAPL, "test"),
        ]
        components = self.manager._build_connected_components(entities, 0.7)

        # All similar to each other -> one component
        assert len(components) == 1
        component = list(components.values())[0]
        assert len(component) == 3

    def test_mixed_components(self):
        """Test mixture of connected and separate entities."""
        entities = [
            # Apple cluster
            PendingEntity("1", "Apple Inc.", "Entity", "Tech company", EMBEDDING_APPLE_INC, "test"),
            PendingEntity("2", "Apple", "Entity", "Tech company", EMBEDDING_APPLE, "test"),
            # Google cluster
            PendingEntity("3", "Google", "Entity", "Tech company", EMBEDDING_GOOGLE, "test"),
            PendingEntity("4", "Alphabet", "Entity", "Tech company", EMBEDDING_ALPHABET, "test"),
            # Singleton
            PendingEntity("5", "Tim Cook", "Entity", "Person", EMBEDDING_TIM_COOK, "test"),
        ]
        components = self.manager._build_connected_components(entities, 0.7)

        # Should have 3 components: Apple (2), Google (2), Tim Cook (1)
        assert len(components) == 3
        sizes = sorted([len(v) for v in components.values()])
        assert sizes == [1, 2, 2]

    def test_threshold_boundary(self):
        """Test threshold boundary behavior."""
        entities = [
            PendingEntity("1", "Apple Inc.", "Entity", "Tech company", EMBEDDING_APPLE_INC, "test"),
            PendingEntity("2", "Apple", "Entity", "Tech company", EMBEDDING_APPLE, "test"),
        ]

        # With high threshold, they should be separate
        components_high = self.manager._build_connected_components(entities, 0.99)
        assert len(components_high) == 2

        # With low threshold, they should be connected
        components_low = self.manager._build_connected_components(entities, 0.5)
        assert len(components_low) == 1


# =============================================================================
# Entity Registration Tests
# =============================================================================

class TestEntityRegistration:
    """Test entity registration and tracking."""

    def setup_method(self):
        """Set up manager for each test."""
        DeferredDeduplicationManager.reset()
        self.manager = DeferredDeduplicationManager.get_instance()

    def teardown_method(self):
        """Clean up after each test."""
        DeferredDeduplicationManager.reset()

    def test_register_single_entity(self):
        """Test registering a single entity."""
        uuid = self.manager.register_entity(
            uuid="test-1",
            name="Apple Inc.",
            node_type="Entity",
            summary="Technology company",
            embedding=EMBEDDING_APPLE_INC,
            group_id="test"
        )

        assert uuid == "test-1"
        assert "test-1" in self.manager._pending_entities
        entity = self.manager._pending_entities["test-1"]
        assert entity.name == "Apple Inc."
        assert entity.summary == "Technology company"

    def test_register_multiple_entities(self):
        """Test registering multiple entities."""
        self.manager.register_entity("1", "Apple", "Entity", "Tech", EMBEDDING_APPLE, "test")
        self.manager.register_entity("2", "Google", "Entity", "Tech", EMBEDDING_GOOGLE, "test")
        self.manager.register_entity("3", "Microsoft", "Entity", "Tech", EMBEDDING_MICROSOFT, "test")

        assert len(self.manager._pending_entities) == 3

    def test_name_normalization(self):
        """Test that entity names are normalized (title case)."""
        self.manager.register_entity("1", "  apple inc.  ", "Entity", "Tech", EMBEDDING_APPLE, "test")

        entity = self.manager._pending_entities["1"]
        assert entity.name == "Apple Inc."  # Stripped and title-cased


# =============================================================================
# UUID Remapping Tests
# =============================================================================

class TestUUIDRemapping:
    """Test UUID remapping after deduplication."""

    def setup_method(self):
        """Set up manager for each test."""
        DeferredDeduplicationManager.reset()
        self.manager = DeferredDeduplicationManager.get_instance()

    def teardown_method(self):
        """Clean up after each test."""
        DeferredDeduplicationManager.reset()

    def test_no_remapping_for_canonical(self):
        """Test that canonical UUID is not remapped."""
        # Manually set up a remap
        self.manager._uuid_remap["duplicate-1"] = "canonical-1"

        # Canonical should return itself
        assert self.manager.get_remapped_uuid("canonical-1") == "canonical-1"

    def test_remapping_for_duplicate(self):
        """Test that duplicate UUID is remapped to canonical."""
        self.manager._uuid_remap["duplicate-1"] = "canonical-1"

        assert self.manager.get_remapped_uuid("duplicate-1") == "canonical-1"

    def test_unknown_uuid_returns_self(self):
        """Test that unknown UUID returns itself."""
        assert self.manager.get_remapped_uuid("unknown-uuid") == "unknown-uuid"


# =============================================================================
# LLM Deduplication Tests (Mocked)
# =============================================================================

class TestLLMDeduplication:
    """Test LLM-based deduplication with mocked LLM."""

    def setup_method(self):
        """Set up manager for each test."""
        DeferredDeduplicationManager.reset()
        self.manager = DeferredDeduplicationManager.get_instance()

    def teardown_method(self):
        """Clean up after each test."""
        DeferredDeduplicationManager.reset()

    def test_single_entity_no_llm_call(self):
        """Test that single entity doesn't trigger LLM."""
        entities = [
            PendingEntity("1", "Apple Inc.", "Entity", "Tech company", EMBEDDING_APPLE_INC, "test")
        ]

        result = self.manager._llm_identify_distinct_entities(entities)

        assert len(result) == 1
        assert result[0].canonical_name == "Apple Inc."
        assert result[0].member_indices == [0]

    @patch.object(DeferredDeduplicationManager, '_get_llm')
    def test_llm_merges_duplicates(self, mock_get_llm):
        """Test that LLM can merge duplicate entities."""
        # Mock LLM response
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = DeduplicationResult(
            distinct_entities=[
                DistinctEntity(
                    canonical_name="Apple Inc.",
                    member_indices=[0, 1],
                    merged_summary="Apple Inc. is a technology company."
                )
            ]
        )
        mock_get_llm.return_value = mock_llm

        entities = [
            PendingEntity("1", "Apple Inc.", "Entity", "Tech company", EMBEDDING_APPLE_INC, "test"),
            PendingEntity("2", "Apple", "Entity", "Tech company", EMBEDDING_APPLE, "test"),
        ]

        result = self.manager._llm_dedupe_batch(entities)

        assert len(result) == 1
        assert result[0].canonical_name == "Apple Inc."
        assert set(result[0].member_indices) == {0, 1}

    @patch.object(DeferredDeduplicationManager, '_get_llm')
    def test_llm_keeps_distinct_entities(self, mock_get_llm):
        """Test that LLM keeps distinct entities separate."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = DeduplicationResult(
            distinct_entities=[
                DistinctEntity(
                    canonical_name="Apple Inc.",
                    member_indices=[0],
                    merged_summary="Apple Inc. is a technology company."
                ),
                DistinctEntity(
                    canonical_name="Tim Cook",
                    member_indices=[1],
                    merged_summary="Tim Cook is the CEO of Apple."
                )
            ]
        )
        mock_get_llm.return_value = mock_llm

        entities = [
            PendingEntity("1", "Apple Inc.", "Entity", "Tech company", EMBEDDING_APPLE_INC, "test"),
            PendingEntity("2", "Tim Cook", "Entity", "Person", EMBEDDING_TIM_COOK, "test"),
        ]

        result = self.manager._llm_dedupe_batch(entities)

        assert len(result) == 2
        names = {r.canonical_name for r in result}
        assert names == {"Apple Inc.", "Tim Cook"}


# =============================================================================
# Full Clustering Tests (Mocked)
# =============================================================================

class TestFullClustering:
    """Test the full cluster_and_remap flow with mocked LLM."""

    def setup_method(self):
        """Set up manager for each test."""
        DeferredDeduplicationManager.reset()
        self.manager = DeferredDeduplicationManager.get_instance()

    def teardown_method(self):
        """Clean up after each test."""
        DeferredDeduplicationManager.reset()

    def test_empty_entities(self):
        """Test with no entities."""
        stats = self.manager.cluster_and_remap(0.7)

        assert stats["components_found"] == 0
        assert stats["distinct_entities"] == 0

    def test_single_entity_no_clustering(self):
        """Test with single entity - no clustering needed."""
        self.manager.register_entity("1", "Apple", "Entity", "Tech", EMBEDDING_APPLE, "test")

        stats = self.manager.cluster_and_remap(0.7)

        assert stats["distinct_entities"] == 1
        assert stats["duplicates_merged"] == 0

    @patch.object(DeferredDeduplicationManager, '_llm_identify_distinct_entities')
    def test_clustering_with_duplicates(self, mock_llm_identify):
        """Test clustering merges duplicates via LLM."""
        # Mock LLM to say Apple Inc. and Apple are the same
        mock_llm_identify.return_value = [
            DistinctEntity(
                canonical_name="Apple Inc.",
                member_indices=[0, 1],
                merged_summary="Apple Inc. is a technology company."
            )
        ]

        self.manager.register_entity("1", "Apple Inc.", "Entity", "Tech", EMBEDDING_APPLE_INC, "test")
        self.manager.register_entity("2", "Apple", "Entity", "Tech", EMBEDDING_APPLE, "test")

        stats = self.manager.cluster_and_remap(0.7)

        # LLM was called for the component
        assert mock_llm_identify.called

        # Check stats
        assert stats["distinct_entities"] == 1
        assert stats["duplicates_merged"] == 1

        # Check UUID remapping - one should map to the other
        uuid1_canonical = self.manager.get_remapped_uuid("1")
        uuid2_canonical = self.manager.get_remapped_uuid("2")
        assert uuid1_canonical == uuid2_canonical

    @patch.object(DeferredDeduplicationManager, '_llm_identify_distinct_entities')
    def test_clustering_keeps_different_entities(self, mock_llm_identify):
        """Test that different entities stay separate."""
        # This mock won't be called because entities are in different components
        mock_llm_identify.return_value = []

        self.manager.register_entity("1", "Apple Inc.", "Entity", "Tech", EMBEDDING_APPLE_INC, "test")
        self.manager.register_entity("2", "Google", "Entity", "Tech", EMBEDDING_GOOGLE, "test")

        stats = self.manager.cluster_and_remap(0.7)

        # Both are singletons - no LLM call needed
        assert stats["distinct_entities"] == 2
        assert stats["duplicates_merged"] == 0

        # Each maps to itself
        assert self.manager.get_remapped_uuid("1") == "1"
        assert self.manager.get_remapped_uuid("2") == "2"


# =============================================================================
# Merge History Tests
# =============================================================================

class TestMergeHistory:
    """Test merge history tracking."""

    def setup_method(self):
        """Set up manager for each test."""
        DeferredDeduplicationManager.reset()
        self.manager = DeferredDeduplicationManager.get_instance()

    def teardown_method(self):
        """Clean up after each test."""
        DeferredDeduplicationManager.reset()

    @patch.object(DeferredDeduplicationManager, '_llm_identify_distinct_entities')
    def test_merge_history_recorded(self, mock_llm_identify):
        """Test that merges are recorded in history."""
        mock_llm_identify.return_value = [
            DistinctEntity(
                canonical_name="Apple Inc.",
                member_indices=[0, 1],
                merged_summary="Apple Inc. is a technology company."
            )
        ]

        self.manager.register_entity("1", "Apple Inc.", "Entity", "Tech company", EMBEDDING_APPLE_INC, "test")
        self.manager.register_entity("2", "Apple", "Entity", "Tech company", EMBEDDING_APPLE, "test")

        self.manager.cluster_and_remap(0.7)

        history = self.manager.get_merge_history()
        assert len(history) == 1

        merge = history[0]
        assert merge.canonical_name == "Apple Inc."
        assert len(merge.merged_uuids) == 2
        assert len(merge.merged_names) == 2


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        """Set up manager for each test."""
        DeferredDeduplicationManager.reset()
        self.manager = DeferredDeduplicationManager.get_instance()

    def teardown_method(self):
        """Clean up after each test."""
        DeferredDeduplicationManager.reset()

    def test_entities_without_embeddings_skipped(self):
        """Test that entities without embeddings are handled."""
        self.manager.register_entity("1", "Apple", "Entity", "Tech", None, "test")
        self.manager.register_entity("2", "Google", "Entity", "Tech", EMBEDDING_GOOGLE, "test")

        # Should not crash
        stats = self.manager.cluster_and_remap(0.7)
        # Only entity with embedding is processed
        assert stats["distinct_entities"] == 1

    def test_reset_clears_state(self):
        """Test that reset clears all state."""
        self.manager.register_entity("1", "Apple", "Entity", "Tech", EMBEDDING_APPLE, "test")
        self.manager._uuid_remap["test"] = "canonical"

        DeferredDeduplicationManager.reset()
        manager = DeferredDeduplicationManager.get_instance()

        assert len(manager._pending_entities) == 0
        assert len(manager._uuid_remap) == 0
        assert len(manager._merge_history) == 0

    def test_different_group_ids_separate(self):
        """Test that different group_ids don't cluster together."""
        # Same embedding but different groups
        self.manager.register_entity("1", "Apple", "Entity", "Tech", EMBEDDING_APPLE, "group1")
        self.manager.register_entity("2", "Apple", "Entity", "Tech", EMBEDDING_APPLE, "group2")

        stats = self.manager.cluster_and_remap(0.7)

        # Should be separate because different groups
        assert stats["distinct_entities"] == 2

    def test_different_node_types_separate(self):
        """Test that different node_types don't cluster together."""
        # Same embedding but different types
        self.manager.register_entity("1", "Apple", "Entity", "Tech", EMBEDDING_APPLE, "test")
        self.manager.register_entity("2", "Apple", "Topic", "Tech", EMBEDDING_APPLE, "test")

        stats = self.manager.cluster_and_remap(0.7)

        # Should be separate because different types
        assert stats["distinct_entities"] == 2


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
