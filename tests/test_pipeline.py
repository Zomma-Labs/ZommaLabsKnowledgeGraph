#!/usr/bin/env python3
"""
Integration Tests for Pipeline (src/pipeline.py)

Tests the end-to-end pipeline flow with mocked dependencies.
"""

import os
import sys
import json
import tempfile
import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import (
    create_document_node,
    create_episodic_node,
    create_entity_node,
    create_fact_node,
    create_relationship,
    _normalize_rel_type,
    _merge_summaries,
    extract_chunk,
    resolve_entities,
    resolve_topics,
    assemble_chunk,
    process_chunk,
    process_chunk_with_lookup,
)
from src.schemas.extraction import (
    ChainOfThoughtResult,
    EnumeratedEntity,
    ExtractedFact,
    EntityResolution,
)


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_neo4j():
    """Create a mock Neo4j client."""
    mock = MagicMock()
    mock.query.return_value = []
    return mock


@pytest.fixture
def mock_embeddings():
    """Create a mock embeddings client."""
    mock = MagicMock()
    mock.embed_query.return_value = [0.1] * 1024
    mock.embed_documents.return_value = [[0.1] * 1024]
    return mock


@pytest.fixture
def mock_llm():
    """Create a mock LLM client."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content="Merged summary")
    return mock


@pytest.fixture
def mock_entity_registry():
    """Create a mock EntityRegistry."""
    mock = MagicMock()
    mock.resolve.return_value = EntityResolution(
        uuid="test-uuid",
        canonical_name="Test Entity",
        is_new=True,
        updated_summary="Test summary",
        source_chunks=["chunk-1"],
        aliases=[]
    )
    return mock


@pytest.fixture
def mock_topic_librarian():
    """Create a mock TopicLibrarian."""
    mock = MagicMock()
    mock.batch_define_topics.return_value = {"Topic1": "Topic1: A topic"}
    mock.resolve_with_definition.return_value = {"label": "Topic1", "uri": "http://example.com"}
    return mock


# =============================================================================
# Neo4j Operation Tests
# =============================================================================

class TestNeo4jOperations:
    """Test Neo4j helper functions."""

    def test_create_document_node(self, mock_neo4j):
        """Test document node creation."""
        create_document_node(
            mock_neo4j,
            uuid="doc-1",
            name="Test Document",
            group_id="test",
            document_date=datetime(2024, 1, 1)
        )

        mock_neo4j.query.assert_called_once()
        call_args = mock_neo4j.query.call_args
        assert "MERGE" in call_args[0][0]
        assert "DocumentNode" in call_args[0][0]
        assert call_args[1]["uuid"] == "doc-1"

    def test_create_episodic_node(self, mock_neo4j):
        """Test episodic node creation."""
        create_episodic_node(
            mock_neo4j,
            uuid="ep-1",
            document_uuid="doc-1",
            content="Test content",
            header_path="Section > Subsection",
            group_id="test"
        )

        mock_neo4j.query.assert_called_once()
        call_args = mock_neo4j.query.call_args
        assert "EpisodicNode" in call_args[0][0]
        assert "CONTAINS_CHUNK" in call_args[0][0]

    def test_create_entity_node(self, mock_neo4j):
        """Test entity node creation."""
        create_entity_node(
            mock_neo4j,
            uuid="entity-1",
            name="Apple Inc.",
            summary="Tech company",
            group_id="test",
            embedding=[0.1] * 1024
        )

        mock_neo4j.query.assert_called_once()
        call_args = mock_neo4j.query.call_args
        assert "EntityNode" in call_args[0][0]
        assert call_args[1]["name"] == "Apple Inc."

    def test_create_fact_node(self, mock_neo4j):
        """Test fact node creation."""
        create_fact_node(
            mock_neo4j,
            uuid="fact-1",
            content="Apple acquired startup",
            group_id="test",
            embedding=[0.1] * 1024
        )

        mock_neo4j.query.assert_called_once()
        call_args = mock_neo4j.query.call_args
        assert "FactNode" in call_args[0][0]
        assert call_args[1]["content"] == "Apple acquired startup"

    def test_create_relationship(self, mock_neo4j):
        """Test relationship creation."""
        create_relationship(
            mock_neo4j,
            from_uuid="a",
            to_uuid="b",
            rel_type="ACQUIRED",
            properties={"date": "2024"}
        )

        mock_neo4j.query.assert_called_once()
        call_args = mock_neo4j.query.call_args
        assert "ACQUIRED" in call_args[0][0]
        assert "MERGE" in call_args[0][0]

    def test_create_relationship_no_properties(self, mock_neo4j):
        """Test relationship creation without properties."""
        create_relationship(
            mock_neo4j,
            from_uuid="a",
            to_uuid="b",
            rel_type="RELATED_TO"
        )

        mock_neo4j.query.assert_called_once()


# =============================================================================
# Relationship Normalization Tests
# =============================================================================

class TestRelationshipNormalization:
    """Test relationship type normalization."""

    def test_simple_relationship(self):
        """Test simple relationship normalization."""
        assert _normalize_rel_type("acquired") == "ACQUIRED"

    def test_multi_word_relationship(self):
        """Test multi-word relationship (max 3 words)."""
        assert _normalize_rel_type("merged into company") == "MERGED_INTO_COMPANY"

    def test_truncate_long_relationship(self):
        """Test that long relationships are truncated."""
        result = _normalize_rel_type("acquired majority stake in subsidiary company")
        words = result.split("_")
        assert len(words) <= 3

    def test_special_characters_removed(self):
        """Test that special characters are removed."""
        result = _normalize_rel_type("acquired (via merger)")
        assert "(" not in result
        assert ")" not in result

    def test_empty_relationship(self):
        """Test empty relationship defaults to RELATED_TO."""
        assert _normalize_rel_type("") == "RELATED_TO"

    def test_whitespace_handling(self):
        """Test whitespace handling."""
        result = _normalize_rel_type("  acquired   company  ")
        assert "__" not in result


# =============================================================================
# Summary Merging Tests
# =============================================================================

class TestSummaryMerging:
    """Test summary merging functionality."""

    def test_merge_empty_old(self, mock_llm):
        """Test merging with empty old summary."""
        result = _merge_summaries("", "New info", mock_llm)
        assert result == "New info"

    def test_merge_empty_new(self, mock_llm):
        """Test merging with empty new summary."""
        result = _merge_summaries("Old info", "", mock_llm)
        assert result == "Old info"

    def test_merge_both_empty(self, mock_llm):
        """Test merging with both empty."""
        result = _merge_summaries("", "", mock_llm)
        assert result == ""

    def test_merge_similar_summaries(self, mock_llm):
        """Test that similar summaries don't call LLM."""
        result = _merge_summaries(
            "Apple is a tech company",
            "tech company",
            mock_llm
        )
        # Should return old without LLM call
        assert result == "Apple is a tech company"
        mock_llm.invoke.assert_not_called()

    def test_merge_different_summaries(self, mock_llm):
        """Test merging different summaries calls LLM."""
        mock_llm.invoke.return_value = MagicMock(content="Merged summary text")

        result = _merge_summaries(
            "Apple makes computers",
            "Apple makes phones",
            mock_llm
        )

        mock_llm.invoke.assert_called_once()
        assert "Merged summary text" in result

    def test_merge_llm_error_fallback(self, mock_llm):
        """Test fallback when LLM fails."""
        mock_llm.invoke.side_effect = Exception("LLM error")

        result = _merge_summaries(
            "Old info",
            "New info",
            mock_llm
        )

        # Should concatenate
        assert "Old info" in result
        assert "New info" in result


# =============================================================================
# Extraction Tests
# =============================================================================

class TestExtraction:
    """Test chunk extraction."""

    @pytest.mark.asyncio
    async def test_extract_chunk_success(self):
        """Test successful chunk extraction."""
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = ChainOfThoughtResult(
            entities=[EnumeratedEntity(name="Apple", entity_type="Company", summary="Tech")],
            facts=[ExtractedFact(
                fact="Apple announced iPhone",
                subject="Apple",
                subject_type="Company",
                object="iPhone",
                object_type="Product",
                relationship="announced"
            )]
        )

        semaphore = __import__('asyncio').Semaphore(1)

        result = await extract_chunk(
            extractor=mock_extractor,
            chunk_idx=0,
            chunk_text="Apple announced iPhone",
            header_path="News",
            semaphore=semaphore
        )

        assert result["success"] is True
        assert result["chunk_idx"] == 0
        assert len(result["extraction"].entities) == 1
        assert len(result["extraction"].facts) == 1

    @pytest.mark.asyncio
    async def test_extract_chunk_error(self):
        """Test extraction error handling."""
        mock_extractor = MagicMock()
        mock_extractor.extract.side_effect = Exception("Extraction failed")

        semaphore = __import__('asyncio').Semaphore(1)

        result = await extract_chunk(
            extractor=mock_extractor,
            chunk_idx=0,
            chunk_text="Test",
            header_path="",
            semaphore=semaphore
        )

        assert result["success"] is False
        assert "error" in result


# =============================================================================
# Entity Resolution Tests
# =============================================================================

class TestEntityResolution:
    """Test entity resolution in pipeline context."""

    def test_resolve_entities_filters_topics(self, mock_entity_registry):
        """Test that Topic entities are filtered out."""
        extraction = ChainOfThoughtResult(
            entities=[
                EnumeratedEntity(name="Apple", entity_type="Company", summary="Tech"),
                EnumeratedEntity(name="Inflation", entity_type="Topic", summary="Economic"),
            ],
            facts=[]
        )

        result = resolve_entities(extraction, "ep-1", mock_entity_registry)

        # Should only resolve Apple, not Inflation
        assert "Apple" in result
        assert "Inflation" not in result

    def test_resolve_entities_returns_lookup(self, mock_entity_registry):
        """Test that resolution returns lookup dict."""
        extraction = ChainOfThoughtResult(
            entities=[
                EnumeratedEntity(name="Apple", entity_type="Company", summary="Tech"),
                EnumeratedEntity(name="Google", entity_type="Company", summary="Search"),
            ],
            facts=[]
        )

        mock_entity_registry.resolve.side_effect = [
            EntityResolution(uuid="1", canonical_name="Apple Inc.", is_new=True, updated_summary=""),
            EntityResolution(uuid="2", canonical_name="Alphabet", is_new=True, updated_summary=""),
        ]

        result = resolve_entities(extraction, "ep-1", mock_entity_registry)

        assert len(result) == 2
        assert result["Apple"].canonical_name == "Apple Inc."
        assert result["Google"].canonical_name == "Alphabet"


# =============================================================================
# Topic Resolution Tests
# =============================================================================

class TestTopicResolutionPipeline:
    """Test topic resolution in pipeline context."""

    def test_resolve_topics_extracts_from_facts(self, mock_topic_librarian):
        """Test that topics are extracted from facts."""
        extraction = ChainOfThoughtResult(
            entities=[],
            facts=[
                ExtractedFact(
                    fact="Test fact",
                    subject="A",
                    subject_type="Company",
                    object="B",
                    object_type="Company",
                    relationship="related",
                    topics=["Topic1", "Topic2"]
                ),
                ExtractedFact(
                    fact="Another fact",
                    subject="C",
                    subject_type="Company",
                    object="D",
                    object_type="Company",
                    relationship="related",
                    topics=["Topic1", "Topic3"]
                )
            ]
        )

        result = resolve_topics(extraction, "Test text", mock_topic_librarian)

        # batch_define_topics should be called with unique topics
        mock_topic_librarian.batch_define_topics.assert_called_once()
        call_args = mock_topic_librarian.batch_define_topics.call_args[0][0]
        # Should have 3 unique topics
        assert len(set(call_args)) == 3

    def test_resolve_topics_empty_facts(self, mock_topic_librarian):
        """Test resolution with no facts."""
        extraction = ChainOfThoughtResult(entities=[], facts=[])

        result = resolve_topics(extraction, "Test", mock_topic_librarian)

        assert result == {}


# =============================================================================
# Process Chunk Tests
# =============================================================================

class TestProcessChunk:
    """Test single chunk processing."""

    def test_process_chunk_failed_extraction(
        self,
        mock_neo4j,
        mock_embeddings,
        mock_llm,
        mock_entity_registry,
        mock_topic_librarian
    ):
        """Test processing failed extraction."""
        extraction_result = {
            "success": False,
            "chunk_idx": 0,
            "error": "Extraction failed",
            "extraction": ChainOfThoughtResult(entities=[], facts=[]),
            "chunk_text": "",
            "header_path": ""
        }

        result = process_chunk(
            extraction_result,
            document_uuid="doc-1",
            document_name="test",
            document_date=None,
            group_id="test",
            neo4j=mock_neo4j,
            embeddings=mock_embeddings,
            llm=mock_llm,
            entity_registry=mock_entity_registry,
            topic_librarian=mock_topic_librarian
        )

        assert result["success"] is False
        assert result["error"] == "Extraction failed"

    def test_process_chunk_no_facts(
        self,
        mock_neo4j,
        mock_embeddings,
        mock_llm,
        mock_entity_registry,
        mock_topic_librarian
    ):
        """Test processing extraction with no facts."""
        extraction_result = {
            "success": True,
            "chunk_idx": 0,
            "extraction": ChainOfThoughtResult(
                entities=[EnumeratedEntity(name="Apple", entity_type="Company", summary="Tech")],
                facts=[]
            ),
            "chunk_text": "Test",
            "header_path": ""
        }

        result = process_chunk(
            extraction_result,
            document_uuid="doc-1",
            document_name="test",
            document_date=None,
            group_id="test",
            neo4j=mock_neo4j,
            embeddings=mock_embeddings,
            llm=mock_llm,
            entity_registry=mock_entity_registry,
            topic_librarian=mock_topic_librarian
        )

        assert result["success"] is True
        assert result["facts"] == 0


# =============================================================================
# Process Chunk with Lookup Tests
# =============================================================================

class TestProcessChunkWithLookup:
    """Test chunk processing with pre-resolved entity lookup."""

    def test_process_with_lookup_uses_global_lookup(
        self,
        mock_neo4j,
        mock_embeddings,
        mock_llm,
        mock_topic_librarian
    ):
        """Test that global lookup is used for entities."""
        extraction_result = {
            "success": True,
            "chunk_idx": 0,
            "extraction": ChainOfThoughtResult(
                entities=[EnumeratedEntity(name="Apple", entity_type="Company", summary="Tech")],
                facts=[ExtractedFact(
                    fact="Apple announced",
                    subject="Apple",
                    subject_type="Company",
                    object="iPhone",
                    object_type="Product",
                    relationship="announced"
                )]
            ),
            "chunk_text": "Test",
            "header_path": ""
        }

        entity_lookup_global = {
            "Apple": EntityResolution(
                uuid="pre-resolved-uuid",
                canonical_name="Apple Inc.",
                is_new=False,
                updated_summary="Pre-resolved"
            )
        }

        result = process_chunk_with_lookup(
            extraction_result,
            document_uuid="doc-1",
            document_name="test",
            document_date=None,
            group_id="test",
            entity_lookup_global=entity_lookup_global,
            neo4j=mock_neo4j,
            embeddings=mock_embeddings,
            llm=mock_llm,
            topic_librarian=mock_topic_librarian
        )

        assert result["success"] is True

    def test_process_with_lookup_filters_topics(
        self,
        mock_neo4j,
        mock_embeddings,
        mock_llm,
        mock_topic_librarian
    ):
        """Test that topic entities are filtered when building lookup."""
        extraction_result = {
            "success": True,
            "chunk_idx": 0,
            "extraction": ChainOfThoughtResult(
                entities=[
                    EnumeratedEntity(name="Apple", entity_type="Company", summary="Tech"),
                    EnumeratedEntity(name="Inflation", entity_type="Topic", summary="Economic"),
                ],
                facts=[]
            ),
            "chunk_text": "Test",
            "header_path": ""
        }

        entity_lookup_global = {
            "Apple": EntityResolution(uuid="1", canonical_name="Apple", is_new=True, updated_summary=""),
            "Inflation": EntityResolution(uuid="2", canonical_name="Inflation", is_new=True, updated_summary=""),
        }

        result = process_chunk_with_lookup(
            extraction_result,
            document_uuid="doc-1",
            document_name="test",
            document_date=None,
            group_id="test",
            entity_lookup_global=entity_lookup_global,
            neo4j=mock_neo4j,
            embeddings=mock_embeddings,
            llm=mock_llm,
            topic_librarian=mock_topic_librarian
        )

        # Should succeed and filter topics
        assert result["success"] is True


# =============================================================================
# Integration Tests (File Processing)
# =============================================================================

class TestFileProcessing:
    """Test file-level processing."""

    def test_create_test_jsonl(self):
        """Helper to create test JSONL file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump({"body": "Apple announced iPhone 16", "metadata": {}}, f)
            f.write('\n')
            json.dump({"body": "Google released Pixel 9", "metadata": {}}, f)
            f.write('\n')
            return f.name

    def test_jsonl_parsing(self):
        """Test JSONL file parsing."""
        filepath = self.test_create_test_jsonl()

        chunks = []
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get("body"):
                        chunks.append(data)
                except json.JSONDecodeError:
                    continue

        assert len(chunks) == 2
        assert "Apple" in chunks[0]["body"]
        assert "Google" in chunks[1]["body"]

        # Cleanup
        os.unlink(filepath)


# =============================================================================
# Assembly Tests
# =============================================================================

class TestAssembly:
    """Test graph assembly."""

    def test_assemble_chunk_creates_nodes(
        self,
        mock_neo4j,
        mock_embeddings,
        mock_llm
    ):
        """Test that assembly creates appropriate nodes."""
        extraction = ChainOfThoughtResult(
            entities=[EnumeratedEntity(name="Apple", entity_type="Company", summary="Tech")],
            facts=[ExtractedFact(
                fact="Apple acquired startup",
                subject="Apple",
                subject_type="Company",
                object="Startup",
                object_type="Company",
                relationship="acquired",
                topics=["M&A"]
            )]
        )

        entity_lookup = {
            "Apple": EntityResolution(uuid="e1", canonical_name="Apple", is_new=True, updated_summary="Tech"),
            "Startup": EntityResolution(uuid="e2", canonical_name="Startup", is_new=True, updated_summary="Company"),
        }

        topic_lookup = {
            "M&A": {"label": "M&A", "uri": "http://example.com"}
        }

        counts = assemble_chunk(
            extraction=extraction,
            entity_lookup=entity_lookup,
            topic_lookup=topic_lookup,
            document_uuid="doc-1",
            episodic_uuid="ep-1",
            chunk_text="Test",
            header_path="",
            group_id="test",
            neo4j=mock_neo4j,
            embeddings=mock_embeddings,
            llm=mock_llm
        )

        # Should have created entities, facts, and relationships
        assert counts["entities"] == 2
        assert counts["facts"] == 1
        assert counts["relationships"] > 0


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
