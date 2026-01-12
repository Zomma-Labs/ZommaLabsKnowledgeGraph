#!/usr/bin/env python3
"""
Tests for Extractor V2 (src/agents/extractor_v2.py)

Tests chain-of-thought extraction with entity enumeration.
"""

import os
import sys
import pytest
from unittest.mock import Mock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.extractor_v2 import ExtractorV2
from src.schemas.extraction import (
    ChainOfThoughtResult,
    EnumeratedEntity,
    ExtractedFact,
    CritiqueResult,
)


# =============================================================================
# Test Data
# =============================================================================

SAMPLE_TEXT = """
Apple Inc. announced today that CEO Tim Cook will present the company's new
iPhone 16 at the annual product launch event. The event will be held at
Apple Park in Cupertino, California. Analysts expect the new device to
boost Apple's revenue in Q4 2024.
"""

SAMPLE_HEADER = "Company News > Product Launches"


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_llm():
    """Create a mock LLM client."""
    mock = MagicMock()
    return mock


@pytest.fixture
def extractor(mock_llm):
    """Create an ExtractorV2 with mocked LLM."""
    return ExtractorV2(llm=mock_llm, max_retries=2)


# =============================================================================
# Initialization Tests
# =============================================================================

class TestExtractorInit:
    """Test ExtractorV2 initialization."""

    def test_init_with_explicit_llm(self, mock_llm):
        """Test initialization with explicit LLM."""
        extractor = ExtractorV2(llm=mock_llm)

        assert extractor.llm == mock_llm
        assert extractor.max_retries == 2  # Default

    def test_init_custom_retries(self, mock_llm):
        """Test initialization with custom retries."""
        extractor = ExtractorV2(llm=mock_llm, max_retries=5)

        assert extractor.max_retries == 5

    def test_init_creates_structured_extractors(self, mock_llm):
        """Test that structured extractors are created."""
        extractor = ExtractorV2(llm=mock_llm)

        # Should have called with_structured_output twice
        assert mock_llm.with_structured_output.call_count == 2


# =============================================================================
# Entity Enumeration Tests
# =============================================================================

class TestEntityEnumeration:
    """Test entity enumeration functionality."""

    def test_extract_returns_entities(self, extractor, mock_llm):
        """Test that extraction returns enumerated entities."""
        mock_result = ChainOfThoughtResult(
            entities=[
                EnumeratedEntity(name="Apple Inc.", entity_type="Company", summary="Tech company"),
                EnumeratedEntity(name="Tim Cook", entity_type="Person", summary="CEO of Apple"),
            ],
            facts=[]
        )

        mock_structured = MagicMock()
        mock_structured.invoke.return_value = {"parsed": mock_result, "parsing_error": None}
        extractor.structured_extractor = mock_structured

        # Mock critic to approve
        mock_critic = MagicMock()
        mock_critic.invoke.return_value = {"parsed": CritiqueResult(is_approved=True), "parsing_error": None}
        extractor.structured_critic = mock_critic

        result = extractor.extract(SAMPLE_TEXT, SAMPLE_HEADER)

        assert len(result.entities) == 2
        assert result.entities[0].name == "Apple Inc."
        assert result.entities[1].name == "Tim Cook"

    def test_empty_text_returns_empty_result(self, extractor, mock_llm):
        """Test that empty text returns empty result."""
        mock_result = ChainOfThoughtResult(entities=[], facts=[])

        mock_structured = MagicMock()
        mock_structured.invoke.return_value = {"parsed": mock_result, "parsing_error": None}
        extractor.structured_extractor = mock_structured

        result = extractor.extract("", "")

        assert len(result.entities) == 0
        assert len(result.facts) == 0


# =============================================================================
# Fact Extraction Tests
# =============================================================================

class TestFactExtraction:
    """Test fact extraction functionality."""

    def test_extract_returns_facts(self, extractor, mock_llm):
        """Test that extraction returns facts."""
        mock_result = ChainOfThoughtResult(
            entities=[
                EnumeratedEntity(name="Apple", entity_type="Company", summary="Tech"),
                EnumeratedEntity(name="iPhone 16", entity_type="Product", summary="Phone"),
            ],
            facts=[
                ExtractedFact(
                    fact="Apple announced the iPhone 16.",
                    subject="Apple",
                    subject_type="Company",
                    object="iPhone 16",
                    object_type="Product",
                    relationship="announced",
                    topics=["Product Launch"]
                )
            ]
        )

        mock_structured = MagicMock()
        mock_structured.invoke.return_value = {"parsed": mock_result, "parsing_error": None}
        extractor.structured_extractor = mock_structured

        mock_critic = MagicMock()
        mock_critic.invoke.return_value = {"parsed": CritiqueResult(is_approved=True), "parsing_error": None}
        extractor.structured_critic = mock_critic

        result = extractor.extract(SAMPLE_TEXT, SAMPLE_HEADER)

        assert len(result.facts) == 1
        assert result.facts[0].subject == "Apple"
        assert result.facts[0].object == "iPhone 16"
        assert result.facts[0].relationship == "announced"

    def test_facts_include_date_context(self, extractor, mock_llm):
        """Test that facts include date context."""
        mock_result = ChainOfThoughtResult(
            entities=[EnumeratedEntity(name="Apple", entity_type="Company", summary="Tech")],
            facts=[
                ExtractedFact(
                    fact="Apple reported Q4 2024 earnings.",
                    subject="Apple",
                    subject_type="Company",
                    object="Earnings",
                    object_type="Topic",
                    relationship="reported",
                    date_context="Q4 2024"
                )
            ]
        )

        mock_structured = MagicMock()
        mock_structured.invoke.return_value = {"parsed": mock_result, "parsing_error": None}
        extractor.structured_extractor = mock_structured

        mock_critic = MagicMock()
        mock_critic.invoke.return_value = {"parsed": CritiqueResult(is_approved=True), "parsing_error": None}
        extractor.structured_critic = mock_critic

        result = extractor.extract("Apple Q4 2024 earnings", "")

        assert result.facts[0].date_context == "Q4 2024"

    def test_facts_include_topics(self, extractor, mock_llm):
        """Test that facts include topics."""
        mock_result = ChainOfThoughtResult(
            entities=[EnumeratedEntity(name="Apple", entity_type="Company", summary="Tech")],
            facts=[
                ExtractedFact(
                    fact="Apple acquired a startup.",
                    subject="Apple",
                    subject_type="Company",
                    object="Startup",
                    object_type="Company",
                    relationship="acquired",
                    topics=["M&A", "Technology"]
                )
            ]
        )

        mock_structured = MagicMock()
        mock_structured.invoke.return_value = {"parsed": mock_result, "parsing_error": None}
        extractor.structured_extractor = mock_structured

        mock_critic = MagicMock()
        mock_critic.invoke.return_value = {"parsed": CritiqueResult(is_approved=True), "parsing_error": None}
        extractor.structured_critic = mock_critic

        result = extractor.extract("Apple acquisition", "")

        assert "M&A" in result.facts[0].topics


# =============================================================================
# Critique Tests
# =============================================================================

class TestCritique:
    """Test the critique (reflexion) functionality."""

    def test_approved_extraction_not_reextracted(self, extractor, mock_llm):
        """Test that approved extractions don't trigger re-extraction."""
        mock_result = ChainOfThoughtResult(
            entities=[EnumeratedEntity(name="Apple", entity_type="Company", summary="Tech")],
            facts=[ExtractedFact(
                fact="Test", subject="Apple", subject_type="Company",
                object="Test", object_type="Topic", relationship="related"
            )]
        )

        mock_structured = MagicMock()
        mock_structured.invoke.return_value = {"parsed": mock_result, "parsing_error": None}
        extractor.structured_extractor = mock_structured

        mock_critic = MagicMock()
        mock_critic.invoke.return_value = {"parsed": CritiqueResult(is_approved=True), "parsing_error": None}
        extractor.structured_critic = mock_critic

        result = extractor.extract(SAMPLE_TEXT, SAMPLE_HEADER)

        # Extractor should be called once (initial), critic once
        assert mock_structured.invoke.call_count == 1

    def test_rejected_extraction_triggers_reextract(self, extractor, mock_llm):
        """Test that rejected extractions trigger re-extraction."""
        mock_result = ChainOfThoughtResult(
            entities=[EnumeratedEntity(name="Apple", entity_type="Company", summary="Tech")],
            facts=[ExtractedFact(
                fact="Test", subject="Apple", subject_type="Company",
                object="Test", object_type="Topic", relationship="related"
            )]
        )

        mock_structured = MagicMock()
        mock_structured.invoke.return_value = {"parsed": mock_result, "parsing_error": None}
        extractor.structured_extractor = mock_structured

        # First critique rejects, not called again after reextract
        mock_critic = MagicMock()
        mock_critic.invoke.return_value = {
            "parsed": CritiqueResult(
                is_approved=False,
                critique="Missing entity: Tim Cook",
                missed_facts=["Tim Cook is CEO of Apple"],
                corrections=[]
            ),
            "parsing_error": None
        }
        extractor.structured_critic = mock_critic

        result = extractor.extract(SAMPLE_TEXT, SAMPLE_HEADER)

        # Should have initial extract + reextract = 2 calls
        assert mock_structured.invoke.call_count == 2

    def test_empty_extraction_skips_critique(self, extractor, mock_llm):
        """Test that empty extraction skips critique step."""
        mock_result = ChainOfThoughtResult(entities=[], facts=[])

        mock_structured = MagicMock()
        mock_structured.invoke.return_value = {"parsed": mock_result, "parsing_error": None}
        extractor.structured_extractor = mock_structured

        mock_critic = MagicMock()
        extractor.structured_critic = mock_critic

        result = extractor.extract("", "")

        # Critic should not be called for empty extraction
        mock_critic.invoke.assert_not_called()


# =============================================================================
# Re-extraction Tests
# =============================================================================

class TestReextraction:
    """Test the re-extraction functionality."""

    def test_reextract_includes_corrections(self, extractor, mock_llm):
        """Test that re-extraction includes corrections from critique."""
        initial_result = ChainOfThoughtResult(
            entities=[EnumeratedEntity(name="Apple", entity_type="Company", summary="Tech")],
            facts=[ExtractedFact(
                fact="Test", subject="Apple", subject_type="Company",
                object="Test", object_type="Topic", relationship="related"
            )]
        )

        improved_result = ChainOfThoughtResult(
            entities=[
                EnumeratedEntity(name="Apple", entity_type="Company", summary="Tech"),
                EnumeratedEntity(name="Tim Cook", entity_type="Person", summary="CEO"),
            ],
            facts=[ExtractedFact(
                fact="Tim Cook is CEO of Apple", subject="Tim Cook", subject_type="Person",
                object="Apple", object_type="Company", relationship="leads"
            )]
        )

        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = [
            {"parsed": initial_result, "parsing_error": None},
            {"parsed": improved_result, "parsing_error": None},
        ]
        extractor.structured_extractor = mock_structured

        mock_critic = MagicMock()
        mock_critic.invoke.return_value = {
            "parsed": CritiqueResult(
                is_approved=False,
                critique="Missing Tim Cook",
                missed_facts=["Tim Cook is CEO"],
                corrections=["Add Tim Cook entity"]
            ),
            "parsing_error": None
        }
        extractor.structured_critic = mock_critic

        result = extractor.extract(SAMPLE_TEXT, SAMPLE_HEADER)

        # Should return improved result with Tim Cook
        assert len(result.entities) == 2
        entity_names = [e.name for e in result.entities]
        assert "Tim Cook" in entity_names


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Test error handling during extraction."""

    def test_parsing_error_retries(self, extractor, mock_llm):
        """Test that parsing errors trigger retries."""
        mock_result = ChainOfThoughtResult(
            entities=[EnumeratedEntity(name="Apple", entity_type="Company", summary="Tech")],
            facts=[]
        )

        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = [
            {"parsed": None, "parsing_error": "Invalid JSON"},
            {"parsed": mock_result, "parsing_error": None},
        ]
        extractor.structured_extractor = mock_structured

        mock_critic = MagicMock()
        mock_critic.invoke.return_value = {"parsed": CritiqueResult(is_approved=True), "parsing_error": None}
        extractor.structured_critic = mock_critic

        result = extractor.extract(SAMPLE_TEXT, SAMPLE_HEADER)

        # Should succeed on second try
        assert len(result.entities) == 1

    def test_all_retries_fail_returns_empty(self, extractor, mock_llm):
        """Test that exhausted retries return empty result."""
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = {"parsed": None, "parsing_error": "Always fails"}
        extractor.structured_extractor = mock_structured

        result = extractor.extract(SAMPLE_TEXT, SAMPLE_HEADER)

        assert len(result.entities) == 0
        assert len(result.facts) == 0

    def test_exception_during_extraction_retries(self, extractor, mock_llm):
        """Test that exceptions during extraction trigger retries."""
        mock_result = ChainOfThoughtResult(
            entities=[EnumeratedEntity(name="Apple", entity_type="Company", summary="Tech")],
            facts=[]
        )

        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = [
            Exception("Network error"),
            {"parsed": mock_result, "parsing_error": None},
        ]
        extractor.structured_extractor = mock_structured

        mock_critic = MagicMock()
        mock_critic.invoke.return_value = {"parsed": CritiqueResult(is_approved=True), "parsing_error": None}
        extractor.structured_critic = mock_critic

        result = extractor.extract(SAMPLE_TEXT, SAMPLE_HEADER)

        # Should succeed after retry
        assert len(result.entities) == 1

    def test_critique_error_assumes_approved(self, extractor, mock_llm):
        """Test that critique error assumes approval."""
        mock_result = ChainOfThoughtResult(
            entities=[EnumeratedEntity(name="Apple", entity_type="Company", summary="Tech")],
            facts=[ExtractedFact(
                fact="Test", subject="Apple", subject_type="Company",
                object="Test", object_type="Topic", relationship="related"
            )]
        )

        mock_structured = MagicMock()
        mock_structured.invoke.return_value = {"parsed": mock_result, "parsing_error": None}
        extractor.structured_extractor = mock_structured

        mock_critic = MagicMock()
        mock_critic.invoke.side_effect = Exception("Critique failed")
        extractor.structured_critic = mock_critic

        result = extractor.extract(SAMPLE_TEXT, SAMPLE_HEADER)

        # Should return initial result (no re-extraction)
        assert len(result.entities) == 1
        # Extractor only called once
        assert mock_structured.invoke.call_count == 1


# =============================================================================
# Format Helper Tests
# =============================================================================

class TestFormatHelpers:
    """Test formatting helper methods."""

    def test_format_entities_for_review(self, extractor):
        """Test entity formatting for review."""
        entities = [
            EnumeratedEntity(name="Apple Inc.", entity_type="Company", summary="Tech company"),
            EnumeratedEntity(name="Tim Cook", entity_type="Person", summary="CEO of Apple"),
        ]

        result = extractor._format_entities_for_review(entities)

        assert "1. Apple Inc. (Company)" in result
        assert "2. Tim Cook (Person)" in result
        assert "Tech company" in result
        assert "CEO of Apple" in result

    def test_format_entities_empty(self, extractor):
        """Test formatting empty entities list."""
        result = extractor._format_entities_for_review([])

        assert "No entities enumerated" in result

    def test_format_facts_for_review(self, extractor):
        """Test fact formatting for review."""
        facts = [
            ExtractedFact(
                fact="Apple announced iPhone 16.",
                subject="Apple",
                subject_type="Company",
                object="iPhone 16",
                object_type="Product",
                relationship="announced",
                date_context="2024",
                topics=["Product Launch"]
            )
        ]

        result = extractor._format_facts_for_review(facts)

        assert "Apple announced iPhone 16" in result
        assert "Subject: Apple" in result
        assert "Object: iPhone 16" in result
        assert "Relationship: announced" in result
        assert "Date: 2024" in result
        assert "Topics: Product Launch" in result

    def test_format_facts_empty(self, extractor):
        """Test formatting empty facts list."""
        result = extractor._format_facts_for_review([])

        assert "No facts extracted" in result


# =============================================================================
# Compatibility Tests
# =============================================================================

class TestCompatibility:
    """Test compatibility methods."""

    def test_extract_to_result_returns_extraction_result(self, extractor, mock_llm):
        """Test extract_to_result returns ExtractionResult."""
        mock_result = ChainOfThoughtResult(
            entities=[EnumeratedEntity(name="Apple", entity_type="Company", summary="Tech")],
            facts=[ExtractedFact(
                fact="Test", subject="Apple", subject_type="Company",
                object="Test", object_type="Topic", relationship="related"
            )]
        )

        mock_structured = MagicMock()
        mock_structured.invoke.return_value = {"parsed": mock_result, "parsing_error": None}
        extractor.structured_extractor = mock_structured

        mock_critic = MagicMock()
        mock_critic.invoke.return_value = {"parsed": CritiqueResult(is_approved=True), "parsing_error": None}
        extractor.structured_critic = mock_critic

        from src.schemas.extraction import ExtractionResult
        result = extractor.extract_to_result(SAMPLE_TEXT, SAMPLE_HEADER)

        # Should be ExtractionResult (not ChainOfThoughtResult)
        assert isinstance(result, ExtractionResult)
        assert len(result.facts) == 1


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
