import pytest
from unittest.mock import MagicMock, patch
import datetime
from src.agents.temporal_extractor import TemporalExtractor, DateExtraction
from src.schemas.document_types import Chunk

@pytest.fixture
def clean_extractor():
    with patch('src.agents.temporal_extractor.get_llm') as mock_get_llm:
        # Mock the LLM and its structured output
        mock_llm_instance = MagicMock()
        mock_structured = MagicMock()
        mock_llm_instance.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm_instance
        
        extractor = TemporalExtractor()
        return extractor, mock_structured

def test_extract_date_success(clean_extractor):
    extractor, mock_structured = clean_extractor
    
    # Setup mock return
    mock_structured.invoke.return_value = DateExtraction(
        date_found=True,
        year=2023,
        month=10,
        day=5,
        reasoning="Found in header"
    )
    
    date = extractor.extract_date(["start"], ["end"], "doc_title")
    assert date == "2023-10-05"

def test_extract_date_partial_defaults_to_first(clean_extractor):
    extractor, mock_structured = clean_extractor
    
    # Returns Year/Month only
    mock_structured.invoke.return_value = DateExtraction(
        date_found=True,
        year=2023,
        month=11,
        day=None,
        reasoning="November 2023"
    )
    
    date = extractor.extract_date(["start"], ["end"], "doc_title")
    assert date == "2023-11-01"

def test_extract_date_not_found_defaults_to_today(clean_extractor):
    extractor, mock_structured = clean_extractor
    
    mock_structured.invoke.return_value = DateExtraction(
        date_found=False,
        year=None,
        month=None,
        day=None,
        reasoning="No date found"
    )
    
    date = extractor.extract_date(["start"], ["end"], "doc_title")
    assert date == datetime.date.today().isoformat()

def test_enrich_chunks(clean_extractor):
    extractor, mock_structured = clean_extractor
    
    mock_structured.invoke.return_value = DateExtraction(
        date_found=True,
        year=2020,
        month=1,
        day=1,
        reasoning="Found"
    )
    
    # Create chunks
    chunks = [
        Chunk(chunk_id="1", doc_id="d1", heading="h", body="text1", metadata={"doc_item_refs":[], "page_numbers":[]}),
        Chunk(chunk_id="2", doc_id="d1", heading="h", body="text2", metadata={"doc_item_refs":[], "page_numbers":[]})
    ]
    
    enriched = extractor.enrich_chunks(chunks, "My Doc")
    
    assert len(enriched) == 2
    assert enriched[0].metadata.get("doc_date") == "2020-01-01"
    assert enriched[1].metadata.get("doc_date") == "2020-01-01"
