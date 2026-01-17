"""
MODULE: Temporal Extractor
DESCRIPTION: Extracts document creation/publication date from first and last chunks.

Uses LLM to identify document dates from title pages, headers, footers, etc.
Ignores historical dates mentioned in the body text.
"""

import os
from datetime import date
from typing import List, Optional

from pydantic import BaseModel, Field

from src.util.llm_client import get_nano_llm

# Control verbose output
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"


def log(msg: str):
    """Print only if VERBOSE mode is enabled."""
    if VERBOSE:
        print(f"[TemporalExtractor] {msg}")


class DateExtraction(BaseModel):
    """Extracted date information from the document."""
    date_found: bool = Field(
        ...,
        description="Whether a valid document creation/publication date was found"
    )
    year: Optional[int] = Field(
        default=None,
        description="The year of the document (YYYY)"
    )
    month: Optional[int] = Field(
        default=None,
        description="The month of the document (1-12)"
    )
    day: Optional[int] = Field(
        default=None,
        description="The day of the document (1-31)"
    )
    reasoning: str = Field(
        default="",
        description="Quote the text that indicates this date"
    )


class TemporalExtractor:
    """
    Extracts document dates from first/last chunks of a document.

    Strategy:
    - Analyzes first 6 and last 6 text chunks
    - Looks for: title page dates, headers, footers, 'Published on', 'Date:', 'As of'
    - Ignores: historical dates in body text
    - Falls back to today's date if not found
    """

    def __init__(self):
        # Use cheap Gemini model for this simple task
        self.llm = get_nano_llm()
        self.structured_llm = self.llm.with_structured_output(DateExtraction)

    def extract_date(
        self,
        first_chunks: List[str],
        last_chunks: List[str],
        title: str
    ) -> Optional[str]:
        """
        Extract document date from first/last chunks.

        Args:
            first_chunks: First 6 text chunks from the document
            last_chunks: Last 6 text chunks from the document
            title: Document title for context

        Returns:
            ISO date string (YYYY-MM-DD) or None if not found.
        """
        # Combine chunks for context
        context_text = f"--- DOCUMENT TITLE: {title} ---\n\n"
        context_text += "--- BEGINNING OF DOCUMENT ---\n"
        context_text += "\n...\n".join(first_chunks) if first_chunks else "(No beginning chunks provided)"
        context_text += "\n\n--- END OF DOCUMENT ---\n"
        context_text += "\n...\n".join(last_chunks) if last_chunks else "(No ending chunks provided)"

        prompt = (
            "You are a specialist in temporal data extraction.\n"
            "Your TASK: Determine the Creation Date or Publication Date of this document.\n\n"
            f"{context_text}\n\n"
            "RULES:\n"
            "1. LOOK FOR: Title page dates, headers, footers, 'Published on', 'Date:', 'As of'.\n"
            "2. IGNORE: Historical dates mentioned in the text (e.g. 'In 1990, the company...'). "
            "ONLY extract the date of the document itself.\n"
            "3. IF NOT FOUND: Return date_found=False.\n"
            "4. IF PARTIAL: If valid Year/Month found but no Day, return them.\n"
        )

        try:
            result = self.structured_llm.invoke(prompt)

            if not result.date_found or not result.year:
                log(f"No date found for '{title}'.")
                return None

            # Construct date with fallbacks
            year = result.year
            month = result.month if result.month else 1
            day = result.day if result.day else 1

            # Validate date
            try:
                dt = date(year, month, day)
                log(f"Found {dt.isoformat()} for '{title}' (Reason: {result.reasoning[:50]}...)")
                return dt.isoformat()
            except ValueError:
                # Handle invalid dates (e.g. Feb 30) -> Fallback to today
                log(f"Invalid date {year}-{month}-{day}.")
                return None

        except Exception as e:
            log(f"Extraction failed: {e}")
            return None


# Convenience function for direct use
def extract_document_date(
    first_chunks: List[str],
    last_chunks: List[str],
    title: str
) -> Optional[str]:
    """
    Extract document date from first/last chunks.

    Args:
        first_chunks: First 6 text chunks from the document
        last_chunks: Last 6 text chunks from the document
        title: Document title for context

    Returns:
        ISO date string (YYYY-MM-DD) or None
    """
    extractor = TemporalExtractor()
    return extractor.extract_date(first_chunks, last_chunks, title)
