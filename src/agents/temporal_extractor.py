import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from src.util.llm_client import get_llm
from src.schemas.document_types import Chunk

class DateExtraction(BaseModel):
    """Extracted date information from the document."""
    date_found: bool = Field(description="Whether a valid document creation/publication date was found.")
    year: Optional[int] = Field(description="The year of the document (YYYY).")
    month: Optional[int] = Field(description="The month of the document (1-12).")
    day: Optional[int] = Field(description="The day of the document (1-31).")
    reasoning: str = Field(description="Quote the text that indicates this date.")

class TemporalExtractor:
    def __init__(self):
        self.llm = get_llm()
        self.structured_llm = self.llm.with_structured_output(DateExtraction)

    def extract_date(self, first_chunks: List[str], last_chunks: List[str], title: str) -> str:
        """
        Extracts the document date from the first 6 and last 6 chunks.
        Returns YYYY-MM-DD string.
        Defaults to Today if not found.
        Defaults to 1st of month if day missing.
        """
        
        # Combine chunks for context
        context_text = f"--- DOCUMENT TITLE: {title} ---\n\n"
        context_text += "--- BEGINNING OF DOCUMENT ---\n"
        context_text += "\n...\n".join(first_chunks) if first_chunks else "(No beginning chunks provided)"
        context_text += "\n\n--- END OF DOCUMENT ---\n"
        context_text += "\n...\n".join(last_chunks) if last_chunks else "(No ending chunks provided)"
        
        prompt = (
            f"You are a specialist in temporal data extraction.\n"
            f"Your TASK: Determine the Creation Date or Publication Date of this document.\n\n"
            f"{context_text}\n\n"
            f"RULES:\n"
            f"1. LOOK FOR: Titles page dates, headers, footers, 'Published on', 'Date:', 'As of'.\n"
            f"2. IGNORE: Historical dates mentioned in the text (e.g. 'In 1990, the company...'). ONLY extract the date of the document itself.\n"
            f"3. IF NOT FOUND: Return date_found=False.\n"
            f"4. IF PARTIAL: If valid Year/Month found but no Day, return them.\n"
        )
        
        try:
            result = self.structured_llm.invoke(prompt)
            
            if not result.date_found or not result.year:
                print(f"   ⚠️ Temporal Extractor: No date found for '{title}'. Using default (Today).")
                return datetime.date.today().isoformat()
            
            # Construct date
            year = result.year
            month = result.month if result.month else 1
            day = result.day if result.day else 1
            
            # Basic validation
            try:
                dt = datetime.date(year, month, day)
                # print(f"   📅 Temporal Extractor: Found {dt.isoformat()} for '{title}' (Reason: {result.reasoning})")
                return dt.isoformat()
            except ValueError:
                # Handle invalid dates (e.g. Feb 30) -> Fallback to 1st or Today
                print(f"   ⚠️ Temporal Extractor: Invalid date {year}-{month}-{day}. Fallback to Today.")
                return datetime.date.today().isoformat()

        except Exception as e:
            print(f"   ⚠️ Temporal Extractor Failed: {e}")
            return datetime.date.today().isoformat()

    def enrich_chunks(self, chunks: List[Chunk], title: str) -> List[Chunk]:
        """
        Extracts date and adds it to the metadata of all chunks.
        """
        if not chunks:
            return chunks
        
        # Select first 6 and last 6 text chunks
        text_chunks = [c.body for c in chunks if c.body] # filter empty bodies just in case
        first_6 = text_chunks[:6]
        last_6 = text_chunks[-6:] if len(text_chunks) > 6 else []
        if len(text_chunks) <= 6: 
            last_6 = [] # overlap handled naturally but being explicit
            
        doc_date = self.extract_date(first_6, last_6, title)
        print(f"   📅 Temporal Extractor: Applying date {doc_date} to {len(chunks)} chunks.")
        
        for chunk in chunks:
            # chunk.metadata is a TypedDict or Dict, so we can assign
            chunk.metadata["doc_date"] = doc_date
            
        return chunks
