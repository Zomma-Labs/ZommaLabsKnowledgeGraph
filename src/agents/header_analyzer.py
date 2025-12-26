from enum import Enum
from typing import List, Union, Optional, TYPE_CHECKING
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.util.services import Services

class DimensionType(str, Enum):
    TOPIC = "TOPIC"
    ENTITY = "ENTITY"

class Dimension(BaseModel):
    value: str = Field(description="The extracted value (e.g., 'New York', 'Labor Markets')")
    type: DimensionType = Field(description="The type of dimension: TOPIC or ENTITY")
    description: str = Field(description="A concise description of what this dimension represents in the given context (e.g., 'Federal Reserve Bank of Boston')")
    original_header: str = Field(description="The original header string this was extracted from")

class HeaderAnalysisResult(BaseModel):
    dimensions: List[Dimension] = Field(description="List of extracted dimensions from the path")

class HeaderAnalyzer:
    def __init__(self, services: Optional["Services"] = None):
        if services is None:
            from src.util.services import get_services
            services = get_services()
        self.llm = services.llm
        self.structured_llm = self.llm.with_structured_output(HeaderAnalysisResult)

    def extract_document_context(self, text_input: Union[str, List[str]], filename: str) -> str:
        """
        Extracts a concise document context/title from the chunk text(s) or filename using LLM.
        Accepts a single string or a list of text chunks (e.g., first 3 chunks).
        """
        if isinstance(text_input, list):
            # Join the first few chunks (limit to 3 if more are passed, though caller should control this)
            preview_text = "\n---\n".join(text_input[:3])
        else:
            preview_text = text_input if text_input else ""
        
        system_prompt = (
            "You are a Document Classifier.\n"
            "Your task is to identify the DOCUMENT TYPE and IMPLIED CONTEXT from the filename and text snippet.\n"
            "Return a concise string (max 10 words) describing the document context.\n"
            "If the document is unknown, return a small description of what the document is about.\n"
            "Examples:\n"
            "- 'Federal Reserve Beige Book (Economic Report)'\n"
            "- 'Apple Inc. 10-K Annual Report'\n"
            "- 'FOMC Meeting Minutes'\n"
            "- 'FED Press Release'\n"
            "- 'News article about how companies are using AI to improve their products and its impact on the economy'\n"
            "- 'Unknown Document Name: This document is about the economy and its impact on the stock market'\n"
        )
        
        try:
            response = self.llm.invoke([
                ("system", system_prompt),
                ("human", f"Filename: {filename}\nText Snippet: {preview_text}")
            ])
            return response.content.strip()
        except Exception as e:
            print(f"Context Extraction failed: {e}")
            return f"Document: {filename}" if filename else "Unknown Document"

    def analyze_path(self, headers: List[str], document_context: str = "") -> List[Dimension]:
        """
        Analyzes a full header path (e.g., ["Regional Reports", "New York", "Labor Markets"])
        and extracts meaningful dimensions (Topics/Entities).
        """
        if not headers:
            return []

        path_str = " > ".join(headers)
        
        system_prompt = (
            "You are a Knowledge Graph Architect.\n"
            "Your task is to analyze a document header path and extract meaningful DIMENSIONS (Topics or Entities).\n\n"
            "Definitions:\n"
            "- TOPIC: A general theme, concept, or subject matter. Examples: 'Inflation', 'Labor Markets', 'Risk Factors', 'Financial Results'.\n"
            "- ENTITY: A specific actor, organization, person, or named place. Examples: 'Federal Reserve', 'Apple Inc.', 'Elon Musk', 'New York'.\n\n"
            "Rules:\n"
            "1. IGNORE structural headers like 'Body', 'Introduction', 'Section 1'.\n"
            "2. Extract specific concepts. If a header is 'Regional Reports', extract 'Regional Reports' as a TOPIC.\n"
            "3. If a header is 'New York', extract 'New York' as an ENTITY (Place/Actor).\n"
            "4. If a header is 'Labor Markets', extract 'Labor Markets' as a TOPIC.\n"
            "5. Return a list of ALL valid dimensions found in the path.\n"
            "6. GENERATE DESCRIPTIONS: For each dimension, generate a concise DESCRIPTION based on the document context.\n"
            "   - This description will be used to resolve the entity in the Knowledge Graph.\n"
            "   - Example: Header='Boston', Context='Federal Reserve Report' -> Value='Boston', Description='Federal Reserve Bank of Boston / District 1'\n"
            "   - Example: Header='Risk Factors', Context='Apple Annual Report' -> Value='Risk Factors', Description='Risk Factors related to Apple Inc. business operations'\n"
        )

        try:
            result = self.structured_llm.invoke([
                ("system", system_prompt),
                ("human", f"Context: {document_context}\nHeader Path: {path_str}")
            ])
            return result.dimensions
        except Exception as e:
            print(f"Header Analysis failed: {e}")
            # Fallback: Treat each header as a Topic
            return [Dimension(value=h, type=DimensionType.TOPIC, description=h, original_header=h) for h in headers]
