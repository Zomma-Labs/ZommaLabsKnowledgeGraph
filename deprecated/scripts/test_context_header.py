import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.agents.header_analyzer import HeaderAnalyzer, DimensionType

def test_context_aware_analysis():
    analyzer = HeaderAnalyzer()
    
    headers = ["Boston"]

    # Test Case 1: LLM Extraction from Filename/Text
    filename = "BeigeBook_20251015.pdf"
    text_snippet = "Summary of Economic Activity. Overall economic activity was relatively flat..."
    
    print(f"\n--- Test: LLM Extraction from '{filename}' ---")
    extracted_context = analyzer.extract_document_context(text_snippet, filename)
    print(f"Extracted Context: {extracted_context}")
    
    # Verify Context is reasonable (contains "Beige Book" or "Federal Reserve")
    if "beige book" in extracted_context.lower() or "federal reserve" in extracted_context.lower():
        print("✅ Context extraction successful.")
    else:
        print("❌ Context extraction failed (unexpected output).")

    # Test Case 2: Analysis using Extracted Context
    print(f"\n--- Test: Analysis using Extracted Context ---")
    dims = analyzer.analyze_path(headers, document_context=extracted_context)
    for d in dims:
        print(f" - {d.type}: {d.value}")
        print(f"   Description: {d.description}")
        
        if d.type == DimensionType.ENTITY and "Federal Reserve" in d.description:
             print("✅ Description correctly identifies Federal Reserve context.")

if __name__ == "__main__":
    test_context_aware_analysis()
