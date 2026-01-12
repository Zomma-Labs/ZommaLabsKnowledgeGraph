import pytest
from src.agents.entity_extractor import EntityExtractor
from src.schemas.financial_relation import FinancialRelation

def test_single_extraction():
    extractor = EntityExtractor()
    chunk = "Inflation rose by 2% in the last quarter."
    fact = "Inflation rose by 2%."
    header = "Economy > Inflation"
    
    relations = extractor.extract(fact, chunk, header)
    assert len(relations) >= 1
    # Check if 'Inflation' is subject
    assert any(r.subject.lower() in ["inflation", "prices"] for r in relations)

def test_aggregate_splitting():
    extractor = EntityExtractor()
    chunk = "Contacts in Minneapolis and Dallas reported strong growth."
    fact = "Contacts in a few districts reported strong growth."
    header = "Districts > Summary"
    
    # We expect the extractor to resolve "A few districts" to Minneapolis and Dallas using the chunk context
    relations = extractor.extract(fact, chunk, header)
    
    # Needs at least 2 relations (Minneapolis, Dallas)
    print(f"\nDistricts Found: {[r.subject for r in relations]}")
    assert len(relations) >= 2
    
    subjects = [r.subject.lower() for r in relations]
    assert any("minneapolis" in s for s in subjects)
    assert any("dallas" in s for s in subjects)

def test_header_resolution():
    extractor = EntityExtractor()
    chunk = "Employment levels remained steady."
    fact = "Employment levels remained steady."
    header = "District 9 > Labor Market"
    
    relations = extractor.extract(fact, chunk, header)
    
    # We hope it resolves to 'District 9' or links 'Employment' contextually
    # Ideally subject is 'District 9' or 'Employment' with 'District 9' context?
    # Or maybe Topic='Labor Market'
    # Prompt says: "RESOLVE... 'The District' to specific names". 
    # Here the fact doesn't say "The District", but it's implied context.
    # Let's try a fact that explicitly needs resolution
    
    fact_needs_res = "The District saw steady growth."
    chunk_needs_res = "The District saw steady growth in labor."
    header_needs_res = "Federal Reserve Bank of Minneapolis"
    
    relations = extractor.extract(fact_needs_res, chunk_needs_res, header_needs_res)
    print(f"\nResolution Result: {[r.subject for r in relations]}")
    
    assert any("minneapolis" in r.subject.lower() for r in relations)
