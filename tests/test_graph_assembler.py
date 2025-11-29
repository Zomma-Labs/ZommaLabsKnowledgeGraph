import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.schemas.atomic_fact import AtomicFact
from src.schemas.relationship import RelationshipClassification, RelationshipType
from src.agents.graph_assembler import GraphAssembler

def test_assembler():
    print("🧪 Testing Graph Assembler...")
    
    # 1. Mock Data
    fact = AtomicFact(
        fact="Apple sued Microsoft for patent infringement in 2024.",
        subject="Apple Inc.",
        object="Microsoft Corp.",
        date_context="2024-01-01"
    )
    
    rel = RelationshipClassification(
        relationship=RelationshipType.SUED,
        confidence=0.95,
        reasoning="The text explicitly states 'sued'."
    )
    
    # Mock URIs (simulating Librarian output)
    subj_uri = "fibo:Apple_Inc"
    obj_uri = "fibo:Microsoft_Corp"
    
    # 2. Initialize Assembler
    # NOTE: This requires a running Neo4j instance and valid .env
    # If not available, it will fail gracefully in the try/except block of the assembler or client.
    assembler = GraphAssembler()
    
    # 3. Run Assembly
    print(f"   Input Fact: {fact.fact}")
    print(f"   Subject: {fact.subject} ({subj_uri})")
    print(f"   Object: {fact.object} ({obj_uri})")
    print(f"   Relationship: {rel.relationship}")
    
    assembler.assemble_and_write(fact, subj_uri, obj_uri, rel)
    
    # 4. Test Concept Fallback
    print("\n🧪 Testing Concept Fallback...")
    fact_concept = AtomicFact(
        fact="Apple revenue grew by 5%.",
        subject="Apple Inc.",
        object="Revenue", # Concept
        date_context="2024-01-01"
    )
    rel_concept = RelationshipClassification(
        relationship=RelationshipType.REPORTED_FINANCIALS, # Closest match
        confidence=0.8,
        reasoning="Revenue growth is a financial report."
    )
    
    print(f"   Input Fact: {fact_concept.fact}")
    print(f"   Subject: {fact_concept.subject} ({subj_uri})")
    print(f"   Object: {fact_concept.object} (None - Concept)")
    
    assembler.assemble_and_write(fact_concept, subj_uri, None, rel_concept)

    # 5. Test Missing Date
    print("\n🧪 Testing Missing Date...")
    fact_no_date = AtomicFact(
        fact="The Federal Reserve raised interest rates.",
        subject="Federal Reserve",
        object="Interest Rates",
        date_context=None # Missing date
    )
    rel_macro = RelationshipClassification(
        relationship=RelationshipType.RAISED_POLICY_RATE,
        confidence=0.9,
        reasoning="Explicit statement about raising rates."
    )
    assembler.assemble_and_write(fact_no_date, "fibo:Federal_Reserve", None, rel_macro)

    # 6. Test Low Confidence
    print("\n🧪 Testing Low Confidence...")
    fact_ambiguous = AtomicFact(
        fact="Rumors suggest Apple might buy Disney.",
        subject="Apple Inc.",
        object="Disney",
        date_context="2024-05-01"
    )
    rel_low_conf = RelationshipClassification(
        relationship=RelationshipType.ACQUIRED,
        confidence=0.4, # Low confidence
        reasoning="Rumor, not confirmed fact."
    )
    assembler.assemble_and_write(fact_ambiguous, subj_uri, "fibo:Disney", rel_low_conf)

    # 7. Test Error Handling (Invalid Input)
    print("\n🧪 Testing Error Handling...")
    try:
        # Intentionally passing None as fact object to trigger error
        assembler.assemble_and_write(None, subj_uri, obj_uri, rel)
    except Exception as e:
        print(f"   ✅ Caught expected error: {e}")

    assembler.close()
    print("\n✅ Test Complete (Check Neo4j for results)")

if __name__ == "__main__":
    test_assembler()
