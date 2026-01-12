import json
import sys
from src.agents.entity_extractor import EntityExtractor

def test_table_row_extraction():
    extractor = EntityExtractor()
    
    # User provided example
    row_data = {
        "Ranks": "18 McKesson",
        "Name | Industry": "Healthcare",
        "Revenue | USD (in millions)": "$308,951",
        "Profit | USD (in millions)": "$3,002",
        "Employees": "48,000",
        "Headquarters [note 1]": "Japan",
        "Ref.": "[21]"
    }
    fact_text = json.dumps(row_data)
    chunk_text = "Table of Top Companies. " + fact_text
    header_path = "Business > Rankings"

    print(f"\nTesting Fact: {fact_text}\n")

    try:
        relations = extractor.extract(fact_text, chunk_text, header_path)
    except Exception as e:
        print(f"❌ Extraction failed with error: {e}")
        sys.exit(1)
    
    print("\n--- Extracted Relations ---")
    for r in relations:
        print(f"Subj: {r.subject} ({r.subject_type}) -> Obj: {r.object} ({r.object_type}) | Topics: {r.topics}")

    # Assertions
    subjects = [r.subject for r in relations]
    objects = [r.object for r in relations]
    topics = []
    for r in relations:
        topics.extend(r.topics)
    
    # 1. Start with what we WANT TO FIND
    # "McKesson" should be extracted
    mckesson_found = any("McKesson" in s for s in subjects) or any("McKesson" in str(o) for o in objects if o)
    if not mckesson_found:
        print("❌ FAILED: Company 'McKesson' was not found in Subject or Object.")
        
    # "Healthcare" should be extracted
    healthcare_found = any("Healthcare" in s for s in subjects) or any("Healthcare" in str(o) for o in objects if o)
    if not healthcare_found:
        print("❌ FAILED: Industry 'Healthcare' was not found.")

    # "Japan" should be extracted
    japan_found = any("Japan" in s for s in subjects) or any("Japan" in str(o) for o in objects if o)
    if not japan_found:
        print("❌ FAILED: Location 'Japan' was not found.")
    
    # "Revenue" should be a Topic (or involved in relation)
    revenue_found = any("Revenue" in t for t in topics) or any("Revenue" in str(r) for r in relations)
    if not revenue_found:
        print("❌ FAILED: Topic 'Revenue' was not found.")

    # 2. Start with what we DO NOT WANT
    # Should NOT extract the raw numbers as entities/topics
    forbidden_values = ["$308,951", "308,951", "$3,002", "3,002", "48,000", "[21]"]
    
    failed_forbidden = False
    for val in forbidden_values:
        if any(val in s for s in subjects):
            print(f"❌ FAILED: Extracted forbidden value '{val}' as Subject.")
            failed_forbidden = True
        if any(val in str(o) for o in objects if o):
            print(f"❌ FAILED: Extracted forbidden value '{val}' as Object.")
            failed_forbidden = True
        if any(val in str(t) for t in topics):
            print(f"❌ FAILED: Extracted forbidden value '{val}' as Topic.")
            failed_forbidden = True

    if not all([mckesson_found, healthcare_found, japan_found, revenue_found]) or failed_forbidden:
        print("\n❌ TEST FAILED")
        sys.exit(1)
    else:
        print("\n✅ TEST PASSED")
        sys.exit(0)

if __name__ == "__main__":
    test_table_row_extraction()
