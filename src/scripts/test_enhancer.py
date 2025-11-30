import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.agents.graph_enhancer import GraphEnhancer

def test_enhancer():
    print("Initializing GraphEnhancer...")
    try:
        enhancer = GraphEnhancer()
    except Exception as e:
        print(f"Failed to init GraphEnhancer: {e}")
        return

    # 1. Test Reflexion
    print("\n--- Testing Reflexion ---")
    text = "Apple released the new iPhone 15. The tech giant expects high sales."
    existing_facts = ["Apple released iPhone 15."]
    
    print(f"Text: {text}")
    print(f"Existing Facts: {existing_facts}")
    
    try:
        missed = enhancer.reflexion_check(text, existing_facts)
        print(f"Missed Facts: {missed}")
    except Exception as e:
        print(f"Reflexion test failed: {e}")
    
    # 2. Test Resolution (Mock)
    print("\n--- Testing Resolution Logic ---")
    name = "ZommaLabs"
    candidates = [
        {"uuid": "123", "name": "Zomma Labs Inc", "score": 0.95},
        {"uuid": "456", "name": "Zomma Corp", "score": 0.80}
    ]
    
    try:
        decision = enhancer.resolve_entity_against_graph(name, candidates)
        print(f"Decision for '{name}' with candidates: {decision}")
    except Exception as e:
        print(f"Resolution test failed: {e}")
    
    # 3. Test Resolution (Empty)
    try:
        decision_empty = enhancer.resolve_entity_against_graph("NewEntity", [])
        print(f"Decision for 'NewEntity' (no candidates): {decision_empty}")
    except Exception as e:
        print(f"Empty resolution test failed: {e}")

if __name__ == "__main__":
    test_enhancer()
