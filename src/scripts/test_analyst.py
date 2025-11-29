import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.agents.analyst import AnalystAgent

def main():
    agent = AnalystAgent()
    
    test_facts = [
        "Google completed its acquisition of Fitbit for $2.1 billion.",
        "The DOJ filed a lawsuit against Apple regarding its App Store monopoly.",
        "Microsoft and OpenAI announced a strategic partnership.",
        "Tesla opened a new factory in Texas.",
        "Amazon launched a new pharmacy service."
    ]
    
    print("Testing Analyst Agent...")
    for fact in test_facts:
        print(f"\nFact: {fact}")
        result = agent.classify_relationship(fact)
        if result:
            print(f"Result: {result.relationship} (Confidence: {result.confidence})")
            print(f"Reasoning: {result.reasoning}")
        else:
            print("Result: Could not classify.")

if __name__ == "__main__":
    main()
