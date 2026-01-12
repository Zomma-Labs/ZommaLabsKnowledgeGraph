import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.agents.analyst import AnalystAgent

def main():
    agent = AnalystAgent()
    
    test_facts = [
        "The Federal Reserve hiked the benchmark interest rate by 0.5%.",
        "Alphabet created a new subsidiary, Waymo, to handle its self-driving car business.",
        "The US government imposed sanctions on several foreign banks.",
        "Johnson & Johnson spun off its consumer health division into a new company called Kenvue.",
        "The Bureau of Labor Statistics released data showing inflation rose to 3.2%.",
        "Microsoft integrated its Nuance acquisition into the healthcare cloud division."
    ]
    
    print("Testing Analyst Agent with Expanded Relationships...")
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
