import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.agents.analyst import AnalystAgent

def main():
    agent = AnalystAgent()
    
    test_facts = [
        # Deep Signals
        "Contacts noted that wage pressures were easing as the supply of labor improved.",
        "Retailers noted that customers were becoming more price-sensitive, limiting their ability to raise prices.",
        "Banks reported tighter lending standards for commercial real estate loans.",
        "Office vacancy rates remained high, leading to distress in the downtown CRE market.",
        
        # General / Mixed
        "Economic activity expanded slightly in the Richmond district.",
        "Hiring was flat across most sectors.",
        "Input costs rose moderately due to higher fuel prices.",
        "Consumer spending was mixed, with strength in services but softness in goods."
    ]
    
    print("Testing Analyst Agent with Beige Book Relationships...")
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
