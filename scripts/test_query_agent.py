import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.query_agent import QueryAgent

def test_query_agent():
    print("Initializing QueryAgent...")
    try:
        agent = QueryAgent()
    except Exception as e:
        print(f"Failed to initialize QueryAgent: {e}")
        return

    queries = [
        "What are the reported labor market conditions in Boston?",
        "What are the reported labor market conditions in New York?",
    ]
    
    for query in queries:
        print(f"\nRunning query: '{query}'")
        try:
            response = agent.query_graph(query)
            print("\n--- Response ---")
            print(response)
            print("----------------")
        except Exception as e:
            print(f"Error running query: {e}")

if __name__ == "__main__":
    test_query_agent()
