import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.workflows.query_workflow import build_query_graph

def main():
    try:
        app = build_query_graph()
    except Exception as e:
        print(f"Failed to initialize QueryWorkflow: {e}")
        return

    questions = [
        "What did contacts note about wage pressures?",
        "How were office vacancy rates?",
        "What did retailers say about customers?",
        "What sectors had flat hiring?"
    ]
    
    print("Testing Agentic Query Workflow with Beige Book Questions...")
    for q in questions:
        print(f"\n{'='*50}")
        print(f"Question: {q}")
        print(f"{'='*50}")
        
        try:
            # Initialize state with empty lists
            inputs = {
                "input": q,
                "plan": [],
                "past_steps": []
            }
            result = app.invoke(inputs)
            print(f"\nFinal Answer: {result.get('response', 'No response generated.')}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
