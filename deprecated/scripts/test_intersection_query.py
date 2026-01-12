import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.workflows.intersection_query import IntersectionQueryAgent

def run_test():
    agent = IntersectionQueryAgent()
    
    # Use the group_id that was likely used in the demo/pipeline
    # The user clarified that the tenant is default_tenant
    group_id = "default_tenant" 
    
    queries = [
        "How are the labor markets in New York?",
        "How are the labor markets in Boston?"
    ]
    
    print(f"--- Testing Intersection Query Agent (Group: {group_id}) ---")
    
    for q in queries:
        print(f"\n\n{'='*60}")
        print(f"QUERY: {q}")
        print(f"{'='*60}")
        
        response = agent.run(q, group_id=group_id)
        
        print(f"\n🤖 FINAL ANSWER:\n{response}")

if __name__ == "__main__":
    run_test()
