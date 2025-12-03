import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.workflows.query_workflow import build_query_graph

def run_test(query: str):
    print(f"\n\n{'='*50}")
    print(f"TEST QUERY: {query}")
    print(f"{'='*50}\n")
    
    graph = build_query_graph()
    
    # Run the graph
    inputs = {"input": query, "plan": [], "past_steps": []}
    
    final_response = "No response generated."
    
    for event in graph.stream(inputs):
        for key, value in event.items():
            print(f"\n--- Node: {key} ---")
            if key == "replanner" and "response" in value:
                final_response = value['response']
                print(f"\nFINAL RESPONSE:\n{final_response}")

    # Log to file
    with open("test_results.md", "a") as f:
        f.write(f"## Query: {query}\n\n")
        f.write(f"**Response:**\n{final_response}\n\n")
        f.write("---\n\n")

if __name__ == "__main__":
    # Clear previous results
    with open("test_results.md", "w") as f:
        f.write("# Query Agent Test Results\n\n")

    # test_queries = [
    #     "What are the reports on wage pressures?",
    #     "How is the labor market performing in the New York district?",
    #     "What sectors are seeing price increases?",
    #     "Tell me about 'Specific Non-FIBO Entity' if it exists, or just 'wage pressures' again to test fallback."
    # ]
    test_queries = [
        "What are the reported labor market conditions in Boston?",
        "What are the reported labor market conditions in New York?",
    ]
    
    for q in test_queries:
        run_test(q)
