import operator
from typing import Annotated, List, Tuple, TypedDict, Union, Optional
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent

from src.util.llm_client import get_llm
from src.tools.agent_tools import lookup_entity, lookup_relationship, execute_cypher, search_graph_text

# --- State Definition ---
class PlanExecuteState(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str

# --- Nodes ---

class PlanExecuteWorkflow:
    def __init__(self):
        self.llm = get_llm()
        self.tools = [lookup_entity, lookup_relationship, execute_cypher]
        
        # The Executor is a standard ReAct agent
        self.executor_agent = create_react_agent(self.llm, self.tools)

    def planner_node(self, state: PlanExecuteState):
        """
        Generates the initial plan (Todo List).
        """
        print(f"--- Planning: {state['input']} ---")
        
        class Plan(BaseModel):
            steps: List[str] = Field(description="List of steps to answer the user's question.")
            
        planner_prompt = ChatPromptTemplate.from_template(
            """For the given objective, create a step-by-step plan to "crawl" the knowledge graph and find the answer.
            
            Objective: {objective}
            
            Tools available:
            - lookup_entity(query): Finds existing entities in the graph by name. Returns Name and URI.
            - lookup_relationship(description): Finds schema relationship types. Use this to understand how to traverse.
            - execute_cypher(query): Runs Cypher queries. THIS IS YOUR MAIN TOOL FOR CRAWLING.
            
            Constraints:
            - DO NOT plan to search the internet, external websites, or other datasets.
            - ONLY use the provided tools to query the Neo4j Knowledge Graph.
            - If you need information, you must find it in the graph.
            
            Strategy:
            1. Identify key entities in the request and find their URIs using `lookup_entity`. **Anticipate multiple matches** and plan to handle them (e.g., "Check all 'New York' entities").
            2. Identify the `fact_type` (relationship type) using `lookup_relationship`.
            3. Construct a Cypher query using the **Fact-as-Node** pattern:
               `MATCH (e:Entity {{uri: '...'}})-[:PERFORMED|TARGET]-(f:FactNode {{fact_type: '...'}}) RETURN f.content`
            4. Crawl by traversing from FactNodes to other Entities or causally linked FactNodes.
            
            Example Plan:
            1. Find URI for "Inflation".
            2. Look up fact types for "increasing".
            3. Query: `MATCH (e:Entity {{uri: '...'}})-[:PERFORMED|TARGET]-(f:FactNode)-[:TARGET|PERFORMED]-(other) WHERE f.fact_type = 'REPORTED_PRICE_PRESSURES' RETURN f.content, other.name`.
            """
        )
        planner = planner_prompt | self.llm.with_structured_output(Plan)
        plan = planner.invoke({"objective": state["input"]})
        
        return {"plan": plan.steps}

    def executor_node(self, state: PlanExecuteState):
        """
        Executes the first step of the plan using the ReAct agent.
        """
        plan = state["plan"]
        task = plan[0]
        print(f"--- Executing Step: {task} ---")
        
        # Provide context from past steps
        context = "\n".join([f"Step: {step}\nResult: {result}" for step, result in state["past_steps"]])
        
        prompt = f"""You are an intelligent graph crawler. Your goal is to execute the following task by exploring the Knowledge Graph.
        
        Task: {task}
        
        Context from previous steps:
        {context}
        
        Use your tools to explore. 
        - Always start by finding existing entity URIs using `lookup_entity`.
        - Use `lookup_relationship` to find the correct `fact_type` (e.g., `REPORTED_WAGE_PRESSURE`).
        
        IMPORTANT SCHEMA RULES (Fact-as-Node Pattern):
        1. The graph uses a **Hypergraph** model. Relationships are NOT direct edges between entities.
        2. **Nodes**: `Entity` (has `name`, `uri`), `FactNode` (has `content`, `fact_type`).
        3. **Structure**: `(Entity)-[:PERFORMED]->(FactNode)-[:TARGET]->(Entity)`.
        4. **Causality**: `(FactNode)-[:CAUSES]->(FactNode)`.
        
        HOW TO QUERY:
        - To find facts about an entity: 
          `MATCH (e:Entity {{uri: $uri}})-[:PERFORMED|TARGET]-(f:FactNode) RETURN f.fact_type, f.content`
        - To find specific relationships:
          `MATCH (e:Entity {{uri: $uri}})-[:PERFORMED]-(f:FactNode {{fact_type: 'REPORTED_WAGE_TRENDS'}})-[:TARGET]-(target) RETURN target.name, f.content`
        
        HANDLING MULTIPLE RESULTS:
        - If `lookup_entity` returns multiple entities (e.g., "New York" matches "New York City", "New York State", "Federal Reserve Bank of New York"):
            - **Do NOT** arbitrarily pick one.
            - **Filter** them based on the context of the user's question.
            - If multiple are relevant, **explore ALL of them** or construct a query that matches any of them (e.g., `WHERE e.uri IN [...]`).
        - If a Cypher query returns many nodes:
            - **Analyze** the results to find the ones that best answer the specific question.
            - **Synthesize** information from multiple relevant nodes if needed.
        
        Constraints:
        - DO NOT guess relationship types like `[:REPORTED_...]`. These are `fact_type` properties on `FactNode`s!
        - The only edge types are `PERFORMED`, `TARGET`, `MENTIONED_IN`, `CAUSES`.
        """
        
        # Run the ReAct agent for this single step
        result = self.executor_agent.invoke({"messages": [HumanMessage(content=prompt)]})
        output = result["messages"][-1].content
        
        return {"past_steps": [(task, output)]}

    def replanner_node(self, state: PlanExecuteState):
        """
        Updates the plan based on the result of the last step.
        """
        print("--- Replanning ---")
        
        class Response(BaseModel):
            response: Optional[str] = Field(description="Final answer to the user, if ready.")
            new_plan: Optional[List[str]] = Field(description="Updated plan if not finished.")
            
        replanner_prompt = ChatPromptTemplate.from_template(
            """Your task is to decide what to do next based on the results of the last step.
            
            Original Objective: {input}
            
            Plan:
            {plan}
            
            Past Steps and Results:
            {past_steps}
            
            Constraints:
            - DO NOT plan to search the internet, external websites, or other datasets.
            - ONLY use the provided tools to query the Neo4j Knowledge Graph.
            - If you cannot find the answer in the graph, state that information is missing in the graph.
            
            Update the plan. Remove completed steps. Add new steps if needed (e.g., if a lookup failed, try a different query).
            If you have enough information to answer the user's objective, provide the 'response'.
            """
        )
        
        replanner = replanner_prompt | self.llm.with_structured_output(Response)
        
        # Format past steps for prompt
        past_steps_str = "\n".join([f"Step: {s}\nResult: {r}" for s, r in state["past_steps"]])
        
        result = replanner.invoke({
            "input": state["input"],
            "plan": state["plan"],
            "past_steps": past_steps_str
        })
        
        if result.response:
            return {"response": result.response, "plan": []}
        else:
            return {"plan": result.new_plan}

    def should_end(self, state: PlanExecuteState):
        if state.get("response"):
            return True
        if not state.get("plan"):
            return True # No more steps
        return False

def build_query_graph():
    workflow = PlanExecuteWorkflow()
    
    graph = StateGraph(PlanExecuteState)
    
    graph.add_node("planner", workflow.planner_node)
    graph.add_node("executor", workflow.executor_node)
    graph.add_node("replanner", workflow.replanner_node)
    
    graph.set_entry_point("planner")
    
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "replanner")
    
    graph.add_conditional_edges(
        "replanner",
        workflow.should_end,
        {
            True: END,
            False: "executor"
        }
    )
    
    return graph.compile()
