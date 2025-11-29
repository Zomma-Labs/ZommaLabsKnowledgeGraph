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
        self.tools = [lookup_entity, lookup_relationship, execute_cypher, search_graph_text]
        
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
            """For the given objective, come up with a simple step-by-step plan.
            This plan should involve looking up entities, finding relationship types, and then executing Cypher queries.
            
            Objective: {objective}
            
            Tools available:
            - lookup_entity(query): Resolves names to FIBO URIs.
            - lookup_relationship(description): Finds schema relationship types.
            - execute_cypher(query): Runs Cypher.
            - search_graph_text(keywords): Searches for text in facts (e.g. "wage pressures"). Use this FIRST if the query is about a general topic.
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
        
        prompt = f"""You are executing the following task: {task}
        
        Context from previous steps:
        {context}
        
        Use your tools to complete this task. Return the result of your work.
        
        IMPORTANT SCHEMA RULES:
        1. Nodes are ONLY labeled as `Entity` or `Concept`. Do NOT use other labels like `Person`, `Company`, etc.
        2. Relationships must be one of the types returned by `lookup_relationship`.
        3. RELATIONSHIP TYPES MUST BE ALL CAPS (e.g., `REPORTED_WAGE_PRESSURE_RISING`, not `Reported_Wage_Pressure_Rising`).
        4. `Entity` nodes have `uri` and `name` properties.
        5. Relationships have `fact`, `date`, and `confidence` properties.
        6. When searching for text in facts, check `r.fact`.
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
