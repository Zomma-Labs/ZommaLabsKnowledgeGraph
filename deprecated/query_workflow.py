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
            - lookup_entity(query): Finds existing entities in the graph by name. Returns Name and UUID.
            - lookup_relationship(description): Finds schema relationship types (edge labels). Use this to understand how to traverse.
            - execute_cypher(query): Runs Cypher queries. THIS IS YOUR MAIN TOOL FOR CRAWLING.
            
            Constraints:
            - DO NOT plan to search the internet, external websites, or other datasets.
            - ONLY use the provided tools to query the Neo4j Knowledge Graph.
            - If you need information, you must find it in the graph.
            
            GRAPH SCHEMA (Hypergraph with EpisodicNode Hub):
            - **EntityNode**: Named entities (companies, people, locations). Has `name`, `uuid`, `summary`.
            - **TopicNode**: Thematic concepts (Inflation, Labor Market). Has `name`, `uuid`, `summary`.
            - **EpisodicNode**: The central hub containing the raw text chunk. Has `uuid`, `content`, `header_path`.
            - **FactNode**: Atomic facts extracted from chunks. Has `uuid`, `content`, `fact_type`.
            
            EDGE PATTERNS:
            - `(EntityNode)-[SEMANTIC_EDGE {{fact_id: ...}}]->(EpisodicNode)-[PASSIVE_EDGE {{fact_id: ...}}]->(EntityNode)`
            - Active edges: `INVESTED`, `REPORTED_WAGE_TRENDS`, `INCREASED`, etc.
            - Passive edges: `GOT_INVESTED`, `GOT_REPORTED_WAGE_TRENDS`, etc.
            - The `fact_id` property links active and passive edges through the same EpisodicNode.
            - `(TopicNode)-[:ABOUT]->(EpisodicNode)` for topic associations.
            - `(FactNode)-[:MENTIONED_IN]->(EpisodicNode)` for provenance.
            - `(FactNode)-[:CAUSES]->(FactNode)` for causality.
            
            Strategy:
            1. Identify key entities and find their UUIDs using `lookup_entity`. **Anticipate multiple matches**.
            2. Identify edge types using `lookup_relationship` (e.g., "wage pressure" → `REPORTED_WAGE_TRENDS`).
            3. Construct a Cypher query to traverse through EpisodicNode:
               `MATCH (e:EntityNode {{name: '...'}})-[r1]->(ep:EpisodicNode)-[r2]->(target) WHERE r1.fact_id = r2.fact_id RETURN ...`
            4. Retrieve the chunk content from EpisodicNode for evidence.
            
            Example Plan:
            1. Find UUID for "Minneapolis District".
            2. Look up edge types for "wage pressure".
            3. Query: `MATCH (e:EntityNode {{name: 'Minneapolis District'}})-[r:REPORTED_WAGE_TRENDS]->(ep:EpisodicNode) RETURN ep.content, ep.header_path`.
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
        - Always start by finding existing entity names/UUIDs using `lookup_entity`.
        - Use `lookup_relationship` to find the correct edge type (e.g., `REPORTED_WAGE_TRENDS`).
        
        IMPORTANT SCHEMA RULES (Hypergraph with EpisodicNode Hub):
        1. The graph uses a **Hypergraph** model. Entities connect THROUGH an EpisodicNode (chunk), not directly.
        2. **Node Types**:
           - `EntityNode`: Named entities (has `name`, `uuid`, `summary`, `group_id`).
           - `TopicNode`: Thematic concepts (has `name`, `uuid`, `summary`, `group_id`).
           - `EpisodicNode`: The chunk hub (has `uuid`, `content`, `header_path`, `group_id`).
           - `FactNode`: Atomic facts (has `uuid`, `content`, `fact_type`, `group_id`).
        3. **Edge Structure**:
           - `(EntityNode)-[ACTIVE_EDGE {{fact_id: $id}}]->(EpisodicNode)-[PASSIVE_EDGE {{fact_id: $id}}]->(EntityNode)`
           - The `fact_id` property links the active and passive edges through the same chunk.
           - Active edges are semantic verbs: `INVESTED`, `INCREASED`, `REPORTED_WAGE_TRENDS`, etc.
           - Passive edges are prefixed: `GOT_INVESTED`, `GOT_INCREASED`, etc.
        4. **Other Edges**:
           - `(TopicNode)-[:ABOUT]->(EpisodicNode)` - Topic associations.
           - `(FactNode)-[:MENTIONED_IN]->(EpisodicNode)` - Provenance.
           - `(FactNode)-[:CAUSES]->(FactNode)` - Causality chains.
        
        HOW TO QUERY:
        - To find all relationships for an entity:
          `MATCH (e:EntityNode {{name: $name}})-[r]->(ep:EpisodicNode) RETURN type(r) as edge_type, ep.content, ep.header_path`
        - To find specific relationships with targets:
          `MATCH (e:EntityNode {{name: $name}})-[r1:REPORTED_WAGE_TRENDS]->(ep:EpisodicNode)-[r2]->(target) WHERE r1.fact_id = r2.fact_id RETURN target.name, ep.content`
        - To find chunks about a topic:
          `MATCH (t:TopicNode {{name: $topic}})-[:ABOUT]->(ep:EpisodicNode) RETURN ep.content`
        
        HANDLING MULTIPLE RESULTS:
        - If `lookup_entity` returns multiple entities (e.g., "New York" matches multiple districts):
            - **Do NOT** arbitrarily pick one.
            - **Filter** them based on the context of the user's question.
            - If multiple are relevant, **explore ALL of them** using `WHERE e.name IN [...]`.
        - If a Cypher query returns many nodes:
            - **Analyze** the results to find the ones that best answer the specific question.
            - **Synthesize** information from multiple relevant EpisodicNode contents if needed.
        
        Constraints:
        - Edge types ARE the relationship labels (e.g., `[:REPORTED_WAGE_TRENDS]`), NOT properties on FactNodes.
        - Always traverse through EpisodicNode to find connected entities.
        - Use `fact_id` matching when you need to find the target of a specific relationship.
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
