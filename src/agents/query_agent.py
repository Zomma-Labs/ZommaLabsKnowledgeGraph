from src.tools.neo4j_client import Neo4jClient
from src.util.llm_client import get_llm
from src.schemas.relationship import RelationshipType
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class QueryAgent:
    def __init__(self):
        self.neo4j = Neo4jClient()
        self.llm = get_llm()
        
    def query_graph(self, user_query: str):
        # 1. Generate Cypher
        cypher_query = self._generate_cypher(user_query)
        print(f"Generated Cypher: {cypher_query}")
        
        # 2. Execute Cypher
        try:
            results = self.neo4j.query(cypher_query)
        except Exception as e:
            return f"Error executing Cypher query: {e}"
            
        # 3. Synthesize Answer
        return self._synthesize_answer(user_query, results)

    def _generate_cypher(self, user_query: str) -> str:
        # Construct prompt with schema
        rel_types = [r.value for r in RelationshipType]
        schema_desc = f"""
        Nodes: 
        - Entity (properties: uri, name)
        - Concept (properties: uri, name)
        
        Relationships:
        - Types: {', '.join(rel_types)}
        - Properties on relationships: fact (string), date (string), confidence (float)
        
        Pattern: (Entity)-[RELATIONSHIP]->(Entity|Concept)
        """
        
        template = """You are a Neo4j Cypher expert. Convert the user's natural language query into a Cypher query.
        Use the following schema:
        {schema}
        
        IMPORTANT: 
        - Return ONLY the Cypher query. No markdown, no explanations.
        - Use case-insensitive matching for names if needed (e.g. toLower(n.name) CONTAINS ...).
        - Always return relevant properties like s.name, r.fact, r.date, o.name.
        
        User Query: {query}
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({"schema": schema_desc, "query": user_query})
        # Clean up response (remove markdown code blocks if any)
        return response.replace("```cypher", "").replace("```", "").strip()

    def _synthesize_answer(self, user_query: str, results: list) -> str:
        if not results:
            return "No information found in the knowledge graph matching your query."
            
        template = """You are a helpful assistant. Answer the user's query based on the following results from the knowledge graph.
        
        User Query: {query}
        
        Graph Results:
        {results}
        
        Answer:
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        
        return chain.invoke({"query": user_query, "results": str(results)})
