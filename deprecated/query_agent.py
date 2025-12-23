from src.tools.neo4j_client import Neo4jClient
from src.util.llm_client import get_embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from src.schemas.relationship import RelationshipType
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class QueryAgent:
    def __init__(self):
        self.neo4j = Neo4jClient()
        # Use Gemini 2.5 Flash as requested for speed and long context
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        self.embeddings = get_embeddings()
        
    def query_graph(self, user_query: str):
        print(f"🔎 Researching: {user_query}")
        
        # 1. Resolve Entities (Vector Search)
        entities = self._resolve_entities(user_query)
        if not entities:
            return "I couldn't find any relevant entities in the Knowledge Graph to answer your question."
            
        print(f"   ✅ Identified {len(entities)} entities: {[e['name'] for e in entities]}")
        
        # 2. Expand Context (Facts + Chunks)
        context_data = self._expand_context(entities)
        if not context_data['chunks']:
            return "I found the entities, but there is no detailed information (chunks) available in the graph to answer your question."
            
        print(f"   ✅ Retrieved {len(context_data['facts'])} facts and {len(context_data['chunks'])} source chunks.")
        
        # 3. Synthesize Answer (Deep Research)
        return self._synthesize_answer(user_query, context_data)

    def _resolve_entities(self, user_query: str) -> list:
        """
        Uses vector search to find relevant entities AND sections from the query.
        """
        # Embed the query
        query_vector = self.embeddings.embed_query(user_query)
        
        entities = []
        
        # 1. Search in 'entity_embeddings' index
        results_entities = self.neo4j.vector_search(
            index_name="entity_embeddings",
            query_vector=query_vector,
            top_k=5 
        )
        for row in results_entities:
            node = row['node']
            node['type'] = 'Entity' # Tag it
            score = row['score']
            if score > 0.85: 
                entities.append(node)

        return entities

    def _expand_context(self, entities: list) -> dict:
        """
        Retrieves facts and chunks. Handles both Entities and SectionNodes.
        """
        entity_names = [e['name'] for e in entities if e.get('type') == 'Entity']
        
        facts = []
        chunks = {} 
        
        # 1. Entity Traversal
        if entity_names:
            cypher_entity = """
            MATCH (e:EntityNode)-[r]->(ep:EpisodicNode)
            WHERE e.name IN $names
            OPTIONAL MATCH (f:FactNode)-[:MENTIONED_IN]->(ep)
            RETURN f.content AS fact, f.fact_type AS type, ep.content AS chunk, ep.source AS source, ep.valid_at AS date
            UNION
            MATCH (f:FactNode)-[:MENTIONED_IN]->(ep:EpisodicNode)
            WHERE any(word IN $names WHERE toLower(f.content) CONTAINS toLower(word))
            RETURN f.content AS fact, f.fact_type AS type, ep.content AS chunk, ep.source AS source, ep.valid_at AS date
            """
            results_entity = self.neo4j.query(cypher_entity, {"names": entity_names})
            for row in results_entity:
                facts.append(f"[{row['type']}] {row['fact']}")
                chunks[row['chunk']] = {
                    "content": row['chunk'],
                    "source": row['source'],
                    "date": row['date']
                }


            
        return {
            "facts": list(set(facts)), 
            "chunks": list(chunks.values()) 
        }

    def _synthesize_answer(self, user_query: str, context: dict) -> str:
        """
        Uses Gemini to synthesize an answer from the raw chunks.
        """
        chunks_text = "\n\n".join([
            f"--- SOURCE ({c['date']}) ---\n{c['content']}\n" 
            for c in context['chunks']
        ])
        
        template = """You are a specialized Research Agent. Your goal is to answer the user's question in depth using ONLY the provided source text.
        
        User Question: {query}
        
        --- SOURCE MATERIALS START ---
        {chunks}
        --- SOURCE MATERIALS END ---
        
        Instructions:
        1. Answer the question comprehensively.
        2. STRICTLY use ONLY the provided Source Materials. Do NOT use external knowledge.
        3. If the answer is not in the sources, say "I cannot answer this based on the available information."
        4. CITE YOUR SOURCES. When you use information, reference the specific details or context from the chunks.
        5. Synthesize the facts into a cohesive narrative.
        
        Answer:
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        
        return chain.invoke({"query": user_query, "chunks": chunks_text})
