"""
MODULE: Entity Extractor
SYSTEM: Financial-GraphRAG Ingestion Pipeline
AUTHOR: ZommaLabs
VERSION: 2.0.0 (Context-Aware & Reflexion)

DESCRIPTION:
    This module defines the `EntityExtractor` agent. 
    It takes a pre-decomposed "Atomic Propostion" (fact) and the ORIGINAL chunk context,
    PLUS the document header path, and extracts the structured entities (Subject, Object, Topics).

    It uses the "Context-Aware" extraction pattern:
    "RESOLVE, EXTRACT THE ENTITIES in this fact... FOR HELP here is the chunk & header..."
    
    It returns a LIST of FinancialRelations, allowing it to split aggregate entities 
    (e.g. "Contacts in a Few Districts" -> [Rel(District A), Rel(District B)]).

INPUT:
    - `fact_text`: The distinct atomic fact string.
    - `chunk_text`: The full text of the chunk where the fact originated.
    - `header_path`: The document structure/breadcrumbs.

OUTPUT:
    - `List[FinancialRelation]`: List of extracted relations (Subject/Object/Topics).
"""

from typing import List, Optional
from src.schemas.financial_relation import FinancialRelation, FinancialRelationList
from src.util.llm_client import get_llm

class EntityExtractor:
    def __init__(self):
        self.llm = get_llm()
        self.structured_llm = self.llm.with_structured_output(FinancialRelationList)

    def extract(self, fact_text: str, chunk_text: str, header_path: str = "") -> List[FinancialRelation]:
        """
        Extracts entities from a fact using the chunk context and header path.
        Returns a list of relations to support splitting aggregate entities.
        """
        prompt = (
            f"HEADER: {header_path}\n"
            f"CHUNK: \"{chunk_text}\"\n\n"
            
            f"FACT TO ANALYZE: \"{fact_text}\"\n\n"
            
            f"GOAL: Extract the detailed FINANCIAL RELATIONSHIPS from the FACT, "
            f"resolving any generic terms using the HEADER and CHUNK context.\n\n"
            
            f"STRICT RULES:\n"
            f"1. CONTEXT AWARENESS: Use the HEADER (e.g. 'District 9 > Retail') "
            f"to resolve generic terms like 'The District' or 'The Sector' to specific names.\n"
            f"2. SPLITTING AGGREGATES: If the fact mentions 'Contacts in a few districts', "
            f"and the context mentions specific districts (e.g. 'Minneapolis', 'Dallas'), "
            f"create SEPARATE FinancialRelation entries for EACH implied entity.\n"
            f"3. SUBJECT/OBJECT TYPES:\n"
            f"   - 'Entity': Specific Companies, People, Locations (e.g. 'Apple', 'Minneapolis District').\n"
            f"   - 'Topic': Concepts acting as agents (e.g. 'Inflation' hurt earnings).\n"
            f"4. TOPICS: Extract key financial concepts alluded to (e.g. 'Inflation', 'Labor Market').\n"
            f"5. DATE CONTEXT: If the text implies a specific time (e.g. 'Q3'), extract it.\n"
            f"6. TABLE ROW PROCESSING (If input is a table row):\n"
            f"   - SCAN EVERY CELL: Check ALL values (including Note columns) for Named Entities (Company, Person, Location, Country). Extract them as ENTITIES.\n"
            f"   - COLUMN CONTEXT: Use the COLUMN HEADER to inform the relationship type (e.g. Header 'Manager' -> Relation 'IS_MANAGER', 'Headquarters' -> 'LOCATED_IN').\n"
            f"   - IGNORE QUANTITATIVE DATA: Do NOT extract raw numbers, prices, or dates as nodes. Leave them in the chunk text as evidence.\n"
            f"   - NUMERIC HEADERS: If a column is purely numeric (e.g. 'Revenue'), extract the Column Name itself as a TOPIC node linked via 'ABOUT'.\n"
        )
        
        try:
            response = self.structured_llm.invoke([("human", prompt)])
            relations = response.relations
            
            # --- REFLEXION STEP ---
            # Check if we should split further or if we missed something obvious
            # Only run if we have relations to check
            if relations:
                relations = self.reflexion_check(fact_text, chunk_text, header_path, relations)
                
            return relations

        except Exception as e:
            print(f"Entity extraction failed for '{fact_text[:20]}...': {e}")
            # Fallback: one generic relation
            return [FinancialRelation(
                subject="Unknown",
                subject_type="Entity",
                object=None
            )]

    def reflexion_check(self, fact_text: str, chunk_text: str, header_path: str, current_relations: List[FinancialRelation]) -> List[FinancialRelation]:
        """
        Reflects on the extraction to ensure no aggregate/ambiguous entities remain.
        """
        # Convert current extraction to string for prompt
        current_summary = "\n".join([f"- Subj: {r.subject} ({r.subject_type}) -> Obj: {r.object}" for r in current_relations])
        
        reflexion_prompt = (
            f"HEADER: {header_path}\n"
            f"CHUNK: \"{chunk_text}\"\n"
            f"FACT: \"{fact_text}\"\n\n"
            
            f"CURRENT EXTRACTION:\n{current_summary}\n\n"
            
            f"CRITIC TASKS:\n"
            f"1. IDENTIFY AGGREGATES: Are there terms like 'Many Districts', 'Several Banks', 'Contacts' "
            f"that are still generic but COULD be resolved to specific names from the Context?\n"
            f"   - Example: 'Contacts reported' -> If context lists 'Retailers' and 'Manufacturers', split into two relations.\n"
            f"2. SPECIFICITY CHECK: Did we resolve 'The District' to the specific district name from the Header?\n"
            f"3. COMPLETENESS: Did we miss any distinct entity mentioned in the fact?\n"
            f"4. TABLE SANITY CHECK: Did we accidentally extract a raw number (e.g. '300,000') as an entity? If so, REMOVE IT.\n\n"
            
            f"If changes are needed, return the IMPROVED list of FinancialRelations."
            f"If the current extraction is already optimal, just return it as is."
        )
        
        try:
            response = self.structured_llm.invoke([("human", reflexion_prompt)])
            if response.relations:
                # Basic check: if count increased, or if values look different, we accept it.
                # For now, trust the Reflector if it returns valid output.
                # print(f"   ✨ Entity Reflexion updated {len(current_relations)} -> {len(response.relations)} relations.")
                return response.relations
        except Exception as e:
            # If reflexion fails, return original
            print(f"   ⚠️ Entity Reflexion failed: {e}")
            
        return current_relations
