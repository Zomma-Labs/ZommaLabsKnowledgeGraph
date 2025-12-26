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
from langchain_google_genai import ChatGoogleGenerativeAI
from src.schemas.financial_relation import FinancialRelation, FinancialRelationList

class EntityExtractor:
    def __init__(self):
        # Use Gemini 2.5 Flash Lite for faster, cheaper entity extraction
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
        # Use include_raw=True to get both parsed output and raw response for error handling
        self.structured_llm = self.llm.with_structured_output(FinancialRelationList, include_raw=True)

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

            f"STRICT RULES:\n\n"
            f"1. CONTEXT AWARENESS: Use the HEADER (e.g. 'District 9 > Retail') "
            f"to resolve generic terms like 'The District' or 'The Sector' to specific names.\n\n"
            f"2. SPLITTING AGGREGATES: If the fact mentions 'Contacts in a few districts', "
            f"and the context mentions specific districts (e.g. 'Minneapolis', 'Dallas'), "
            f"create SEPARATE FinancialRelation entries for EACH implied entity.\n\n"
            f"3. LIST EXPANSION - CRITICAL:\n"
            f"   When the fact contains an enumerated list of entities, create a SEPARATE relation for EACH item:\n"
            f"   - Example: 'Alphabet subsidiaries include Google, Waymo, Verily, and DeepMind'\n"
            f"     Must produce 4 separate relations:\n"
            f"     * Subject=Alphabet Inc., Object=Google\n"
            f"     * Subject=Alphabet Inc., Object=Waymo\n"
            f"     * Subject=Alphabet Inc., Object=Verily\n"
            f"     * Subject=Alphabet Inc., Object=DeepMind\n"
            f"   - Example: 'Shareholders include Vanguard (7.25%), BlackRock (6.27%), and State Street (3.36%)'\n"
            f"     Must produce 3 separate relations, one for each shareholder.\n"
            f"   - NEVER combine multiple entities into a single object like 'Google, Waymo, Verily'\n\n"
            f"4. SUBJECT/OBJECT TYPES:\n"
            f"   - 'Entity': Specific Companies, People, Locations (e.g. 'Apple', 'Minneapolis District').\n"
            f"   - 'Topic': Concepts acting as agents (e.g. 'Inflation' hurt earnings).\n\n"
            f"5. TOPICS: Extract key financial concepts alluded to (e.g. 'Inflation', 'Labor Market').\n\n"
            f"6. RELATIONSHIP DESCRIPTION:\n"
            f"   Provide a short phrase (2-5 words) describing the action between subject and object.\n"
            f"   Examples: 'acquired majority stake in', 'filed antitrust lawsuit against', 'appointed as CEO of',\n"
            f"   'structure modeled after', 'raised Series B from', 'settled privacy lawsuit with'.\n\n"
            f"7. DATE CONTEXT - PRESERVE EXACTLY:\n"
            f"   - Full dates: 'January 16, 2020', 'September 1, 2017'\n"
            f"   - Month/Year: 'October 2020', 'April 2024'\n"
            f"   - Quarters: 'Q3 2023'\n"
            f"   - NEVER approximate: 'January 16, 2020' is NOT the same as '2020'\n\n"
            f"8. TABLE ROW PROCESSING (If input is a table row):\n"
            f"   - SCAN EVERY CELL for Named Entities. Extract them as ENTITIES.\n"
            f"   - IGNORE QUANTITATIVE DATA as nodes. Leave them in the chunk text as evidence.\n\n"
            f"9. RELATIONSHIP DIRECTION - CRITICAL:\n"
            f"   For EMPLOYMENT/HIRING relationships (CEO, CFO, hired, leads, manages):\n"
            f"   - Subject = the EMPLOYER (the company/organization doing the hiring)\n"
            f"   - Object = the EMPLOYEE (the person being hired/holding the role)\n"
            f"   - Example: 'Sundar Pichai is CEO of Alphabet' → Subject: Alphabet, Object: Sundar Pichai\n"
            f"   For SUBSIDIARY/OWNERSHIP relationships:\n"
            f"   - Subject = the PARENT company (the one that owns/established)\n"
            f"   - Object = the SUBSIDIARY (the one being owned/established)\n"
            f"   - Example: 'Waymo is a subsidiary of Alphabet' → Subject: Alphabet, Object: Waymo\n\n"
            f"10. ATTRIBUTION VERIFICATION - CRITICAL:\n"
            f"   Before assigning subject/object, verify WHO performed the action using the CHUNK context.\n"
            f"   - If the FACT says 'X revealed Y', check the CHUNK to confirm X is the correct person.\n"
            f"   - Common error: Attributing an action to a well-known founder when a different executive did it.\n"
        )
        
        max_retries = 2
        last_error = None

        for attempt in range(max_retries):
            try:
                # Response format: {"raw": AIMessage, "parsed": FinancialRelationList or None}
                result = self.structured_llm.invoke([("human", prompt)])

                parsed = result.get("parsed") if isinstance(result, dict) else result
                raw = result.get("raw") if isinstance(result, dict) else None

                # Check if parsing succeeded
                if parsed is None:
                    raw_content = raw.content if raw else "No raw response"
                    if attempt < max_retries - 1:
                        # Retry with schema reminder
                        prompt = self._add_schema_reminder(prompt, raw_content)
                        continue
                    else:
                        raise ValueError(f"Structured output parsing failed. Raw: {raw_content[:200]}")

                relations = parsed.relations

                if not relations:
                    if attempt < max_retries - 1:
                        prompt = self._add_schema_reminder(prompt, "Empty relations list")
                        continue
                    # Empty is valid - just means no entities found
                    return []

                return relations

            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    prompt = self._add_schema_reminder(prompt, str(e))
                    continue

        # All retries failed
        print(f"Entity extraction failed for '{fact_text[:30]}...': {last_error}")
        return []  # Return empty list instead of "Unknown" noise

    def _add_schema_reminder(self, prompt: str, error_context: str) -> str:
        """Add explicit schema reminder to prompt for retry."""
        return prompt + (
            f"\n\n⚠️ PREVIOUS ATTEMPT FAILED: {error_context[:100]}\n"
            "You MUST return JSON with EXACTLY these field names:\n"
            '{"relations": [{"subject": "...", "subject_type": "Entity", '
            '"object": "...", "object_type": "Entity", "topics": [...]}]}\n'
            "DO NOT use 'ENTITY', 'ENTITY_2', 'RELATIONSHIP' - use 'subject', 'object' exactly."
        )

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

            f"CRITIC TASKS:\n\n"
            f"1. LIST COMPLETENESS CHECK - CRITICAL:\n"
            f"   - Does the FACT or CHUNK contain a list of entities (subsidiaries, shareholders, products)?\n"
            f"   - Did we create SEPARATE relations for EACH item in that list?\n"
            f"   - If the CHUNK lists 10 subsidiaries but we only extracted 2, we MUST expand to all 10.\n"
            f"   - Count the items in any list and ensure we have that many relations.\n\n"
            f"2. ATTRIBUTION VERIFICATION - CRITICAL:\n"
            f"   - For each relation, verify the SUBJECT is the correct entity performing the action.\n"
            f"   - Cross-reference with the CHUNK. If CHUNK says 'Eric Schmidt revealed X' but we extracted 'Larry Page', that's WRONG.\n"
            f"   - Look for the actual person/entity name in the CHUNK that matches the action.\n\n"
            f"3. IDENTIFY AGGREGATES:\n"
            f"   - Are there terms like 'Many Districts', 'Several Banks' that could be resolved to specific names?\n\n"
            f"4. SPECIFICITY CHECK:\n"
            f"   - Did we resolve 'The District' to the specific district name from the Header?\n\n"
            f"5. DATE PRECISION CHECK:\n"
            f"   - If the CHUNK contains a specific date like 'January 16, 2020', is it in date_context?\n"
            f"   - We should NOT have '2020' if the CHUNK says 'January 16, 2020'.\n\n"
            f"6. TABLE SANITY CHECK:\n"
            f"   - Did we accidentally extract a raw number (e.g. '300,000') as an entity? REMOVE IT.\n\n"

            f"If changes are needed, return the IMPROVED list of FinancialRelations.\n"
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
