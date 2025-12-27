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
from src.util.llm_client import get_llm
from src.schemas.financial_relation import FinancialRelation, FinancialRelationList

class EntityExtractor:
    def __init__(self):
        # Use shared LLM client from project config
        self.llm = get_llm()
        # Use include_raw=True to get both parsed output and raw response for error handling
        self.structured_llm = self.llm.with_structured_output(FinancialRelationList, include_raw=True)

    def extract(self, fact_text: str, chunk_text: str, header_path: str = "") -> List[FinancialRelation]:
        """
        Extracts entities from a fact using the chunk context and header path.
        Returns a list of relations to support splitting aggregate entities.
        """
        # === OLD PROMPT (COMMENTED OUT FOR REFERENCE) ===
        # prompt_old = (
        #     f"HEADER: {header_path}\n"
        #     f"CHUNK: \"{chunk_text}\"\n\n"
        #     f"FACT TO ANALYZE: \"{fact_text}\"\n\n"
        #     f"GOAL: Extract the detailed FINANCIAL RELATIONSHIPS from the FACT, "
        #     f"resolving any generic terms using the HEADER and CHUNK context.\n\n"
        #     f"STRICT RULES:\n\n"
        #     f"1. CONTEXT AWARENESS: Use the HEADER (e.g. 'District 9 > Retail') "
        #     f"to resolve generic terms like 'The District' or 'The Sector' to specific names.\n\n"
        #     f"2. SPLITTING AGGREGATES: If the fact mentions 'Contacts in a few districts', "
        #     f"and the context mentions specific districts (e.g. 'Minneapolis', 'Dallas'), "
        #     f"create SEPARATE FinancialRelation entries for EACH implied entity.\n\n"
        #     f"3. LIST EXPANSION - CRITICAL:\n"
        #     f"   When the fact contains an enumerated list of entities, create a SEPARATE relation for EACH item:\n"
        #     f"   - Example: 'Alphabet subsidiaries include Google, Waymo, Verily, and DeepMind'\n"
        #     f"     Must produce 4 separate relations:\n"
        #     f"     * Subject=Alphabet Inc., Object=Google\n"
        #     f"     * Subject=Alphabet Inc., Object=Waymo\n"
        #     f"     * Subject=Alphabet Inc., Object=Verily\n"
        #     f"     * Subject=Alphabet Inc., Object=DeepMind\n"
        #     f"   - Example: 'Shareholders include Vanguard (7.25%), BlackRock (6.27%), and State Street (3.36%)'\n"
        #     f"     Must produce 3 separate relations, one for each shareholder.\n"
        #     f"   - NEVER combine multiple entities into a single object like 'Google, Waymo, Verily'\n\n"
        #     f"4. SUBJECT/OBJECT TYPES:\n"
        #     f"   - 'Entity': Specific Companies, People, Locations (e.g. 'Apple', 'Minneapolis District').\n"
        #     f"   - 'Topic': Concepts acting as agents (e.g. 'Inflation' hurt earnings).\n\n"
        #     f"5. TOPICS: Extract key financial concepts alluded to (e.g. 'Inflation', 'Labor Market').\n\n"
        #     f"6. RELATIONSHIP DESCRIPTION:\n"
        #     f"   Provide a short phrase (2-5 words) describing the action between subject and object.\n"
        #     f"   Examples: 'acquired majority stake in', 'filed antitrust lawsuit against', 'appointed as CEO of',\n"
        #     f"   'structure modeled after', 'raised Series B from', 'settled privacy lawsuit with'.\n\n"
        #     f"7. DATE CONTEXT - PRESERVE EXACTLY:\n"
        #     f"   - Full dates: 'January 16, 2020', 'September 1, 2017'\n"
        #     f"   - Month/Year: 'October 2020', 'April 2024'\n"
        #     f"   - Quarters: 'Q3 2023'\n"
        #     f"   - NEVER approximate: 'January 16, 2020' is NOT the same as '2020'\n\n"
        #     f"8. TABLE ROW PROCESSING (If input is a table row):\n"
        #     f"   - SCAN EVERY CELL for Named Entities. Extract them as ENTITIES.\n"
        #     f"   - IGNORE QUANTITATIVE DATA as nodes. Leave them in the chunk text as evidence.\n\n"
        #     f"9. RELATIONSHIP DIRECTION - CRITICAL:\n"
        #     f"   For EMPLOYMENT/HIRING relationships (CEO, CFO, hired, leads, manages):\n"
        #     f"   - Subject = the EMPLOYER (the company/organization doing the hiring)\n"
        #     f"   - Object = the EMPLOYEE (the person being hired/holding the role)\n"
        #     f"   - Example: 'Sundar Pichai is CEO of Alphabet' → Subject: Alphabet, Object: Sundar Pichai\n"
        #     f"   For SUBSIDIARY/OWNERSHIP relationships:\n"
        #     f"   - Subject = the PARENT company (the one that owns/established)\n"
        #     f"   - Object = the SUBSIDIARY (the one being owned/established)\n"
        #     f"   - Example: 'Waymo is a subsidiary of Alphabet' → Subject: Alphabet, Object: Waymo\n\n"
        #     f"10. ATTRIBUTION VERIFICATION - CRITICAL:\n"
        #     f"   Before assigning subject/object, verify WHO performed the action using the CHUNK context.\n"
        #     f"   - If the FACT says 'X revealed Y', check the CHUNK to confirm X is the correct person.\n"
        #     f"   - Common error: Attributing an action to a well-known founder when a different executive did it.\n"
        # )
        # === END OLD PROMPT ===

        # === NEW STRUCTURED PROMPT ===
        prompt = (
            f"You are a financial analyst extracting entities and relationships for a knowledge graph.\n\n"

            f"HEADER: {header_path}\n"
            f"CHUNK: \"{chunk_text}\"\n\n"
            f"FACT TO ANALYZE: \"{fact_text}\"\n\n"

            f"GOAL: Extract relationships between entities that someone in finance would reasonably search for.\n\n"

            # ===== THINKING REQUIREMENT =====
            f"=== THINKING REQUIREMENT ===\n\n"
            f"Before extracting each relationship, use the 'thinking' field to reason:\n"
            f"- 'Would a financial analyst search for this entity?'\n"
            f"- 'Is this a proper noun with a real identity (company, person, law, place)?'\n"
            f"- 'Or is this just a generic description?'\n\n"
            f"Only extract relationships between valid, searchable entities.\n\n"

            # ===== CRITICAL: SEMANTIC ROLE ASSIGNMENT =====
            f"=== CRITICAL: SEMANTIC ROLE ASSIGNMENT ===\n\n"
            f"Subject and Object must be assigned based on SEMANTIC ROLES, not word order:\n\n"
            f"- SUBJECT = the AGENT (who performs, initiates, or causes the action)\n"
            f"- OBJECT = the PATIENT (who receives, undergoes, or is affected by the action)\n\n"
            f"PASSIVE VOICE WARNING:\n"
            f"In passive constructions, the grammatical subject is often the semantic PATIENT.\n"
            f"You must identify the true AGENT regardless of word order.\n\n"
            f"Ask yourself: 'Who is DOING the action to whom?'\n"
            f"The doer = Subject. The receiver = Object.\n\n"

            # ===== SECTION A: ENTITY RESOLUTION =====
            f"=== A. ENTITY RESOLUTION ===\n\n"
            f"A1. RESOLVE GENERIC TERMS: Use HEADER and CHUNK context to resolve ambiguous references.\n"
            f"    - 'The District' → specific district name from Header\n"
            f"    - 'The company' → actual company name from Chunk\n"
            f"    - 'Several banks' → list specific banks if mentioned in context\n\n"
            f"A2. ENTITY TYPES:\n"
            f"    - 'Entity': Specific actors - Companies, People, Locations, Products\n"
            f"    - 'Topic': Abstract concepts acting as agents (e.g., 'Inflation' caused losses)\n\n"
            f"A3. ENTITY NAME LENGTH:\n"
            f"    - Entity names should be 1-3 words (proper nouns only)\n"
            f"    - Longer phrases are likely FACTS or EVENTS - split them into Entity + Topic + Relationship\n"
            f"    - Extract the core named entity, not descriptive phrases around it\n\n"
            f"A4. FILTER OUT NON-ENTITIES:\n"
            f"    - QUANTITATIVE DATA (numbers, dollar amounts, percentages, metrics) are EVIDENCE, not entities or topics\n"
            f"    - Numeric values belong in the chunk text as supporting evidence, not as extracted nodes\n"
            f"    - If a value represents a concept, extract the CONCEPT as a topic (not the value itself)\n"
            f"    - Generic phrases ('strong CEOs', 'many subsidiaries') are NOT entities unless resolved\n"
            f"    - Titles/roles alone are NOT entities - extract the person's actual name\n\n"
            f"A5. TABLE PROCESSING: If the fact comes from a table row:\n"
            f"    - Extract named entities from cells (companies, people, locations)\n"
            f"    - Ignore pure numeric data - leave it in the chunk as evidence\n"
            f"    - Use column headers (from HEADER) to understand relationships\n\n"

            # ===== SECTION B: RELATIONSHIP EXTRACTION =====
            f"=== B. RELATIONSHIP EXTRACTION ===\n\n"
            f"B1. RELATIONSHIP DESCRIPTION: Short verb phrase (2-5 words) describing the action.\n"
            f"    Examples: 'acquired', 'partnered with', 'appointed as CEO of', 'sued'\n\n"
            f"B2. RELATIONSHIP DIRECTION - Follow semantic conventions:\n"
            f"    - EMPLOYMENT: Employer → Employee (Company hired Person)\n"
            f"    - OWNERSHIP: Parent → Subsidiary (Parent Co owns Subsidiary Co)\n"
            f"    - INVESTMENT: Investor → Investee (VC invested in Startup)\n"
            f"    - LEGAL: Plaintiff → Defendant (Company A sued Company B)\n\n"
            f"B3. DATE CONTEXT: Preserve EXACT dates from the fact.\n"
            f"    - 'January 16, 2020' stays as 'January 16, 2020', NOT '2020'\n"
            f"    - 'Q3 2023' stays as 'Q3 2023'\n\n"
            f"B4. TOPICS: Extract financial/economic concepts mentioned (e.g., 'Corporate Structure', 'M&A').\n\n"

            # ===== SECTION C: MULTI-RELATION PATTERNS =====
            f"=== C. MULTI-RELATION PATTERNS ===\n"
            f"A single fact can produce MULTIPLE relations. Look for these patterns:\n\n"
            f"C1. LIST EXPANSION: Enumerated items become separate relations.\n"
            f"    - 'Major investors include Vanguard, BlackRock, and Fidelity' → 3 relations\n"
            f"    - Each list item gets its own Subject-Object-Description entry\n\n"
            f"C2. SOURCE ATTRIBUTION: When someone communicates information about other entities,\n"
            f"    extract BOTH the content AND the source.\n"
            f"    - The CONTENT relationship: what the information is about\n"
            f"    - The SOURCE relationship: who provided/communicated it\n"
            f"    Example: 'The researcher found that Drug X treats Disease Y'\n"
            f"      → Relation 1: Drug X --[treats]--> Disease Y\n"
            f"      → Relation 2: Researcher --[discovered]--> Drug X\n"
            f"    The speaker/source should be connected to the primary entity they discussed.\n\n"
            f"C3. AGGREGATE SPLITTING: 'Contacts in a few districts' with context listing\n"
            f"    specific districts → separate relation per district.\n\n"

            # ===== SECTION D: ENTITY DEFINITIONS =====
            f"=== D. ENTITY DEFINITIONS ===\n\n"
            f"For EACH entity (subject and object), provide a 1-2 sentence 'summary' defining what it IS:\n"
            f"- For PEOPLE: Include their role/title and organization (e.g., 'CEO of Apple Inc.')\n"
            f"- For COMPANIES: Include industry and what they do (e.g., 'Multinational technology company')\n"
            f"- For TOPICS: Define the concept in financial terms (e.g., 'The rate at which prices increase')\n"
            f"- For LOCATIONS: Geographic and economic context (e.g., 'Federal Reserve district covering Texas')\n"
            f"Use information from the CHUNK to make definitions specific and accurate.\n\n"

            # ===== SECTION E: VALIDATION =====
            f"=== E. VALIDATION ===\n\n"
            f"E1. ATTRIBUTION CHECK: Verify the correct person/entity is assigned as subject.\n"
            f"    - Cross-reference with CHUNK to confirm WHO performed the action.\n"
            f"    - Don't assume well-known founders - check the actual text.\n\n"
            f"E2. COMPLETENESS CHECK: Did you extract ALL relationships from the fact?\n"
            f"    - All list items expanded?\n"
            f"    - Source/speaker relationship captured?\n"
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

    def reflexion_check(self, chunk_text: str, current_relations: List[FinancialRelation]) -> Optional[str]:
        """
        Reviews extracted entities and returns a critique if any invalid entities are found.

        Returns:
            None if all entities are valid (approved)
            A critique string describing what needs to be fixed
        """
        # Build list of unique entities for review
        entities = set()
        for r in current_relations:
            entities.add((r.subject, r.subject_type))
            if r.object:
                entities.add((r.object, r.object_type))

        if not entities:
            return None  # Nothing to review

        entity_summary = "\n".join([f"- {name} (type: {etype})" for name, etype in entities])

        critique_prompt = (
            f"You are a quality control agent reviewing entity extractions for a financial knowledge graph.\n\n"
            f"CHUNK: \"{chunk_text}\"\n\n"
            f"EXTRACTED ENTITIES:\n{entity_summary}\n\n"

            f"=== YOUR TASK ===\n"
            f"Review each extracted entity and determine if it's a VALID, SEARCHABLE entity.\n\n"

            f"A VALID entity is:\n"
            f"- A proper noun with a real identity (company name, person name, law/regulation, place)\n"
            f"- Something a financial analyst would look up or search for\n"
            f"- Specific and named, not generic or descriptive\n\n"

            f"An INVALID entity is:\n"
            f"- A generic description that isn't a proper noun\n"
            f"- A phrase that describes an event or action rather than a named actor\n"
            f"- Something that only makes sense in the context of the original text\n\n"

            f"=== RESPONSE FORMAT ===\n"
            f"If ALL entities are valid: respond with exactly 'APPROVED'\n\n"
            f"If ANY entity is invalid: provide a specific critique explaining:\n"
            f"1. Which entity/entities are problematic\n"
            f"2. WHY they are invalid\n"
            f"3. What the extraction agent should do instead\n\n"
            f"Be specific so the extraction agent can fix its output."
        )

        try:
            response = self.llm.invoke([("human", critique_prompt)])
            critique = response.content.strip()

            # Check if approved
            if critique.upper() == "APPROVED" or critique.lower().startswith("approved"):
                return None

            return critique

        except Exception as e:
            print(f"   ⚠️ Entity reflexion check failed: {e}")
            return None  # On error, don't block - just approve

    def extract_with_reflexion(self, fact_text: str, chunk_text: str, header_path: str = "", max_iterations: int = 2) -> List[FinancialRelation]:
        """
        Extracts entities with a reflexion loop that critiques and refines the output.

        1. Extract initial relations
        2. Send to reflexion agent for critique
        3. If critique found, re-extract with critique feedback
        4. Return final relations
        """
        # Initial extraction
        relations = self.extract(fact_text, chunk_text, header_path)

        if not relations:
            return []

        # Reflexion loop
        for iteration in range(max_iterations):
            critique = self.reflexion_check(chunk_text, relations)

            if critique is None:
                # Approved - no issues found
                break

            # Re-extract with critique feedback, passing previous relations for context
            relations = self._extract_with_critique(fact_text, chunk_text, header_path, critique, relations)

            if not relations:
                break

        return relations

    def _extract_with_critique(self, fact_text: str, chunk_text: str, header_path: str, critique: str, previous_relations: List[FinancialRelation]) -> List[FinancialRelation]:
        """
        Re-extracts entities after receiving a critique from the reflexion agent.
        Receives the atomic fact, previous extraction, and critique to inform the fix.
        """
        # Build summary of previous extraction
        prev_summary = "\n".join([
            f"- Subject: {r.subject} ({r.subject_type}) -> Object: {r.object} ({r.object_type})"
            for r in previous_relations
        ])

        # Build the prompt with full context
        prompt = (
            f"You are a financial analyst extracting entities and relationships for a knowledge graph.\n\n"

            f"HEADER: {header_path}\n"
            f"CHUNK: \"{chunk_text}\"\n\n"
            f"ATOMIC FACT: \"{fact_text}\"\n\n"

            f"GOAL: Extract relationships between entities that someone in finance would reasonably search for.\n\n"

            f"=== YOUR PREVIOUS EXTRACTION FROM THIS FACT ===\n"
            f"{prev_summary}\n\n"

            f"=== CRITIQUE FROM REVIEWER ===\n"
            f"A quality control agent reviewed your extraction and found issues:\n\n"
            f"{critique}\n\n"
            f"Please fix these issues in your new extraction.\n\n"

            # ===== THINKING REQUIREMENT =====
            f"=== THINKING REQUIREMENT ===\n\n"
            f"Before extracting each relationship, use the 'thinking' field to reason:\n"
            f"- 'Would a financial analyst search for this entity?'\n"
            f"- 'Is this a proper noun with a real identity (company, person, law, place)?'\n"
            f"- 'Or is this just a generic description?'\n\n"
            f"Only extract relationships between valid, searchable entities.\n\n"

            # ===== CRITICAL: SEMANTIC ROLE ASSIGNMENT =====
            f"=== CRITICAL: SEMANTIC ROLE ASSIGNMENT ===\n\n"
            f"Subject and Object must be assigned based on SEMANTIC ROLES, not word order:\n\n"
            f"- SUBJECT = the AGENT (who performs, initiates, or causes the action)\n"
            f"- OBJECT = the PATIENT (who receives, undergoes, or is affected by the action)\n\n"

            # ===== ENTITY RESOLUTION =====
            f"=== ENTITY RESOLUTION ===\n\n"
            f"- Resolve generic terms using HEADER and CHUNK context\n"
            f"- Entity types: 'Entity' for specific actors, 'Topic' for abstract concepts\n"
            f"- Entity names should be 1-3 words (proper nouns only)\n"
            f"- Filter out non-entities: numbers, generic descriptions, titles without names\n\n"

            # ===== ENTITY DEFINITIONS =====
            f"=== ENTITY DEFINITIONS ===\n\n"
            f"For EACH entity, provide a 1-2 sentence 'summary' defining what it IS:\n"
            f"- For PEOPLE: role/title and organization\n"
            f"- For COMPANIES: industry and what they do\n"
            f"- For TOPICS: financial definition\n"
        )

        try:
            result = self.structured_llm.invoke([("human", prompt)])
            parsed = result.get("parsed") if isinstance(result, dict) else result

            if parsed and parsed.relations:
                return parsed.relations
        except Exception as e:
            print(f"   ⚠️ Re-extraction with critique failed: {e}")

        return []
