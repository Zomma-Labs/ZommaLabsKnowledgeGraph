"""
MODULE: Graph Enhancer
DESCRIPTION: 
    Enhances the graph extraction process by:
    1. Reflexion: Checking for missed facts and promoting Concepts to Entities.
    2. Deduplication: Resolving entities against the existing graph (fallback to FIBO).
    3. Enrichment: Extracting attributes and summaries.
"""

import os
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from pydantic import BaseModel, Field
from src.tools.neo4j_client import Neo4jClient

if TYPE_CHECKING:
    from src.util.services import Services

# Control verbose output
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"

def log(msg: str):
    """Print only if VERBOSE mode is enabled."""
    if VERBOSE:
        print(msg)

class MissedFacts(BaseModel):
    missed_facts: List[str] = Field(description="List of facts that were missed in the initial extraction.")
    reasoning: str = Field(description="Why these facts are important and should be included.")

class EntityResolution(BaseModel):
    decision: str = Field(description="One of: 'MERGE', 'CREATE_NEW'")
    target_uuid: Optional[str] = Field(description="UUID of the existing node to merge with, if MERGE.")
    reasoning: str = Field(description="Reason for the decision.")

class GraphEnhancer:
    def __init__(self, services: Optional["Services"] = None):
        if services is None:
            from src.util.services import get_services
            services = get_services()
        self.llm = services.llm
        self.embeddings = services.embeddings
        self.neo4j = services.neo4j

        # Add cheap model for summaries
        from src.util.llm_client import get_nano_llm
        self.nano_llm = get_nano_llm()

    def reflexion_check(self, chunk_text: str, existing_facts: List[Any]) -> List[str]:
        """
        Asks the LLM if any important facts were missed, specifically looking to promote Concepts to Entities.
        """
        structured_llm = self.llm.with_structured_output(MissedFacts)
        
        # Convert existing facts to string for context
        facts_str = "\n".join([str(f) for f in existing_facts])
        
        system_prompt = (
            "You are a Quality Assurance Auditor for a Knowledge Graph.\n"
            "Your goal is to review the Extracted Facts against the Source Text and identify MISSING information.\n\n"
            "CRITICAL GOAL: PROMOTE CONCEPTS TO ENTITIES.\n"
            "If the text says 'the tech giant' and we extracted it as a generic Concept, but the text implies a specific company, "
            "you MUST flag this as a missed fact: 'The tech giant is [Company Name]'.\n\n"
            "Rules:\n"
            "1. Only report SUBSTANTIAL missing facts that change the meaning or add specific entities.\n"
            "2. Ignore minor wording differences.\n"
            "3. Focus on specific Names, Dates, and Financial Metrics."
        )
        
        prompt = f"SOURCE TEXT:\n{chunk_text}\n\nEXTRACTED FACTS:\n{facts_str}"
        
        try:
            response = structured_llm.invoke([
                ("system", system_prompt),
                ("human", prompt)
            ])
            return response.missed_facts
        except Exception as e:
            print(f"Reflexion failed: {e}")
            return []

    def extract_entity_summary(self, entity_name: str, context_text: str) -> str:
        """
        Generates a comprehensive summary of what the entity IS in this context,
        including all relevant facts like roles, dates, events, and relationships.
        """
        prompt = (
            f"Based on the text below, provide a comprehensive summary of what '{entity_name}' IS.\n"
            f"Include ALL relevant facts from the context:\n"
            f"- For people: role/title, organization, key dates (e.g., tenure start/end), notable actions\n"
            f"- For organizations: type, parent/subsidiary relationships, key events, dates\n"
            f"- For events: what happened, when, who was involved\n"
            f"Keep it factual and specific. 2-4 sentences is fine if needed to capture key details.\n"
            f"TEXT: {context_text}\n"
            f"SUMMARY:"
        )
        try:
            response = self.nano_llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            print(f"Summary extraction failed: {e}")
            return "Entity"

    def batch_extract_summaries(self, entity_names: List[str], context_text: str, batch_size: int = 15) -> Dict[str, str]:
        """
        Extracts summaries for multiple entities in batches using a single LLM call per batch.

        Args:
            entity_names: List of entity names to summarize
            context_text: The chunk text for context
            batch_size: Max entities per batch (default 15)

        Returns:
            Dict mapping entity name to summary
        """
        from pydantic import BaseModel, Field

        # Define structured output for batch summaries (using List instead of Dict for OpenAI compatibility)
        class EntitySummary(BaseModel):
            """Single entity summary."""
            name: str = Field(description="Entity name")
            summary: str = Field(description="Comprehensive summary describing what this entity is, including key facts, dates, and relationships")

        class EntitySummaries(BaseModel):
            """Collection of entity summaries."""
            summaries: List[EntitySummary] = Field(
                description="List of entity summaries"
            )

        all_summaries = {}

        # Split into batches of size 15
        for i in range(0, len(entity_names), batch_size):
            batch = entity_names[i:i + batch_size]

            # Create numbered list for prompt
            entity_list = "\n".join([f"{idx+1}. {name}" for idx, name in enumerate(batch)])

            prompt = (
                f"Given the following context, provide a comprehensive summary for EACH entity.\n"
                f"The summary should describe what the entity IS with ALL relevant facts from the context.\n\n"
                f"Guidelines:\n"
                f"- For people: Include role/title, organization, key dates (tenure start/end), notable actions.\n"
                f"- For organizations: Include type, parent/subsidiary relationships, key events, founding dates.\n"
                f"- For events/documents: Include what it is, when it occurred, key details.\n"
                f"- Be specific - avoid generic phrases like 'a person' or 'a company'.\n"
                f"- 2-4 sentences per entity is fine if needed to capture important details.\n\n"
                f"CONTEXT:\n{context_text}\n\n"
                f"ENTITIES TO SUMMARIZE:\n{entity_list}\n\n"
                f"Provide comprehensive summaries for all entities listed above."
            )

            try:
                structured_nano = self.nano_llm.with_structured_output(EntitySummaries)
                response = structured_nano.invoke(prompt)

                # Convert list to dict
                for entity_summary in response.summaries:
                    all_summaries[entity_summary.name] = entity_summary.summary

                log(f"   ✅ Batch summarized {len(batch)} entities")

            except Exception as e:
                print(f"   ⚠️ Batch summary failed for batch {i//batch_size + 1}: {e}")
                # Fallback: use generic summaries for this batch
                for name in batch:
                    all_summaries[name] = "Entity"

        return all_summaries

    def find_graph_candidates(self, entity_name: str, entity_summary: str, group_id: str, node_type: str = "Entity", top_k: int = 5, cached_embedding: Optional[List[float]] = None) -> List[Dict]:
        """
        Finds candidates using Exact Match (Step 1) and Vector Search (Step 2).
        """
        candidates = []
        seen_uuids = set()

        # 1. Exact Name Match
        label = "EntityNode" if node_type == "Entity" else "TopicNode"
        cypher_exact = f"""
        MATCH (n:{label} {{name: $name, group_id: $group_id}})
        RETURN n.uuid as uuid, n.name as name, n.summary as summary, labels(n) as labels
        LIMIT 1
        """
        # Both TopicNode and EntityNode use 'summary' for descriptions.
        if node_type == "Topic":
             cypher_exact = f"""
            MATCH (n:TopicNode {{group_id: $group_id}})
            WHERE toLower(n.name) = toLower($name)
            RETURN n.uuid as uuid, n.name as name, n.summary as summary, labels(n) as labels
            LIMIT 1
            """
        else:
             cypher_exact = f"""
            MATCH (n:EntityNode {{group_id: $group_id}})
            WHERE toLower(n.name) = toLower($name)
            RETURN n.uuid as uuid, n.name as name, n.summary as summary, labels(n) as labels
            LIMIT 1
            """
        try:
            exact_matches = self.neo4j.query(cypher_exact, {"name": entity_name, "group_id": group_id})
            for match in exact_matches:
                match['score'] = 1.0
                candidates.append(match)
                seen_uuids.add(match['uuid'])
        except Exception as e:
            print(f"Exact match query failed: {e}")

        # 2. Vector Search (Fallback/Supplementary)
        # Embed "Name: Summary"
        query_text = f"{entity_name}: {entity_summary}"
        try:
            if cached_embedding:
                vector = cached_embedding
            else:
                vector = self.embeddings.embed_query(query_text)
            
            index_name = "entity_embeddings" if node_type == "Entity" else "topic_embeddings"
            results = self.neo4j.vector_search(index_name, vector, top_k, filters={"group_id": group_id})
            
            for record in results:
                node = record['node']
                uuid = node.get("uuid") or node.get("uri") # Handle potential schema variations
                
                if uuid not in seen_uuids:
                    candidates.append({
                        "uuid": uuid,
                        "name": node.get("name"),
                        "summary": node.get("summary", ""),
                        "labels": node.get("labels", []),
                        "score": record['score']
                    })
                    seen_uuids.add(uuid)
                    
            return candidates
        except Exception as e:
            print(f"Graph candidate search failed: {e}")
            return candidates # Return whatever we found in exact match

    def resolve_entity_against_graph(self, entity_name: str, entity_summary: str, candidates: List[Dict]) -> Dict[str, Any]:
        """
        Uses LLM to decide whether to merge with a candidate or create a new node.
        Uses OutputFixingParser to retry with format correction if initial parse fails.
        """
        from langchain_core.output_parsers import PydanticOutputParser
        from langchain_classic.output_parsers import OutputFixingParser
        
        if not candidates:
            return {"decision": "CREATE_NEW", "target_uuid": None}
        
        candidates_str = "\n".join([
            f"- ID: {c['uuid']}, Name: {c['name']}, Description: {c.get('summary', 'N/A')}, Score: {c['score']:.2f}" 
            for c in candidates
        ])
        
        system_prompt = (
            "You are an Entity Resolution Expert.\n"
            "Decide if the New Entity matches any of the Existing Graph Candidates.\n\n"
            "Rules:\n"
            "1. MERGE only if you are confident it is the SAME real-world entity.\n"
            "2. Pay close attention to DESCRIPTIONS. 'Apple' (Tech Company) != 'Apple' (Fruit).\n"
            "3. If the name is the same but the description implies a different entity, CREATE_NEW.\n"
            "4. If the candidate list is empty or irrelevant, CREATE_NEW."
        )
        
        prompt = (
            f"NEW ENTITY: {entity_name}\n"
            f"SUMMARY: {entity_summary}\n\n"
            f"EXISTING CANDIDATES:\n{candidates_str}"
        )
        
        # First attempt: use structured output with include_raw
        try:
            structured_llm = self.llm.with_structured_output(EntityResolution, include_raw=True)
            result = structured_llm.invoke([
                ("system", system_prompt),
                ("human", prompt)
            ])
            
            # result is a dict with 'raw', 'parsed', and 'parsing_error' keys
            parsed = result.get("parsed") if isinstance(result, dict) else result
            raw = result.get("raw") if isinstance(result, dict) else None
            parsing_error = result.get("parsing_error") if isinstance(result, dict) else None
            
            if parsed is not None:
                return {
                    "decision": parsed.decision,
                    "target_uuid": parsed.target_uuid,
                    "reasoning": parsed.reasoning
                }
            
            # If parsed is None but we have raw, use OutputFixingParser
            if raw is not None:
                log(f"   🔄 Using OutputFixingParser for '{entity_name}'...")
                raw_content = raw.content if hasattr(raw, 'content') else str(raw)
                
                # Create parser and fixing parser
                pydantic_parser = PydanticOutputParser(pydantic_object=EntityResolution)
                fixing_parser = OutputFixingParser.from_llm(parser=pydantic_parser, llm=self.llm)
                
                fixed_response = fixing_parser.parse(raw_content)
                return {
                    "decision": fixed_response.decision,
                    "target_uuid": fixed_response.target_uuid,
                    "reasoning": fixed_response.reasoning
                }
            
            # Both failed
            print(f"Resolution returned None for '{entity_name}', defaulting to CREATE_NEW")
            return {"decision": "CREATE_NEW", "target_uuid": None}
            
        except Exception as e:
            print(f"Resolution failed for '{entity_name}': {e}")
            return {"decision": "CREATE_NEW", "target_uuid": None}

    def extract_attributes(self, entity_name: str, context_text: str) -> Dict[str, Any]:
        """
        Extracts attributes and summary for an entity from the text.
        """
        # Placeholder for attribute extraction logic
        # For now, we can just return a simple summary
        return {"summary": f"Entity extracted from: {context_text[:50]}..."}

    def merge_summaries(self, entity_name: str, existing_summary: str, new_summary: str) -> str:
        """
        Merges two summaries for the same entity into a single, richer summary.
        Called when an entity is deduplicated and both have summary information.

        Args:
            entity_name: The name of the entity
            existing_summary: The summary already in the graph
            new_summary: The new summary from current extraction

        Returns:
            A merged summary combining information from both
        """
        # Skip if either is empty/generic
        if not existing_summary or existing_summary in ["Entity", "", None]:
            return new_summary
        if not new_summary or new_summary in ["Entity", "", None]:
            return existing_summary
        if existing_summary.lower() == new_summary.lower():
            return existing_summary

        prompt = (
            f"Merge these two descriptions of '{entity_name}' into a single, comprehensive 1-2 sentence summary.\n"
            f"Combine unique information from both. Remove redundancy. Keep it concise.\n\n"
            f"Description 1: {existing_summary}\n"
            f"Description 2: {new_summary}\n\n"
            f"Merged summary:"
        )

        try:
            response = self.nano_llm.invoke(prompt)
            merged = response.content.strip()
            # Sanity check: don't return something much longer than inputs combined
            if len(merged) > len(existing_summary) + len(new_summary) + 50:
                return existing_summary  # Fallback to existing
            return merged
        except Exception as e:
            log(f"   ⚠️ Summary merge failed for '{entity_name}': {e}")
            # Fallback: prefer the longer/more detailed summary
            return existing_summary if len(existing_summary) >= len(new_summary) else new_summary

    def batch_merge_summaries(self, merges: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Batch merge multiple entity summaries in a single LLM call.

        Args:
            merges: List of dicts with 'name', 'existing', 'new' keys

        Returns:
            Dict mapping entity name to merged summary
        """
        from pydantic import BaseModel, Field

        class MergedSummary(BaseModel):
            name: str = Field(description="Entity name")
            summary: str = Field(description="Merged 1-2 sentence summary")

        class MergedSummaries(BaseModel):
            summaries: List[MergedSummary]

        # Filter out trivial cases
        to_merge = []
        results = {}

        for m in merges:
            existing = m.get('existing', '')
            new = m.get('new', '')
            name = m['name']

            if not existing or existing in ["Entity", ""]:
                results[name] = new
            elif not new or new in ["Entity", ""]:
                results[name] = existing
            elif existing.lower() == new.lower():
                results[name] = existing
            else:
                to_merge.append(m)

        if not to_merge:
            return results

        # Build prompt for batch merge
        merge_list = "\n".join([
            f"{i+1}. {m['name']}:\n   A: {m['existing']}\n   B: {m['new']}"
            for i, m in enumerate(to_merge)
        ])

        prompt = (
            f"Merge each pair of descriptions into a single 1-2 sentence summary.\n"
            f"Combine unique information. Remove redundancy. Be concise.\n\n"
            f"{merge_list}\n\n"
            f"Provide merged summaries for all entities."
        )

        try:
            structured_nano = self.nano_llm.with_structured_output(MergedSummaries)
            response = structured_nano.invoke(prompt)

            for merged in response.summaries:
                results[merged.name] = merged.summary

            # Fill in any that weren't returned
            for m in to_merge:
                if m['name'] not in results:
                    results[m['name']] = m['existing']

            log(f"   ✅ Batch merged {len(to_merge)} summaries")

        except Exception as e:
            log(f"   ⚠️ Batch summary merge failed: {e}")
            # Fallback: use existing summaries
            for m in to_merge:
                results[m['name']] = m['existing']

        return results

if __name__ == "__main__":
    # Test
    enhancer = GraphEnhancer()
    log("GraphEnhancer initialized.")
