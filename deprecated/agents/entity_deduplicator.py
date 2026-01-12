"""
MODULE: Entity Deduplicator
DESCRIPTION: LLM-based entity deduplication across extraction results.

Uses a two-pass algorithm to ensure cross-batch consistency:
1. Process entities in parallel batches
2. Merge canonical names across batches

Output format is compact grouped JSON to save tokens:
{"Apple Inc.": ["Apple", "AAPL"]} instead of full mapping
"""

import asyncio
import json
import os
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from src.schemas.extraction import EntityDedupeResult

VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"


def log(msg: str):
    if VERBOSE:
        print(f"[EntityDedup] {msg}")


DEDUPE_SYSTEM_PROMPT = """You are an expert in financial entity deduplication for a knowledge graph.

TASK: Group entity names that refer to THE SAME real-world entity.
THIS IS NOT GROUPING SIMILAR OR RELATED ENTITIES.
THIS IS GROUPING **IDENTICAL** ENTITIES.

If you are unsure, err on the side of caution and do not merge.

Each entity includes a description - USE IT to understand what the entity refers to.

ENTITY TYPES (cannot merge across types):
- PERSON: Individuals (executives, analysts, politicians)
- ORGANIZATION: Companies, funds, agencies, institutions
- INDEX: Market indices and benchmarks
- CURRENCY: Fiat and crypto currencies
- COMMODITY: Physical goods (oil, metals, agriculture)

MERGE when names are aliases for the SAME entity:
- Legal name ↔ trade name
- Ticker symbol ↔ company name
- Formal name ↔ nickname
- Full name ↔ abbreviation

NEVER MERGE different entities, even if related:
- People with Company they lead
- Competitors in same industry
- Parent != distinct subsidiary
- Index != constituents of that index

DECISION TEST:
- "Would a Bloomberg terminal treat these as the same security/entity?"
- "If I swapped one name for the other in a sentence, would the meaning stay the same?"
- "If terms/entities would be distinct in speech, they should be distinct here"
- Think about what kind of thing each entity refers to. (e.g their entity type)

Be meticulous in your thinking.

OUTPUT FORMAT (JSON):
{
  "groups": [
    {
      "reasoning": "<why these are the same entity - think step by step>",
      "entity_type": "<PERSON|ORGANIZATION|INDEX|CURRENCY|COMMODITY>",
      "canonical": "<most formal/complete name>",
      "members": ["<other names, not including canonical>"]
    }
  ]
}

Reason BEFORE grouping.
THINK twice before merging something. We would rather have more groups then wrong merges.
Omit entities with no duplicates.
"""


REFLECTION_SYSTEM_PROMPT = """Review entity deduplication results for a financial knowledge graph.

TASK: Group entity names that refer to THE SAME real-world entity.

THIS IS **NOT** GROUPING SIMILAR OR RELATED ENTITIES.
THIS IS GROUPING **IDENTICAL** ENTITIES.

If you are unsure, err on the side of caution and do not merge.

Each entity includes a description - USE IT to understand what the entity refers to.

ENTITY TYPES (cannot merge across types):
- PERSON: Individuals (executives, analysts, politicians)
- ORGANIZATION: Companies, funds, agencies, institutions
- INDEX: Market indices and benchmarks
- CURRENCY: Fiat and crypto currencies
- COMMODITY: Physical goods (oil, metals, agriculture)

MERGE when names are aliases for the SAME entity:
- Legal name ↔ trade name
- Ticker symbol ↔ company name
- Formal name ↔ nickname
- Full name ↔ abbreviation

NEVER MERGE different entities, even if related:
- People with Company they lead
- Competitors in same industry
- Parent != distinct subsidiary
- Index != constituents of that index

DECISION TEST:
- "Would a Bloomberg terminal treat these as the same security/entity?"
- "If I swapped one name for the other in a sentence, would the meaning stay the same?"
- "If terms/entities would be distinct in speech, they should be distinct here"
- Think about what kind of thing each entity refers to. (e.g their entity type)

Be meticulous in your thinking and analysis.

CHECK FOR ERRORS:
1. CROSS-TYPE MERGES: Groups mixing persons with organizations?
2. COMPETITOR MERGES: Groups containing distinct market participants?
3. MISSED ALIASES: Obvious same-entity pairs left ungrouped?

For each error:
- Type: "split" or "merge"
- Entities involved
- Brief reasoning

THINK twice before merging something. We would rather have more groups then wrong merges.
"""


REFLECTION_USER_PROMPT = """Review these groupings for errors:

ENTITY DEFINITIONS (use these to understand what each entity refers to):
{definitions_text}

CURRENT GROUPS:
{groups_text}

UNGROUPED ENTITIES:
{ungrouped_text}

List any cross-type merges, competitor merges, or missed aliases:"""


class ReflectionResult(BaseModel):
    """LLM output for reflection on deduplication."""
    splits: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Wrong merges to fix. Key is the canonical that was wrong, value is list of entities to split out as separate."
    )
    merges: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Missed merges to fix. Key is the canonical to merge INTO, value is list of entities/canonicals to merge."
    )


def groups_to_mapping(groups: List, all_names: List[str]) -> Dict[str, str]:
    """Convert EntityGroup list to full name->canonical mapping."""
    mapping = {}
    for group in groups:
        # Handle both EntityGroup objects and dicts
        if hasattr(group, 'canonical'):
            canonical = group.canonical
            members = group.members
        else:
            canonical = group.get('canonical', '')
            members = group.get('members', [])

        mapping[canonical] = canonical
        for member in members:
            mapping[member] = canonical

    # Names not in any group map to themselves
    for name in all_names:
        if name not in mapping:
            mapping[name] = name
    return mapping


def mapping_to_groups(mapping: Dict[str, str]) -> tuple[Dict[str, List[str]], List[str]]:
    """Convert mapping back to groups format for reflection."""
    groups: Dict[str, List[str]] = {}
    ungrouped: List[str] = []

    for entity, canonical in mapping.items():
        if entity == canonical:
            # Check if this canonical has any aliases
            aliases = [e for e, c in mapping.items() if c == canonical and e != canonical]
            if aliases:
                groups[canonical] = aliases
            else:
                ungrouped.append(entity)
        # Aliases are already captured when we process their canonical

    return groups, ungrouped


def apply_reflection(mapping: Dict[str, str], reflection: "ReflectionResult") -> Dict[str, str]:
    """Apply reflection corrections to mapping."""
    new_mapping = mapping.copy()

    # Apply splits: entities that were wrongly merged
    for wrong_canonical, entities_to_split in reflection.splits.items():
        for entity in entities_to_split:
            if entity in new_mapping:
                # Split out - entity becomes its own canonical
                new_mapping[entity] = entity
                log(f"  Split: '{entity}' from '{wrong_canonical}'")

    # Apply merges: entities that should be together
    for target_canonical, entities_to_merge in reflection.merges.items():
        for entity in entities_to_merge:
            if entity in new_mapping:
                old_canonical = new_mapping[entity]
                new_mapping[entity] = target_canonical
                log(f"  Merge: '{entity}' -> '{target_canonical}' (was: '{old_canonical}')")

                # Also update anything that pointed to entity as canonical
                for e, c in list(new_mapping.items()):
                    if c == entity:
                        new_mapping[e] = target_canonical

    return new_mapping


async def reflect_on_dedup(
    mapping: Dict[str, str],
    llm,
    entities: List[Dict[str, str]]
) -> Dict[str, str]:
    """Run reflection pass to catch and fix errors."""
    groups, ungrouped = mapping_to_groups(mapping)

    if not groups and len(ungrouped) <= 5:
        # Nothing interesting to reflect on
        return mapping

    # Format definitions for prompt
    definitions_lines = []
    for e in entities:
        if e.get("summary"):
            definitions_lines.append(f"- {e['name']}: {e['summary']}")
    definitions_text = "\n".join(definitions_lines) if definitions_lines else "(none provided)"

    # Format groups and ungrouped
    groups_text = json.dumps(groups, indent=2) if groups else "(none)"
    ungrouped_text = json.dumps(ungrouped[:50], indent=2) if ungrouped else "(none)"
    if len(ungrouped) > 50:
        ungrouped_text += f"\n... and {len(ungrouped) - 50} more"

    user_prompt = REFLECTION_USER_PROMPT.format(
        definitions_text=definitions_text,
        groups_text=groups_text,
        ungrouped_text=ungrouped_text
    )
    messages = [("system", REFLECTION_SYSTEM_PROMPT), ("human", user_prompt)]

    try:
        structured_llm = llm.with_structured_output(ReflectionResult)
        result = await asyncio.to_thread(structured_llm.invoke, messages)

        num_splits = sum(len(v) for v in result.splits.values())
        num_merges = sum(len(v) for v in result.merges.values())
        log(f"Reflection: {num_splits} splits, {num_merges} merges")

        if num_splits > 0 or num_merges > 0:
            return apply_reflection(mapping, result)
        return mapping

    except Exception as e:
        print(f"[EntityDedup] Reflection error: {e}")
        return mapping


def _format_entities_for_prompt(entities: List[Dict[str, str]]) -> str:
    """Format entities with their definitions for the LLM prompt."""
    lines = []
    for e in entities:
        name = e.get("name", "")
        summary = e.get("summary", "")
        if summary:
            lines.append(f"- {name}: {summary}")
        else:
            lines.append(f"- {name}")
    return "\n".join(lines)


async def _process_batch(
    batch: List[Dict[str, str]],
    batch_idx: int,
    structured_llm,
    semaphore: asyncio.Semaphore
) -> List:
    """Process a single batch of entities. Returns list of EntityGroup."""
    async with semaphore:
        formatted = _format_entities_for_prompt(batch)
        user_prompt = f"Entities to deduplicate:\n{formatted}"
        messages = [("system", DEDUPE_SYSTEM_PROMPT), ("human", user_prompt)]

        try:
            result = await asyncio.to_thread(structured_llm.invoke, messages)
            log(f"  Batch {batch_idx}: {len(result.groups)} groups")
            return result.groups
        except Exception as e:
            print(f"[EntityDedup] LLM error on batch {batch_idx}: {e}")
            return []


async def dedupe_entities_with_llm_async(
    entities: List[Dict[str, str]],
    llm,
    batch_size: int = 100,
    max_concurrent: int = 10,
    use_reflection: bool = True
) -> Dict[str, str]:
    """
    Use LLM to deduplicate entities with parallel batches.

    Algorithm:
    1. Process entities in parallel batches
    2. Each batch produces groups {canonical: [aliases]}
    3. Final pass: merge canonical names across batches
    4. (Optional) Reflection pass to catch and fix errors

    Args:
        entities: List of {"name": str, "summary": str} dicts
        llm: LLM client (will use .with_structured_output)
        batch_size: Max entities per LLM call
        max_concurrent: Max parallel LLM calls
        use_reflection: Whether to run reflection pass (default: True)

    Returns:
        Dict mapping each entity name to its canonical form
    """
    if not entities:
        return {}

    entity_names = [e["name"] for e in entities]
    structured_llm = llm.with_structured_output(EntityDedupeResult)
    semaphore = asyncio.Semaphore(max_concurrent)

    # PASS 1: Process batches in parallel
    batches = [entities[i:i + batch_size] for i in range(0, len(entities), batch_size)]
    log(f"Processing {len(batches)} batches in parallel (batch_size={batch_size})")

    tasks = [
        _process_batch(batch, i + 1, structured_llm, semaphore)
        for i, batch in enumerate(batches)
    ]
    batch_results = await asyncio.gather(*tasks)

    # Merge all groups from all batches
    all_groups = []
    for groups in batch_results:
        all_groups.extend(groups)

    # Convert to mapping for pass 1
    pass1_mapping = groups_to_mapping(all_groups, entity_names)

    # PASS 2: Merge canonical names across batches
    canonical_names = list(set(pass1_mapping.values()))
    log(f"Pass 1 produced {len(canonical_names)} canonical names")

    # Build lookup for summaries by name
    summary_by_name = {e["name"]: e.get("summary", "") for e in entities}

    if len(canonical_names) <= 1:
        final_mapping = pass1_mapping
    elif len(canonical_names) <= batch_size:
        # Small enough - run one final merge pass with definitions
        canonical_entities = [{"name": c, "summary": summary_by_name.get(c, "")} for c in canonical_names]
        formatted = _format_entities_for_prompt(canonical_entities)
        user_prompt = f"Entities to deduplicate (merge any that refer to the same entity):\n{formatted}"
        messages = [("system", DEDUPE_SYSTEM_PROMPT), ("human", user_prompt)]

        try:
            merge_result = await asyncio.to_thread(structured_llm.invoke, messages)
            canonical_merge = groups_to_mapping(merge_result.groups, canonical_names)
            log(f"Pass 2: merged to {len(set(canonical_merge.values()))} final canonicals")
        except Exception as e:
            print(f"[EntityDedup] Merge pass error: {e}")
            canonical_merge = {c: c for c in canonical_names}

        # Update original mappings
        final_mapping = {}
        for original, canonical in pass1_mapping.items():
            final_mapping[original] = canonical_merge.get(canonical, canonical)
    else:
        # Too many canonicals - run merge pass with definitions
        log(f"Too many canonicals ({len(canonical_names)}) - running larger merge pass")
        canonical_entities = [{"name": c, "summary": summary_by_name.get(c, "")} for c in canonical_names]
        formatted = _format_entities_for_prompt(canonical_entities)
        user_prompt = f"Entities to deduplicate (merge any that refer to the same entity):\n{formatted}"
        messages = [("system", DEDUPE_SYSTEM_PROMPT), ("human", user_prompt)]

        try:
            merge_result = await asyncio.to_thread(structured_llm.invoke, messages)
            canonical_merge = groups_to_mapping(merge_result.groups, canonical_names)
            log(f"Merge pass: {len(canonical_names)} -> {len(set(canonical_merge.values()))} canonicals")
        except Exception as e:
            print(f"[EntityDedup] Large merge pass error: {e}")
            canonical_merge = {c: c for c in canonical_names}

        final_mapping = {}
        for original, canonical in pass1_mapping.items():
            final_mapping[original] = canonical_merge.get(canonical, canonical)

    # REFLECTION PASS: Review and fix errors
    if use_reflection:
        log("Running reflection pass...")
        final_mapping = await reflect_on_dedup(final_mapping, llm, entities)

    return final_mapping


def collect_entities_from_extractions(
    extractions: List[Dict[str, Any]]
) -> tuple[List[Dict[str, str]], Dict[str, List[Dict]]]:
    """
    Collect all entities with summaries from extractions.

    Args:
        extractions: List of extraction results from Phase 1

    Returns:
        - List of {"name": str, "summary": str} for unique entities
        - Dict mapping name -> list of {chunk_idx, entity} occurrences
    """
    by_name: Dict[str, List] = {}
    for ext in extractions:
        if not ext["success"]:
            continue
        for entity in ext["extraction"].entities:
            if entity.entity_type.lower() != "topic":
                by_name.setdefault(entity.name, []).append({
                    "chunk_idx": ext["chunk_idx"],
                    "entity": entity
                })

    # Build entity list with summaries (use first occurrence's summary)
    entities = []
    for name, occurrences in by_name.items():
        first_entity = occurrences[0]["entity"]
        entities.append({
            "name": name,
            "summary": first_entity.summary if hasattr(first_entity, 'summary') else ""
        })

    return entities, by_name
