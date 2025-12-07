# Atomizer Workflow Refactor (V2)

## Overview
This document outlines the architectural changes to the ingestion pipeline, specifically the splitting of the `Atomizer` agent into two distinct responsibilities: **Decomposition** and **Entity Extraction**.

## Motivation
Previously, the `Atomizer` was responsible for both breaking text into atomic facts AND extracting structured entities (Subjects, Objects). This conflation led to:
1.  **Hallucinations**: The agent sometimes invented entities not present in the text.
2.  **Loss of Context**: By the time the agent was extracting entities, it was often looking at an isolated sentence without the full context of the original chunk, leading to pronoun resolution issues (e.g., resolving "It" incorrectly).
3.  **Complexity**: The prompt was trying to do too much.

## New Architecture

The workflow has been split into two sequential agents:

### 1. `Atomizer` (Decomposition Agent)
**Responsibility**: STRICTLY de-contextualize text into simple sentences (Propositions).
- **Input**: Original Chunk.
- **Micro-Steps**:
    - **Coreference Resolution**: Replace pronouns ("he", "it") with specific names using chunk context.
    - **Temporal Grounding**: Resolve relative dates ("yesterday") to absolute dates.
    - **Atomicity**: Split compound sentences.
- **Output**: `List[str]` (Simple English sentences).
- **Reflexion**: Includes a self-correction loop to ensure no key information was dropped.

### 2. `EntityExtractor` (Context-Aware Extraction Agent)
**Responsibility**: Extract structured `AtomicFact` objects (Subject, Object, Topics) from the propositions.
- **Input**: 
    - A single `Proposition` (from Atomizer).
    - The **ORIGINAL CHUNK** (for context).
- **Key Feature**: The "In-Context" Prompt.
    - The prompt explicitly provides the full chunk: *"FOR HELP here is the chunk in which it is a part of..."*.
    - This allows the agent to disambiguate "It" or "The company" even if the Atomizer missed it, and provides rich context for Topic classification.
- **Rules**:
    - **Entities vs. Topics**: Strict differentiation (People/Orgs vs. Concepts).
    - **Canonicalization**: Maps "Price Increases" -> "Inflation".

## Workflow Diagram

```mermaid
graph TD
    A[Input Chunk] -->|Text| B[Atomizer]
    B -->|Propositions (List[str])| C[Entity Extractor]
    A -.->|Context| C
    C -->|Atomic Facts| D[Resolution & Assembly]
```

## Benefits
- **Higher Precision**: "Context-Aware" extraction significantly reduces entity resolution errors.
- **Modularity**: We can improve decomposition or extraction independently.
- **Traceability**: Easier to debug why a specific entity was extracted (can trace back to the exact proposition).
