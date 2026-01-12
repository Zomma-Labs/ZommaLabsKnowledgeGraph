"""
MODULE: Topic Librarian
DESCRIPTION:
    Resolves extracted topic names to the curated topic ontology using:
    1. Vector Search (Semantic Match) via Qdrant + Voyage AI embeddings
    2. LLM Verification - Uses LLM to decide if candidates actually match the input

    Embeddings capture semantic meaning to find candidate topics.
    LLM makes the final decision on whether the extracted topic matches a candidate.
"""

import os
from typing import List, Dict, Optional, TYPE_CHECKING
from pydantic import BaseModel, Field
from src.util.llm_client import get_nano_llm

if TYPE_CHECKING:
    from src.util.services import Services


class TopicResolutionResponse(BaseModel):
    """Structured response for topic resolution."""
    selected_number: Optional[int] = Field(
        None,
        description="The number of the matching topic (1-indexed), or null if no match"
    )


class TopicDefinition(BaseModel):
    """A topic with its contextual definition."""
    topic: str = Field(description="The topic term exactly as provided")
    definition: str = Field(description="A one-sentence definition of what this topic means")


class BatchTopicDefinitions(BaseModel):
    """Batch of topic definitions."""
    definitions: List[TopicDefinition] = Field(description="List of topics with their definitions")

# Control verbose output
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"

QDRANT_PATH = "./qdrant_topics"
COLLECTION_NAME = "topic_ontology"


def log(msg: str):
    """Print only if VERBOSE mode is enabled."""
    if VERBOSE:
        print(msg)


class TopicLibrarian:
    def __init__(self, services: Optional["Services"] = None):
        log("Initializing Topic Librarian...")

        if services is None:
            from src.util.services import get_services
            services = get_services()

        self.embeddings = services.embeddings

        # Initialize Qdrant client for topics
        from qdrant_client import QdrantClient
        self.client = QdrantClient(path=QDRANT_PATH)

        # LLM for candidate verification (with structured output)
        self.llm = get_nano_llm().with_structured_output(TopicResolutionResponse)
        self.definition_llm = get_nano_llm().with_structured_output(BatchTopicDefinitions)

    def batch_define_topics(self, topics: List[str], context: str) -> Dict[str, str]:
        """
        Generate definitions for a batch of topics from a single chunk.

        Args:
            topics: List of topic terms extracted from the chunk
            context: The full chunk text

        Returns:
            Dict mapping topic -> "topic: definition" for enriched embedding
        """
        if not topics:
            return {}

        # Deduplicate while preserving order
        unique_topics = list(dict.fromkeys(topics))

        topics_list = "\n".join([f"- {t}" for t in unique_topics])

        prompt = f"""Define each financial/business topic in one sentence.

CONTEXT:
"{context}"

TOPICS TO DEFINE:
{topics_list}

For each topic, provide a concise one-sentence definition explaining what it means in financial/business terms. Return the topic exactly as written."""

        try:
            response: BatchTopicDefinitions = self.definition_llm.invoke(prompt)

            # Build result mapping
            result = {}
            defined = {d.topic.lower(): d.definition for d in response.definitions}

            for topic in unique_topics:
                definition = defined.get(topic.lower(), "")
                if definition:
                    result[topic] = f"{topic}: {definition}"
                else:
                    result[topic] = topic  # Fallback to raw

            log(f"   Defined {len(result)} topics for enriched matching")
            return result

        except Exception as e:
            log(f"   Batch definition failed: {e}")
            # Fallback: return raw topics
            return {t: t for t in unique_topics}

    def resolve_with_definition(self, text: str, enriched_text: str, context: str = "",
                                 top_k: int = 15, candidate_threshold: float = 0.40) -> Optional[Dict]:
        """
        Resolves a topic using enriched text (topic: definition) for better semantic matching.

        Args:
            text: The original topic name
            enriched_text: "topic: definition" string for embedding
            context: The source fact/sentence
            top_k: Number of candidates to retrieve
            candidate_threshold: Minimum similarity score

        Returns:
            Match dictionary or None
        """
        if not text or not text.strip():
            return None

        text = text.strip()

        # Vector search using ENRICHED text
        candidates = self._vector_search(enriched_text, k=top_k)

        sorted_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)

        if not sorted_candidates or sorted_candidates[0]['score'] < candidate_threshold:
            log(f"   No topic candidates for '{text}' (below threshold)")
            return None

        viable_candidates = [c for c in sorted_candidates if c['score'] >= candidate_threshold][:20]

        # LLM verification uses original text + context
        selected = self._llm_verify_topic(text, context, viable_candidates)

        if selected:
            log(f"   Topic '{text}' -> '{selected['label']}' (definition-enriched)")
            return selected
        else:
            log(f"   Topic '{text}' rejected by LLM")
            return None

    def resolve(self, text: str, context: str = "", top_k: int = 15, candidate_threshold: float = 0.40) -> Optional[Dict]:
        """
        Resolves a topic name to the best matching ontology concept.

        Args:
            text: The topic name to resolve
            context: The source fact/sentence where this topic was extracted from
            top_k: Number of candidates to retrieve from vector search
            candidate_threshold: Minimum cosine similarity to consider a candidate

        Returns:
            Match dictionary with 'label', 'uri', 'definition', 'score' or None
        """
        if not text or not text.strip():
            return None

        text = text.strip()

        # Vector Search (Semantic) - embeddings capture meaning
        candidates = self._vector_search(text, k=top_k)

        # Sort by score
        sorted_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)

        # Auto-reject if no candidates above threshold
        if not sorted_candidates or sorted_candidates[0]['score'] < candidate_threshold:
            log(f"   No topic candidates for '{text}' (below threshold)")
            return None

        # Get top candidates for LLM verification
        viable_candidates = [c for c in sorted_candidates if c['score'] >= candidate_threshold][:20]

        # LLM decides which candidate (if any) actually matches
        selected = self._llm_verify_topic(text, context, viable_candidates)

        if selected:
            log(f"   Topic '{text}' -> '{selected['label']}' (LLM verified)")
            return selected
        else:
            log(f"   Topic '{text}' rejected by LLM (no valid match)")
            return None

    def _llm_verify_topic(self, input_text: str, context: str, candidates: List[Dict]) -> Optional[Dict]:
        """
        Uses LLM to decide which candidate (if any) actually matches the input.
        Returns the matching candidate dict or None if no match.
        """
        if not candidates:
            return None

        # Build candidate list with Topic : Definition : Examples format
        candidate_list = "\n".join([
            f"{i+1}. {c['label']}: {c.get('definition', 'No definition')} (e.g., {c.get('synonyms', 'N/A')})"
            for i, c in enumerate(candidates)
        ])

        # Include context if provided
        context_section = f'"{context}"' if context else "No context provided."

        prompt = f"""TASK: Match an extracted topic to its canonical form from our ontology.

EXTRACTED TOPIC: "{input_text}"

SOURCE CONTEXT (use this to understand what the extracted topic means):
{context_section}

CANDIDATE TOPICS FROM ONTOLOGY:
{candidate_list}

INSTRUCTIONS:
1. Use the SOURCE CONTEXT to understand what "{input_text}" refers to in this specific usage
2. Compare the MEANING of the extracted topic against each candidate's definition
3. If the extracted topic clearly matches one candidate's meaning, return that number
4. If no candidate reliably matches what the extracted topic means, return null

Return the matching candidate number (1-{len(candidates)}), or null if no reliable match."""

        try:
            response: TopicResolutionResponse = self.llm.invoke(prompt)

            if response.selected_number is None:
                return None

            idx = response.selected_number - 1  # Convert to 0-indexed
            if 0 <= idx < len(candidates):
                return candidates[idx]

            return None

        except Exception as e:
            log(f"   LLM verification failed: {e}")
            return None

    def resolve_topics(self, topics: List[str], context: str = "") -> List[str]:
        """
        Resolves a list of extracted topics to canonical names.
        Returns only valid topics that match the ontology.
        """
        resolved = []
        seen = set()

        for topic in topics:
            match = self.resolve(topic, context=context)
            if match:
                canonical = match['label']
                if canonical not in seen:
                    resolved.append(canonical)
                    seen.add(canonical)

        return resolved

    def _vector_search(self, text: str, k: int) -> List[Dict]:
        """Queries Qdrant for semantic matches."""
        if not self.client.collection_exists(COLLECTION_NAME):
            return []

        try:
            vector = self.embeddings.embed_query(text)

            results = self.client.query_points(
                collection_name=COLLECTION_NAME,
                query=vector,
                limit=k,
                with_payload=True
            ).points

            candidates = []
            for hit in results:
                candidates.append({
                    "uri": hit.payload["uri"],
                    "label": hit.payload["label"],
                    "definition": hit.payload.get("definition", ""),
                    "synonyms": hit.payload.get("synonyms", ""),
                    "score": hit.score
                })
            return candidates

        except Exception as e:
            log(f"Topic vector search failed: {e}")
            return []


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

    librarian = TopicLibrarian()

    # Test cases: (topic, context, expected_result)
    test_cases = [
        ("Inflation", "Inflation is causing prices to rise.", "should match"),
        ("M&A", "The company engaged in M&A activity.", "should match"),
        ("$1 Trillion Market Value", "Apple reached $1 Trillion Market Value.", "should match Market Valuation"),
        ("$7.5 Million", "Purchased equipment for $7.5 Million.", "should REJECT"),
        ("CEO", "The CEO announced layoffs.", "should REJECT"),
        ("Random Garbage", "Some random text.", "should REJECT"),
    ]

    print("\nTopic Resolution Tests (with context):")
    print("-" * 70)

    for topic, context, expected in test_cases:
        result = librarian.resolve(topic, context=context)
        status = f"-> {result['label']}" if result else "-> REJECTED"
        print(f"  {topic:30} {status:25} (expected: {expected})")
