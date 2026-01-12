"""
MODULE: LLM Client
DESCRIPTION: Centralized client for LLM interactions to ensure singleton initialization.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()

class LLMClient:
    _instance = None

    @staticmethod
    def get_instance(model="gemini", temperature=0) -> BaseChatModel:
        return ChatOpenAI(model="gpt-5.2", temperature=temperature)
        if model == "gemini":
            return ChatAnthropic(model="claude-sonnet-4-5", temperature=temperature)
            # return ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=temperature)
        elif model == "claude" or model == "sonnet":
            return ChatAnthropic(model="claude-sonnet-4-5", temperature=temperature)
        elif model == "opus":
            return ChatAnthropic(model="claude-opus-4-5", temperature=temperature)
        else:
            raise ValueError(f"Unknown model: {model}")

    @staticmethod
    def get_embeddings(model="voyage-finance-2"):
        """
        Returns a NEW embedding model instance (non-singleton for testing).
        Defaults to Voyage AI (voyage-finance-2) as requested.
        """
        from langchain_voyageai import VoyageAIEmbeddings
        return VoyageAIEmbeddings(model=model)

def get_llm(model="gemini", temperature=0) -> BaseChatModel:
    """
    Helper function to get the LLM instance.
    Used for: main extraction (GPT-5.1 - good balance of quality/cost)

    Args:
        model: "gemini" (default), "claude"/"sonnet", or "opus"
        temperature: LLM temperature (default 0)
    """
    # Use GPT-5.1 for extraction (cheaper than 5.2, similar quality)
    return ChatOpenAI(model="gpt-5.1", temperature=temperature)

def get_claude_llm(temperature=0) -> BaseChatModel:
    """
    Helper function to get Claude Sonnet LLM.
    Used for: entity deduplication (better at nuanced disambiguation).
    """
    return ChatAnthropic(model="claude-sonnet-4-5", temperature=temperature)

def get_nano_llm() -> BaseChatModel:
    """
    Helper function to get a cheap, fast LLM for simple tasks.
    Used for: topic definitions, topic verification, summary merging.
    """
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

def get_nano_gpt_llm() -> BaseChatModel:
    """
    Helper function to get a cheap, fast GPT model for simple tasks.
    Used for: entity graph resolution (Phase 2d).
    """
    return ChatOpenAI(model="gpt-5-mini", temperature=0)

def get_critique_llm() -> BaseChatModel:
    """
    Helper function to get the critique LLM (higher quality for review).
    Used for: extraction critique/reflexion step, query decomposition, synthesis.
    """
    return ChatOpenAI(model="gpt-5.1", temperature=0)

def get_dedup_llm() -> BaseChatModel:
    """
    Helper function to get the entity deduplication LLM.
    Used for: in-document entity clustering (Phase 2b-c).
    GPT-5.1 is good enough for dedup - no need for 5.2.
    """
    return ChatOpenAI(model="gpt-5.1", temperature=0)

def get_embeddings():
    """
    Helper function to get the embedding model (voyage-3-large).
    Used for: fact embeddings, entity embeddings in Neo4j, Qdrant searches.
    """
    return LLMClient.get_embeddings(model="voyage-3-large")

def get_dedup_embeddings():
    """
    Helper function to get embeddings for entity deduplication (voyage-3-large).
    Better at alias/identity matching than domain-specific models.
    """
    return LLMClient.get_embeddings(model="voyage-3-large")
