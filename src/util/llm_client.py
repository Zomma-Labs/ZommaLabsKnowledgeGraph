"""
MODULE: LLM Client
DESCRIPTION: Centralized client for LLM interactions to ensure singleton initialization.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()

class LLMClient:
    _instance = None


    @classmethod
    def get_instance(cls) -> BaseChatModel:
        if cls._instance is None:
            # Default to gpt-4o as it is a strong default, configurable via env
            model = os.getenv("LLM_MODEL", "gpt-4.1-mini")
            
            if model.startswith("gemini"):
                if not os.getenv("GOOGLE_API_KEY"):
                     raise ValueError("GOOGLE_API_KEY not found in environment variables")
                cls._instance = ChatGoogleGenerativeAI(model=model, temperature=0)
            else:
                cls._instance = ChatOpenAI(model=model, temperature=0)
        return cls._instance

    @staticmethod
    def get_embeddings():
        """
        Returns the embedding model instance.
        Defaults to Voyage AI (voyage-finance-2) as requested.
        """
        from langchain_voyageai import VoyageAIEmbeddings
        
        # Ensure VOYAGE_API_KEY is set in environment
        return VoyageAIEmbeddings(model="voyage-finance-2")

def get_llm() -> BaseChatModel:
    """
    Helper function to get the singleton LLM instance.
    """
    return LLMClient.get_instance()

def get_embeddings():
    """
    Helper function to get the embedding model.
    """
    return LLMClient.get_embeddings()
