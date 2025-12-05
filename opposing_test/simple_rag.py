import json
import os
import sys
import numpy as np
from typing import List, Dict
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

def load_chunks(filepath: str) -> List[Document]:
    """Loads chunks from a JSONL file."""
    documents = []
    print(f"Loading chunks from {filepath}...")
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line)
            # Assuming the JSONL has 'text' or 'content' field. 
            # The JSONL has 'body' field for the text content.
            text = data.get('body') or data.get('chunk_text') or data.get('text')
            metadata = data.get('metadata', {})
            
            if text:
                documents.append(Document(page_content=text, metadata=metadata))
    
    print(f"Loaded {len(documents)} chunks.")
    return documents

def run_simple_rag():
    # Configuration
    chunk_file = "src/chunker/SAVED/beigebook_20251015.jsonl"
    query = "How are the labor markets in New York and Boston?"
    
    # 1. Load Data
    docs = load_chunks(chunk_file)
    
    if not docs:
        print("No documents found. Exiting.")
        return

    # 2. Embed and Index
    print("Embedding chunks with OpenAI...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small") # Using a standard OpenAI model
    
    # Using FAISS for simple in-memory vector search
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # 3. Retrieve
    print(f"Searching for: '{query}'")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    relevant_docs = retriever.invoke(query)
    
    print(f"Found {len(relevant_docs)} relevant chunks.")
    
    # 4. Generate Answer
    print("Generating answer...")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    context_str = "\n\n".join([f"Chunk {i+1}:\n{d.page_content}" for i, d in enumerate(relevant_docs)])
    
    prompt = (
        "You are a helpful assistant. Answer the user's question based ONLY on the provided context. Be indepth and provide a detailed answer. This is for a financial analyst.\n"
        f"Question: {query}\n\n"
        f"Context:\n{context_str}\n\n"
        "Answer:"
    )
    
    response = llm.invoke(prompt)
    
    print("\n" + "="*50)
    print("FINAL ANSWER")
    print("="*50)
    print(response.content)
    print("="*50)

if __name__ == "__main__":
    run_simple_rag()
