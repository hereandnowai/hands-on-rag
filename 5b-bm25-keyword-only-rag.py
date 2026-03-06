"""
RAG Strategy: KEYWORD SEARCH ONLY (using BM25)

This file uses ONLY BM25 keyword/lexical search to retrieve relevant chunks.
BM25 ranks documents based on term frequency and inverse document frequency —
essentially, it finds chunks that contain the exact words from the query.

STRENGTHS:
  - Excellent at finding exact keyword matches (names, dates, acronyms)
  - Fast and lightweight — no embeddings needed
  - Great for specific, factual queries (e.g., "who launched MCP?")

WEAKNESSES:
  - Cannot understand synonyms or paraphrases ("launched" won't match "introduced")
  - Misses semantically relevant chunks that use different wording
  - Poor at understanding context or meaning behind the query

Compare with: 5a-vector-only-rag.py         (vector search)
              5-rag-from-vectordb copy.py    (hybrid search - best of both)
"""

from dotenv import load_dotenv
from os import getenv
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
import time

load_dotenv()
llm = ChatGroq(model=getenv("MODEL_NAME"), api_key=getenv("GROQ_API_KEY"))

# Step 1: Load PDF and split into chunks
docs = PyPDFLoader("mcp.pdf").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

# Step 2: Create BM25 keyword index (no embeddings needed!)
keyword_retriever = BM25Retriever.from_documents(chunks, k=4)

print("=" * 60)
print("  RAG Strategy: KEYWORD SEARCH ONLY (BM25)")
print("=" * 60)
print("This uses ONLY keyword/lexical matching to find relevant chunks.")
print("Good for exact term matches, weak for meaning-based queries.")
print("Type 'quit' to exit.\n")

while True:
    query = input("You: ")
    if query.lower() == "quit":
        break

    # Step 3: BM25-only retrieval — find chunks with matching keywords
    start_time = time.time()
    keyword_results = keyword_retriever.invoke(query)
    retrieval_time = time.time() - start_time

    context = "\n".join([doc.page_content for doc in keyword_results])

    # Step 4: Ask LLM with retrieved context
    response = llm.invoke(
        f"Answer the question based ONLY on this context. "
        f"Provide specific details like dates, names, and numbers.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}"
    )

    print(f"\n{'─' * 50}")
    print(f"📊 Retrieval: BM25 keyword search | Chunks found: {len(keyword_results)} | Time: {retrieval_time:.4f}s")
    print(f"{'─' * 50}")
    print(f"Bot: {response.content}\n")
