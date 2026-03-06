"""
RAG Strategy: VECTOR SEARCH ONLY (using FAISS + HuggingFace embeddings)

This file uses ONLY semantic/vector similarity search to retrieve relevant chunks.
Vector search finds documents that are semantically similar to the query,
even if they don't share exact keywords.

STRENGTHS:
  - Understands meaning and context (e.g., "launched" ≈ "introduced" ≈ "released")
  - Handles paraphrased queries well
  - Good for broad, conceptual questions

WEAKNESSES:
  - May miss documents with exact keyword matches if semantically distant
  - Can retrieve semantically similar but factually irrelevant chunks
  - Struggles with specific names, dates, acronyms, and technical terms

Compare with: 5b-bm25-keyword-only-rag.py (keyword search)
              5-rag-from-vectordb copy.py  (hybrid search - best of both)
"""

from dotenv import load_dotenv
from os import getenv, path
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import time

load_dotenv()
llm = ChatGroq(model=getenv("MODEL_NAME"), api_key=getenv("GROQ_API_KEY"))
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 1: Load PDF and split into chunks
docs = PyPDFLoader("mcp.pdf").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

# Step 2: Create FAISS vector index (embeddings-based)
if path.exists("faiss_mcp_index"):
    vectordb = FAISS.load_local("faiss_mcp_index", embeddings, allow_dangerous_deserialization=True)
else:
    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local("faiss_mcp_index")

print("=" * 60)
print("  RAG Strategy: VECTOR SEARCH ONLY (FAISS)")
print("=" * 60)
print("This uses ONLY semantic similarity to find relevant chunks.")
print("Good for meaning-based queries, weak for exact keyword matches.")
print("Type 'quit' to exit.\n")

while True:
    query = input("You: ")
    if query.lower() == "quit":
        break

    # Step 3: Vector-only retrieval — find semantically similar chunks
    start_time = time.time()
    vector_results = vectordb.similarity_search(query, k=4)
    retrieval_time = time.time() - start_time

    context = "\n".join([doc.page_content for doc in vector_results])

    # Step 4: Ask LLM with retrieved context
    response = llm.invoke(
        f"Answer the question based ONLY on this context. "
        f"Provide specific details like dates, names, and numbers.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}"
    )

    print(f"\n{'─' * 50}")
    print(f"📊 Retrieval: Vector search | Chunks found: {len(vector_results)} | Time: {retrieval_time:.4f}s")
    print(f"{'─' * 50}")
    print(f"Bot: {response.content}\n")
