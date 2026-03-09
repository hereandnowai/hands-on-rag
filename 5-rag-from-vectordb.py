from dotenv import load_dotenv
from os import getenv, path
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

load_dotenv()
llm = ChatGroq(model=getenv("MODEL_NAME"), api_key=getenv("GROQ_API_KEY"))
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 1: Load PDF and split into chunks
docs = PyPDFLoader("mcp.pdf").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

# Step 2: Create two search indexes
if path.exists("faiss_mcp_index"):
    vectordb = FAISS.load_local("faiss_mcp_index", embeddings, allow_dangerous_deserialization=True)
else:
    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local("faiss_mcp_index")

keyword_retriever = BM25Retriever.from_documents(chunks, k=4)  # keyword search

print("PDF loaded with hybrid search (keyword + vector). Type 'quit' to exit.\n")
while True:
    query = input("You: ")
    if query.lower() == "quit":
        break
    # Step 3: Hybrid search = keyword results + vector results (deduplicated)
    keyword_results = keyword_retriever.invoke(query)
    vector_results = vectordb.similarity_search(query, k=4)
    seen = set()
    all_docs = []
    for doc in keyword_results + vector_results:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            all_docs.append(doc)
    context = "\n".join([doc.page_content for doc in all_docs])
    # Step 4: Ask LLM with retrieved context
    response = llm.invoke(f"Answer the question based ONLY on this context. Provide specific details like dates, names, and numbers.\n\nContext:\n{context}\n\nQuestion: {query}")
    print(f"Bot: {response.content}\n")
