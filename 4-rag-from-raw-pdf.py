from dotenv import load_dotenv
from os import getenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from pypdf import PdfReader

load_dotenv()
llm = ChatGroq(model=getenv("MODEL_NAME"), api_key=getenv("GROQ_API_KEY"))

# Extract text from PDF (no vectorization needed)
reader = PdfReader("profile-of-ruthran-raghavan-chief-ai-scientist-here-and-now-ai.pdf")
pdf_text = "\n".join([page.extract_text() for page in reader.pages])

history = [SystemMessage(content=f"You are a helpful assistant. Answer questions ONLY from the following document:\n\n{pdf_text}")]

print("PDF RAG Chatbot ready! Type 'quit' to exit.\n")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    history.append(HumanMessage(content=user_input))
    response = llm.invoke(history)
    history.append(response)
    print(f"Bot: {response.content}\n")
