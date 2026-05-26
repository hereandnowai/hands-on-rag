from dotenv import load_dotenv
from os import getenv
from langchain_groq import ChatGroq

load_dotenv()
model = getenv("MODEL_NAME")
api_key = getenv("GROQ_API_KEY")

if not model or not api_key:
    raise ValueError("MODEL_NAME or GROQ_API_KEY not found in .env file.")

llm = ChatGroq(model=model, api_key=api_key) # type: ignore

print("Chatbot ready! Type 'quit' to exit.\n")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    response = llm.invoke(user_input)
    print(f"Bot: {response.content}\n")