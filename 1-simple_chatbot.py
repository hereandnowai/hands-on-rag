from dotenv import load_dotenv
from os import getenv
from langchain_groq import ChatGroq

load_dotenv()
llm = ChatGroq(model=getenv("MODEL_NAME"), api_key=getenv("GROQ_API_KEY"))

print("Chatbot ready! Type 'quit' to exit.\n")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    response = llm.invoke(user_input)
    print(f"Bot: {response.content}\n")