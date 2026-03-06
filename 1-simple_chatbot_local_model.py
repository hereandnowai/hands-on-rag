from dotenv import load_dotenv
from os import getenv
from langchain_ollama import ChatOllama

load_dotenv()
llm = ChatOllama(model=getenv("MODEL_NAME_LOCAL"))

print("Chatbot ready! Type 'quit' to exit.\n")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    response = llm.invoke(user_input)
    print(f"Bot: {response.content}\n")
