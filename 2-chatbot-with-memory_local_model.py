from dotenv import load_dotenv
from os import getenv
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
llm = ChatOllama(model=getenv("MODEL_NAME_LOCAL"))
history = [SystemMessage(content="You are Caramel AI, built by HERE AND NOW AI")]

print("Chatbot ready! Type 'quit' to exit.\n")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    history.append(HumanMessage(content=user_input))
    response = llm.invoke(history)
    history.append(response)
    print(f"Bot: {response.content}\n")
