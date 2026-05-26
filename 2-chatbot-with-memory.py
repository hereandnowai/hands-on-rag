from dotenv import load_dotenv
from os import getenv
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage 

load_dotenv()
model = getenv("MODEL_NAME")
api_key = getenv("GROQ_API_KEY")

if not model or not api_key:
    raise ValueError("MODEL_NAME or GROQ_API_KEY not found in .env file.")

llm = ChatGroq(model=model, api_key=api_key) # type: ignore
history: list[BaseMessage] = [SystemMessage(content="You are Caramel AI, built by HERE AND NOW AI")]

print("Chatbot ready! Type 'quit' to exit.\n")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    history.append(HumanMessage(content=user_input))
    response = llm.invoke(history)
    history.append(response)
    print(f"Bot: {response.content}\n")
