from dotenv import load_dotenv
from os import getenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage 

load_dotenv()
llm = ChatGroq(model=getenv("MODEL_NAME"), api_key=getenv("GROQ_API_KEY"))
history = [SystemMessage(content="You are a helpful assistant. Always remember and use information the user has shared earlier in this conversation, including their name.")]

print("Chatbot ready! Type 'quit' to exit.\n")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    history.append(HumanMessage(content=user_input))
    response = llm.invoke(history)
    history.append(response)
    print(f"Bot: {response.content}\n")
