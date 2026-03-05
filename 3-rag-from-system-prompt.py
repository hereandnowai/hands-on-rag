from dotenv import load_dotenv
from os import getenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
llm = ChatGroq(model=getenv("MODEL_NAME"), api_key=getenv("GROQ_API_KEY"))

content = """
Ruthran Raghavan is a Chief AI Scientist with 10+ years of experience designing, developing, and deploying enterprise-grade AI systems. Expert in LLM-powered applications, advanced Retrieval-Augmented Generation (RAG) pipelines, and complex multi-agent systems using LangChain, LangGraph, and OpenAI’s ecosystem. Proven ability in high-impact AI automation, combining LangGraph agents with Playwright and Playwright MCP for robust web automation. Recognized for building industry-scale AI agents for automation, predictive analytics, and intelligent decision-making that deliver measurable ROI.

Specializes in integrating open-source models (DeepSeek, LLaMA, Mistral, Kimi, GLM, Qwen) and enterprise AI tools (Gemini, GitHub Copilot) into scalable real-world applications. Adept at developing parallel, looping, and collaborative AI agents to solve high-complexity workflows. A leading corporate trainer, having delivered 10,000+ hours of AI and Generative AI training to over 5,000 learners worldwide, enabling organizations to achieve up to 10x productivity improvements through AI adoption.
"""

history = [SystemMessage(content=content)]

print("Chatbot ready! Type 'quit' to exit.\n")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    history.append(HumanMessage(content=user_input))
    response = llm.invoke(history)
    history.append(response)
    print(f"Bot: {response.content}\n")
