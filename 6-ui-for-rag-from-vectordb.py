"""
6 - ChatGPT-style RAG UI  |  HERE AND NOW AI
Uses Gradio Blocks to build a polished chat interface with branded styling,
hybrid search (BM25 + FAISS), and streaming responses from Groq.
"""

import json, gradio as gr
from dotenv import load_dotenv
from os import getenv, path
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever

# ──────────────────────────────────────────────
# 0. Config & branding
# ──────────────────────────────────────────────
load_dotenv()

with open(path.join(path.dirname(path.abspath(__file__)), "branding.json")) as f:
    brand = json.load(f)["brand"]

PRIMARY   = brand["colors"]["primary"]    # #FFDF00  (golden yellow)
SECONDARY = brand["colors"]["secondary"]  # #004040  (dark teal)
LOGO_URL  = brand["logo"]["favicon"]
AVATAR    = brand["chatbot"]["face"]
ORG_NAME  = brand["organizationName"]
SLOGAN    = brand["slogan"]
WEBSITE   = brand["website"]

# ──────────────────────────────────────────────
# 1. LLM + embeddings
# ──────────────────────────────────────────────
llm = ChatGroq(model=getenv("MODEL_NAME"), api_key=getenv("GROQ_API_KEY"))
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ──────────────────────────────────────────────
# 2. Session State Initialization
# ──────────────────────────────────────────────
def get_initial_state():
    """Load default mcp.pdf into session state."""
    pdf_path = path.join(path.dirname(path.abspath(__file__)), "mcp.pdf")
    docs = PyPDFLoader(pdf_path).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    
    faiss_dir = path.join(path.dirname(path.abspath(__file__)), "faiss_mcp_index")
    if path.exists(faiss_dir):
        vectordb = FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)
    else:
        vectordb = FAISS.from_documents(chunks, embeddings)
        vectordb.save_local(faiss_dir)
        
    keyword_retriever = BM25Retriever.from_documents(chunks, k=4)
    return {"chunks": chunks, "vectordb": vectordb, "keyword_retriever": keyword_retriever}

def process_uploaded_pdf(file_path, state):
    """Process a new PDF and merge it into the current session state."""
    if not file_path:
        return state, gr.Info("No file provided.")
        
    gr.Info("Processing uploaded document...")
    docs = PyPDFLoader(file_path).load()
    new_chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    
    # Update state lists
    state["chunks"].extend(new_chunks)
    
    # Update FAISS
    state["vectordb"].add_documents(new_chunks)
    
    # Re-initialize BM25 since it doesn't support adding incrementally easily.
    state["keyword_retriever"] = BM25Retriever.from_documents(state["chunks"], k=4)
    
    gr.Info("Document successfully indexed!")
    return state

# ──────────────────────────────────────────────
# 3. Chat logic
# ──────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a helpful AI assistant powered by HERE AND NOW AI. "
    "Answer questions based ONLY on the provided context. "
    "Provide specific details like dates, names, and numbers when available. "
    "If the context does not contain enough information, say so politely."
)

def hybrid_retrieve(query: str, state: dict, k: int = 4) -> str:
    """Keyword (BM25) + vector (FAISS) search, deduplicated."""
    keyword_results = state["keyword_retriever"].invoke(query)
    vector_results  = state["vectordb"].similarity_search(query, k=k)
    seen, merged = set(), []
    for doc in keyword_results + vector_results:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            merged.append(doc)
    return "\n\n".join(d.page_content for d in merged)


def respond(message: str, history: list[dict], state: dict) -> str:
    """Generate a response using hybrid RAG."""
    context = hybrid_retrieve(message, state)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    # Include recent history for conversational continuity (last 6 turns)
    for turn in history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({
        "role": "user",
        "content": (
            f"Context:\n{context}\n\n"
            f"Question: {message}"
        ),
    })
    response = llm.invoke(messages)
    return response.content


# ──────────────────────────────────────────────
# 4. Custom CSS — Premium glossy / glowing UI
# ──────────────────────────────────────────────
CUSTOM_CSS = """
/* ━━━ KEYFRAME ANIMATIONS ━━━━━━━━━━━━━━━━━━ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes glowPulse {
    0%, 100% { box-shadow: 0 0 20px rgba(255,223,0,.15), 0 0 40px rgba(0,64,64,.1); }
    50%      { box-shadow: 0 0 30px rgba(255,223,0,.25), 0 0 60px rgba(0,64,64,.2); }
}
@keyframes floatIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position: 200% center; }
}
@keyframes borderGlow {
    0%, 100% { border-color: rgba(255,223,0,.3); }
    50%      { border-color: rgba(255,223,0,.6); }
}
@keyframes avatarFloat {
    0%, 100% { transform: translateY(0px); }
    50%      { transform: translateY(-6px); }
}
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(20px) scale(0.98); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}

/* ━━━ GLOBAL RESET ━━━━━━━━━━━━━━━━━━━━━━━━━ */
*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: #0a0a0f !important;
    color: #e8ecf1 !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    margin: 0 !important; padding: 0 !important;
    min-height: 100vh;
}

/* animated bg mesh */
.gradio-container::before {
    content: '';
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background:
        radial-gradient(ellipse 600px 400px at 20% 20%, rgba(0,64,64,.18) 0%, transparent 70%),
        radial-gradient(ellipse 500px 500px at 80% 80%, rgba(255,223,0,.06) 0%, transparent 70%),
        radial-gradient(ellipse 400px 400px at 50% 50%, rgba(88,28,135,.08) 0%, transparent 70%);
    animation: gradientShift 15s ease-in-out infinite;
    background-size: 200% 200%;
}

/* ━━━ CENTERING WRAPPER ━━━━━━━━━━━━━━━━━━━━ */
.gradio-container > .main { max-width: 820px !important; margin: 0 auto !important; padding: 0 16px !important; position: relative; z-index: 1; }
.contain { max-width: 100% !important; }

/* kill default Gradio borders & bg */
.block, #component-0, .wrap, .form { border: none !important; background: transparent !important; box-shadow: none !important; }
.gap { gap: 0 !important; }

/* ━━━ HEADER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
.header-wrap {
    background: linear-gradient(135deg, #003838 0%, #002828 40%, #001a2e 100%);
    border: 1px solid rgba(255,223,0,.2);
    border-radius: 20px;
    margin: 20px 0 16px;
    padding: 18px 28px;
    display: flex; align-items: center; gap: 16px;
    backdrop-filter: blur(20px);
    box-shadow:
        0 4px 30px rgba(0,0,0,.5),
        inset 0 1px 0 rgba(255,255,255,.05),
        0 0 40px rgba(0,64,64,.15);
    animation: fadeSlideUp .6s ease-out, glowPulse 4s ease-in-out infinite;
}
.header-wrap img {
    height: 46px; border-radius: 12px;
    filter: drop-shadow(0 2px 8px rgba(255,223,0,.3));
    transition: transform .3s;
}
.header-wrap img:hover { transform: scale(1.08) rotate(-2deg); }
.header-wrap .h-title {
    font-size: 1.35rem; font-weight: 800;
    background: linear-gradient(135deg, #FFDF00 0%, #FFB800 50%, #FF8C00 100%);
    background-size: 200% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 3s linear infinite;
    letter-spacing: .3px;
}
.header-wrap .h-slogan {
    font-size: .85rem; font-weight: 400;
    color: rgba(255,223,0,.55);
    margin-left: 4px;
}

/* ━━━ CHATBOT AREA ━━━━━━━━━━━━━━━━━━━━━━━━━ */
.chat-panel {
    background: linear-gradient(180deg, rgba(15,20,30,.85) 0%, rgba(10,14,22,.95) 100%);
    border: 1px solid rgba(255,223,0,.1);
    border-radius: 20px;
    padding: 4px;
    backdrop-filter: blur(16px);
    box-shadow: 0 8px 32px rgba(0,0,0,.4), inset 0 1px 0 rgba(255,255,255,.03);
    animation: fadeSlideUp .7s ease-out;
}
.chatbot-area { border: none !important; background: transparent !important; border-radius: 16px !important; }
.chatbot-area .wrapper { background: transparent !important; }

/* messages */
.chatbot-area .message { animation: floatIn .35s ease-out; }

/* user bubble */
.chatbot-area .role-user .message-content, .chatbot-area [class*="user"] .message-bubble-text {
    background: linear-gradient(135deg, #004d4d 0%, #003838 100%) !important;
    color: #e8ecf1 !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 13px 20px !important;
    border: 1px solid rgba(255,223,0,.15) !important;
    box-shadow: 0 2px 12px rgba(0,64,64,.3);
    line-height: 1.6; font-size: .92rem;
}

/* bot bubble */
.chatbot-area .role-assistant .message-content, .chatbot-area [class*="bot"] .message-bubble-text {
    background: linear-gradient(135deg, rgba(30,35,50,.9) 0%, rgba(20,25,40,.95) 100%) !important;
    color: #e8ecf1 !important;
    border-radius: 18px 18px 18px 4px !important;
    padding: 13px 20px !important;
    border: 1px solid rgba(255,255,255,.06) !important;
    box-shadow: 0 2px 12px rgba(0,0,0,.25);
    line-height: 1.6; font-size: .92rem;
}

/* avatar glow */
.chatbot-area .avatar-container img, .chatbot-area .bot-avatar img {
    border-radius: 50% !important;
    box-shadow: 0 0 12px rgba(255,223,0,.25) !important;
    border: 2px solid rgba(255,223,0,.2) !important;
    transition: box-shadow .3s;
}
.chatbot-area .avatar-container img:hover, .chatbot-area .bot-avatar img:hover {
    box-shadow: 0 0 20px rgba(255,223,0,.45) !important;
}

/* placeholder */
.chatbot-area .placeholder { animation: fadeSlideUp 1s ease-out; }

/* ━━━ INPUT AREA ━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
.input-wrap {
    background: linear-gradient(180deg, rgba(15,20,30,.7) 0%, rgba(15,20,30,.9) 100%);
    border: 1px solid rgba(255,223,0,.12);
    border-radius: 16px;
    padding: 12px 14px;
    margin-top: 10px;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 20px rgba(0,0,0,.3);
    transition: border-color .3s;
    animation: fadeSlideUp .8s ease-out;
}
.input-wrap:focus-within {
    border-color: rgba(255,223,0,.35);
    box-shadow: 0 4px 20px rgba(0,0,0,.3), 0 0 25px rgba(255,223,0,.08);
}

textarea, input[type="text"] {
    background: rgba(20,25,40,.6) !important;
    color: #e8ecf1 !important;
    border: 1px solid rgba(255,255,255,.08) !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
    font-size: .95rem !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    transition: all .3s ease;
    line-height: 1.5 !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: rgba(255,223,0,.4) !important;
    box-shadow: 0 0 0 3px rgba(255,223,0,.08), 0 0 20px rgba(255,223,0,.06) !important;
    outline: none !important;
    background: rgba(25,30,50,.7) !important;
}
textarea::placeholder, input::placeholder { color: rgba(255,255,255,.3) !important; }

/* ━━━ SEND BUTTON ━━━━━━━━━━━━━━━━━━━━━━━━━━ */
button.primary, button[variant="primary"] {
    background: linear-gradient(135deg, #FFDF00 0%, #FFB800 50%, #FF9500 100%) !important;
    background-size: 200% auto;
    color: #002020 !important;
    font-weight: 700 !important;
    font-size: .9rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    cursor: pointer;
    transition: all .25s ease;
    box-shadow: 0 4px 15px rgba(255,223,0,.25);
    text-transform: uppercase;
    letter-spacing: .5px;
}
button.primary:hover, button[variant="primary"]:hover {
    background-position: right center !important;
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(255,223,0,.4), 0 0 40px rgba(255,223,0,.15) !important;
}
button.primary:active, button[variant="primary"]:active {
    transform: translateY(0);
}

/* ━━━ NEW CHAT BUTTON ━━━━━━━━━━━━━━━━━━━━━━ */
button.secondary, button[variant="secondary"] {
    background: rgba(20,25,40,.6) !important;
    color: rgba(255,255,255,.6) !important;
    border: 1px solid rgba(255,255,255,.1) !important;
    border-radius: 12px !important;
    padding: 8px 20px !important;
    font-weight: 500 !important;
    transition: all .3s ease;
    backdrop-filter: blur(8px);
}
button.secondary:hover, button[variant="secondary"]:hover {
    border-color: rgba(255,223,0,.3) !important;
    color: #FFDF00 !important;
    background: rgba(255,223,0,.06) !important;
    box-shadow: 0 0 20px rgba(255,223,0,.08);
}

/* ━━━ EXAMPLES ACCORDION ━━━━━━━━━━━━━━━━━━━ */
.examples-accordion {
    margin-top: 10px;
    animation: fadeSlideUp .9s ease-out;
}
.examples-accordion .label-wrap { border: none !important; }
.examples-accordion button, .examples-accordion .label-wrap span {
    color: rgba(255,223,0,.65) !important;
    font-weight: 500; font-size: .9rem;
}
.examples-accordion .gallery, .examples-accordion table {
    background: rgba(15,20,30,.5) !important;
    border: 1px solid rgba(255,255,255,.06) !important;
    border-radius: 12px !important;
}
.examples-accordion td, .examples-accordion .gallery-item {
    background: rgba(20,28,42,.7) !important;
    border: 1px solid rgba(255,255,255,.06) !important;
    border-radius: 10px !important;
    color: #c0c8d4 !important;
    transition: all .25s ease;
    cursor: pointer;
}
.examples-accordion td:hover, .examples-accordion .gallery-item:hover {
    background: rgba(255,223,0,.06) !important;
    border-color: rgba(255,223,0,.2) !important;
    color: #FFDF00 !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,.2);
}

/* ━━━ FOOTER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
.footer-bar {
    text-align: center;
    padding: 16px 10px 24px;
    font-size: .78rem;
    color: rgba(255,255,255,.3);
    animation: fadeSlideUp 1s ease-out;
}
.footer-bar a {
    color: rgba(255,223,0,.5);
    text-decoration: none;
    font-weight: 500;
    transition: color .2s;
}
.footer-bar a:hover {
    color: #FFDF00;
    text-shadow: 0 0 8px rgba(255,223,0,.3);
}
.footer-bar .sep { margin: 0 8px; opacity: .3; }

/* ━━━ SCROLLBAR ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,223,0,.15); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,223,0,.3); }

/* ━━━ MISC OVERRIDES ━━━━━━━━━━━━━━━━━━━━━━━ */
.label-wrap, .svelte-1gfkn6j { border: none !important; }
footer { display: none !important; }
.show-api, .built-with { display: none !important; }
"""

# ──────────────────────────────────────────────
# 5. Build the Gradio app
# ──────────────────────────────────────────────
EXAMPLE_QUERIES = [
    ["What is MCP?"],
    ["Who launched MCP and when?"],
    ["How does MCP work?"],
    ["What are the key benefits of MCP?"],
    ["Who are the contributors to MCP?"],
]

with gr.Blocks(title=f"{ORG_NAME} — RAG Chat") as demo:

    # ── Session State ──
    session_state = gr.State(get_initial_state)

    # ── Branded header ──
    gr.HTML(f"""
    <div class="header-wrap">
        <img src="{LOGO_URL}" alt="logo" />
        <div>
            <span class="h-title">{ORG_NAME}</span>
            <span class="h-slogan">&nbsp;· {SLOGAN}</span>
        </div>
    </div>
    """)

    with gr.Row():
        # ── Sidebar for Uploading ──
        with gr.Column(scale=1):
            gr.Markdown("### Add Knowledge")
            pdf_upload = gr.File(
                label="Upload extra PDF context",
                file_types=[".pdf"],
                elem_classes=["input-wrap"],
                height=150
            )

        # ── Chat area (glassmorphic panel) ──
        with gr.Column(elem_classes=["chat-panel"], scale=3):
            chatbot = gr.Chatbot(
                label="",
                elem_classes=["chatbot-area"],
                avatar_images=(None, AVATAR),
                height=480,
                buttons=["copy"],
                layout="bubble",
                placeholder=(
                    f"<div style='text-align:center;padding:60px 20px'>"
                    f"<img src='{AVATAR}' style='width:88px;height:88px;border-radius:50%;"
                    f"box-shadow:0 0 25px rgba(255,223,0,.25),0 4px 20px rgba(0,0,0,.4);"
                    f"border:2px solid rgba(255,223,0,.2);animation:avatarFloat 3s ease-in-out infinite' />"
                    f"<h2 style='background:linear-gradient(135deg,#FFDF00,#FFB800);-webkit-background-clip:text;"
                    f"-webkit-text-fill-color:transparent;font-weight:700;margin:18px 0 6px;font-size:1.4rem'>"
                    f"How can I help you today?</h2>"
                    f"<p style='color:rgba(255,255,255,.4);font-size:.9rem;font-weight:300'>"
                    f"Ask me anything about the MCP document or upload your own.</p>"
                    f"</div>"
                ),
            )

            # ── Input row ──
            with gr.Row(elem_classes=["input-wrap"]):
                msg = gr.Textbox(
                    placeholder="Type a message…",
                    show_label=False,
                    scale=9,
                    container=False,
                    autofocus=True,
                )
                send_btn = gr.Button("Send ➤", variant="primary", scale=1, min_width=100)

            # ── New chat button ──
            with gr.Row():
                clear_btn = gr.Button("✦  New Chat", variant="secondary", size="sm")

            # ── Example prompts ──
            with gr.Accordion("✨  Try these questions", open=False, elem_classes=["examples-accordion"]):
                gr.Examples(examples=EXAMPLE_QUERIES, inputs=msg, label="")

    # ── Footer ──
    gr.HTML(f"""
    <div class="footer-bar">
        Powered by <a href="{WEBSITE}" target="_blank">{ORG_NAME}</a>
        <span class="sep">·</span>
        <a href="{brand['socialMedia']['linkedin']}" target="_blank">LinkedIn</a>
        <span class="sep">·</span>
        <a href="{brand['socialMedia']['youtube']}" target="_blank">YouTube</a>
        <span class="sep">·</span>
        <a href="{brand['socialMedia']['github']}" target="_blank">GitHub</a>
    </div>
    """)

    # ── Wiring ──
    pdf_upload.upload(
        process_uploaded_pdf,
        inputs=[pdf_upload, session_state],
        outputs=[session_state]
    )

    def user_submit(user_message, history, state):
        """Append user message and generate bot reply."""
        if not user_message.strip():
            return "", history
        history = history + [{"role": "user", "content": user_message}]
        bot_reply = respond(user_message, history, state)
        history = history + [{"role": "assistant", "content": bot_reply}]
        return "", history

    msg.submit(user_submit, [msg, chatbot, session_state], [msg, chatbot])
    send_btn.click(user_submit, [msg, chatbot, session_state], [msg, chatbot])
    clear_btn.click(lambda: ([], get_initial_state(), None), outputs=[chatbot, session_state, pdf_upload])

# ──────────────────────────────────────────────
# 6. Launch
# ──────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(favicon_path=None, css=CUSTOM_CSS, theme=gr.themes.Base())
