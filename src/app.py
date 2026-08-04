import requests
import streamlit as st
from pathlib import Path

from loader import load_pdf
from splitter import split_documents
from vectordb import create_vector_db
from retriever import create_retriever
from embedder import embeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


# ============================================================
# CONFIG
# Visual theme (colors, dark base) lives in .streamlit/config.toml —
# that's Streamlit's own theming engine, so native widgets (selects,
# sliders, alerts, inputs) get correctly contrasted colors from
# Streamlit itself instead of CSS guessing at its internal DOM.
# ============================================================

st.set_page_config(
    page_title="TL;DR — chat with your PDFs",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = str(Path("data") / "chroma")
UPLOAD_DIR = Path("data")
OLLAMA_HOST = "http://localhost:11434"


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "messages": [],
    "retriever": None,
    "file_name": None,
    "num_chunks": None,
    "num_pages": None,
    "model_name": "qwen3:4b",
    "temperature": 0.2,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HELPERS
# ============================================================

@st.cache_data(ttl=30)
def get_ollama_models():
    """Ask the local Ollama daemon which models are actually installed."""
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
        resp.raise_for_status()
        data = resp.json()
        return sorted(m["name"] for m in data.get("models", []))
    except Exception:
        return []


def lerp(a, b, t):
    return a + (b - a) * t


def temperature_to_color(t: float) -> str:
    """Yellow (precise) -> orange -> red (wild)."""
    stops = [
        (0.0, (255, 213, 79)),
        (0.5, (255, 138, 61)),
        (1.0, (229, 62, 62)),
    ]
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t0 <= t <= t1:
            local_t = 0 if t1 == t0 else (t - t0) / (t1 - t0)
            r = round(lerp(c0[0], c1[0], local_t))
            g = round(lerp(c0[1], c1[1], local_t))
            b = round(lerp(c0[2], c1[2], local_t))
            return f"rgb({r},{g},{b})"
    return "rgb(255,138,61)"


# ============================================================
# STYLE — lightweight, only for the brand elements Streamlit's own
# theme doesn't cover (fonts, hero, chips, dividers, chat bubbles).
# Native widgets are left to config.toml so they stay reliable.
# ============================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg:            #0a0908;
        --surface:       #151210;
        --surface-2:     #1c1712;
        --border:        #2b241c;
        --text:          #f5f1ea;
        --text-muted:    #9c9186;
        --accent:        #ff8a3d;
        --accent-soft:   rgba(255, 138, 61, 0.14);
        --accent-strong: #ffb84d;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    [data-testid="stToolbar"] { visibility: hidden; }

    /* ---------- Sidebar: always on, no toggle, period ---------- */
    [data-testid="stSidebarCollapsedControl"] { display: none !important; }
    [data-testid="stSidebarCollapseButton"] { display: none !important; }
    button[title="Collapse sidebar"] { display: none !important; }
    section[data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
        transform: none !important;
        min-width: 300px !important;
        width: 300px !important;
        margin-left: 0 !important;
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"][aria-expanded="false"] {
        display: block !important;
        transform: none !important;
        min-width: 300px !important;
        width: 300px !important;
        margin-left: 0 !important;
    }

    /* ---------- Center + cap the main column width ---------- */
    .block-container {
        max-width: 800px;
        padding-top: 2rem;
        margin: 0 auto;
    }

    /* the searchable field inside selects was grabbing focus and
       popping the on-screen keyboard / autofill strip over the
       dropdown; the wrapper still opens it on click without this */
    [data-baseweb="select"] input {
        pointer-events: none !important;
        caret-color: transparent !important;
    }

    /* ---------- Eyebrow / labels ---------- */
    .eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 0.4rem;
    }

    .hero-wrap {
        text-align: center;
        padding: 2.2rem 0 1.2rem 0;
    }
    .hero-title {
        font-family: 'Instrument Serif', serif;
        font-style: italic;
        font-weight: 400;
        font-size: 5.2rem;
        line-height: 1;
        color: var(--text);
        display: inline-block;
        margin: 0;
    }
    .hero-sub {
        font-family: 'Inter', sans-serif;
        font-size: 1.02rem;
        color: var(--text-muted);
        margin-top: 0.9rem;
    }

    /* ---------- Stepper ---------- */
    .stepper {
        display: flex;
        justify-content: center;
        gap: 2.2rem;
        margin: 1.6rem 0 2.2rem 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        letter-spacing: 0.05em;
        color: var(--text-muted);
    }
    .stepper .step-active { color: var(--accent); }
    .stepper .step-done { color: var(--text); }

    /* ---------- Divider ---------- */
    .stroke-divider {
        height: 3px;
        border: none;
        margin: 2rem 0;
        background: linear-gradient(90deg, transparent 0%, var(--accent) 20%, var(--accent-strong) 50%, var(--accent) 80%, transparent 100%);
        opacity: 0.55;
        border-radius: 2px;
    }

    /* ---------- File uploader ---------- */
    [data-testid="stFileUploaderDropzone"] {
        border-radius: 16px !important;
        transition: border-color 0.2s ease;
    }

    /* ---------- Info chips ---------- */
    .chip {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: var(--text);
        background: var(--surface-2);
        border: 1px solid var(--border);
        border-radius: 999px;
        padding: 0.35rem 0.85rem;
        display: inline-block;
    }
    .chip b { color: var(--accent); }

    /* ---------- Buttons ---------- */
    .stButton button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: transform 0.15s ease !important;
    }
    .stButton button:hover { transform: translateY(-1px); }

    /* ---------- Chat messages ---------- */
    [data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 4px 6px;
        margin-bottom: 0.65rem;
        border: 1px solid var(--border);
    }
    [data-testid="stChatMessage"]:nth-child(odd) {
        border-left: 3px solid var(--border);
    }
    [data-testid="stChatMessage"]:nth-child(even) {
        border-left: 3px solid var(--accent);
    }

    /* ---------- Empty state ---------- */
    .empty-state {
        text-align: center;
        padding: 3rem 1rem;
        color: var(--text-muted);
        border: 1px dashed var(--border);
        border-radius: 16px;
        background: var(--surface);
    }
    .empty-state .glyph {
        font-size: 1.8rem;
        color: var(--accent);
        margin-bottom: 0.6rem;
        display: block;
    }

    h2, h3 { font-family: 'Instrument Serif', serif; font-style: italic; font-weight: 400; }

    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR — settings (always visible, no collapse)
# ============================================================

with st.sidebar:
    st.markdown('<div class="eyebrow">🧠 Model</div>', unsafe_allow_html=True)

    model_col, refresh_col = st.columns([5, 1])
    with refresh_col:
        if st.button("↻", help="Refresh model list", use_container_width=True):
            get_ollama_models.clear()

    available_models = get_ollama_models()

    with model_col:
        if available_models:
            if st.session_state.model_name not in available_models:
                st.session_state.model_name = available_models[0]
            st.session_state.model_name = st.selectbox(
                "Ollama model",
                options=available_models,
                index=available_models.index(st.session_state.model_name),
                label_visibility="collapsed",
                format_func=lambda m: f"🦙 {m}",
            )
        else:
            st.session_state.model_name = st.text_input(
                "Ollama model", value=st.session_state.model_name, label_visibility="collapsed"
            )

    if not available_models:
        st.caption("⚠️ Couldn't reach Ollama on localhost:11434 — showing manual entry.")

    st.markdown('<div class="eyebrow" style="margin-top:1.2rem;">🌡️ Temperature</div>', unsafe_allow_html=True)
    st.session_state.temperature = st.slider(
        "Temperature", min_value=0.0, max_value=1.0,
        value=st.session_state.temperature, step=0.05,
        label_visibility="collapsed",
    )
    temp_color = temperature_to_color(st.session_state.temperature)
    st.markdown(
        f"""
        <style>
        [data-testid="stSlider"] div[role="slider"] {{
            background-color: {temp_color} !important;
            border-color: {temp_color} !important;
            box-shadow: 0 0 10px {temp_color} !important;
        }}
        [data-testid="stThumbValue"] {{
            color: {temp_color} !important;
            font-family: 'JetBrains Mono', monospace !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    mood = "precise 🎯" if st.session_state.temperature < 0.34 else (
        "balanced 🔥" if st.session_state.temperature < 0.67 else "wild 🌶️"
    )
    st.caption(f"{mood} · {st.session_state.temperature:.2f}")

    st.markdown('<div class="eyebrow" style="margin-top:1.4rem;">📄 Document</div>', unsafe_allow_html=True)
    if st.session_state.file_name:
        st.markdown(f'<div class="chip">📄 {st.session_state.file_name}</div>', unsafe_allow_html=True)
        if st.session_state.num_chunks:
            st.markdown(
                f'<div class="chip" style="margin-top:0.4rem;">🧩 chunks: <b>{st.session_state.num_chunks}</b></div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("No document loaded yet.")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🧹 Clear chat", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("🗑️ Remove document", type="secondary", use_container_width=True):
        st.session_state.retriever = None
        st.session_state.file_name = None
        st.session_state.num_chunks = None
        st.session_state.messages = []
        st.rerun()


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero-wrap">
        <div class="hero-title">TL;DR</div>
        <div class="hero-sub">📄 Chat with your PDFs, entirely on your machine — Ollama + RAG</div>
    </div>
    """,
    unsafe_allow_html=True,
)

step_upload = "step-done" if st.session_state.file_name else "step-active"
step_build = "step-active" if st.session_state.file_name and not st.session_state.retriever else (
    "step-done" if st.session_state.retriever else ""
)
step_chat = "step-active" if st.session_state.retriever else ""

st.markdown(
    f"""
    <div class="stepper">
        <span class="{step_upload}">01 · upload</span>
        <span class="{step_build}">02 · build index</span>
        <span class="{step_chat}">03 · ask questions</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LLM CHAIN
# ============================================================

def get_llm():
    return ChatOllama(
        model=st.session_state.model_name,
        temperature=st.session_state.temperature,
    )


prompt = ChatPromptTemplate.from_template(
    """
You answer questions about a document.

Use ONLY the context below. If the answer isn't in the context, say you
couldn't find it in the document — don't guess.

Context:
{context}

Question:
{question}

Answer:
"""
)


# ============================================================
# UPLOAD + BUILD
# ============================================================

uploaded_file = st.file_uploader(
    "Drop your PDF here",
    type="pdf",
    label_visibility="collapsed",
)

if uploaded_file:
    pdf_path = UPLOAD_DIR / uploaded_file.name
    UPLOAD_DIR.mkdir(exist_ok=True)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    already_built = st.session_state.file_name == uploaded_file.name and st.session_state.retriever

    c1, c2 = st.columns([1, 1])
    with c1:
        build_clicked = st.button(
            "🔄 Rebuild knowledge base" if already_built else "⚡ Build knowledge base",
            use_container_width=True,
        )
    with c2:
        st.markdown(
            f'<div class="chip" style="text-align:center;width:100%;">📦 {uploaded_file.size / 1024:.0f} KB · {uploaded_file.name}</div>',
            unsafe_allow_html=True,
        )

    if build_clicked:
        try:
            with st.spinner("📖 Reading pages…"):
                documents = load_pdf(pdf_path)
                assert documents is not None, "load_pdf returned nothing — check loader.py"

            with st.spinner("✂️ Splitting into chunks…"):
                chunks = split_documents(documents)
                assert chunks is not None, "split_documents returned nothing — check splitter.py"

            with st.spinner("🧠 Embedding and indexing…"):
                vector_db = create_vector_db(chunks, embeddings, DB_PATH)
                retriever = create_retriever(vector_db)

            st.session_state.retriever = retriever
            st.session_state.file_name = uploaded_file.name
            st.session_state.num_chunks = len(chunks)
            st.session_state.num_pages = len(documents)
            st.session_state.messages = []

            st.success(f"✅ Indexed {len(chunks)} chunks from {len(documents)} pages. Ready to chat.")
        except Exception as e:
            st.error(f"❌ Couldn't build the index: {e}")


# ============================================================
# CHAT
# ============================================================

st.markdown('<hr class="stroke-divider" />', unsafe_allow_html=True)

if st.session_state.retriever is None:
    st.markdown(
        """
        <div class="empty-state">
            <span class="glyph">◆</span>
            Upload a PDF and build the knowledge base above — then your questions
            show up here.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    for msg in st.session_state.messages:
        avatar = "🧑" if msg["role"] == "user" else "🖍️"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    question = st.chat_input("Ask your document…")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(question)

        retriever = st.session_state.retriever
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | get_llm()
        )

        with st.chat_message("assistant", avatar="🖍️"):
            with st.spinner("Thinking…"):
                try:
                    response = chain.invoke(question)
                    answer = response.content
                except Exception as e:
                    answer = f"⚠️ Something went wrong talking to the model: {e}"
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})