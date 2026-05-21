"""
Streamlit UI for the Deep Research Agent.

This module provides a polished chat-style interface that orchestrates the
existing research pipeline (search → fetch → context_builder → answer → memory)
while streaming intermediate operational updates to the user in real time.

Usage:
    streamlit run streamlit_app.py
"""

import time
import streamlit as st

# ── Import existing pipeline modules ──────────────────────────────────────────
from search import web_search
from fetch import fetch_and_extract
from context_builder import build_context
from answer import generate_answer
import memory

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deep Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS  –  Premium dark-mode-first design with glassmorphism accents
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root variables ──────────────────────────────────────────────────────── */
:root {
    --bg-primary:    #0f1117;
    --bg-secondary:  #161b22;
    --bg-card:       #1c2333;
    --accent:        #58a6ff;
    --accent-glow:   rgba(88, 166, 255, 0.15);
    --accent-2:      #7ee787;
    --text-primary:  #e6edf3;
    --text-muted:    #8b949e;
    --border:        #30363d;
    --danger:        #f85149;
    --warning:       #d29922;
    --radius:        12px;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* ── Scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ── Sidebar styling ─────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--accent) !important;
}

/* ── Chat message containers ─────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    margin-bottom: 1rem !important;
    backdrop-filter: blur(12px);
}

/* ── Status update badge ─────────────────────────────────────────────────── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    margin: 3px 0;
    animation: fadeInSlide 0.35s ease-out;
}

.status-badge.planning    { background: rgba(210, 153, 34, 0.12); color: #d29922; border: 1px solid rgba(210,153,34,0.25); }
.status-badge.searching   { background: rgba(88, 166, 255, 0.10); color: #58a6ff; border: 1px solid rgba(88,166,255,0.25); }
.status-badge.fetching    { background: rgba(188, 140, 255, 0.10); color: #bc8cff; border: 1px solid rgba(188,140,255,0.25); }
.status-badge.selecting   { background: rgba(126, 231, 135, 0.10); color: #7ee787; border: 1px solid rgba(126,231,135,0.25); }
.status-badge.generating  { background: rgba(255, 123, 114, 0.10); color: #ff7b72; border: 1px solid rgba(255,123,114,0.25); }
.status-badge.done        { background: rgba(126, 231, 135, 0.12); color: #7ee787; border: 1px solid rgba(126,231,135,0.30); }
.status-badge.error       { background: rgba(248, 81, 73, 0.12); color: #f85149; border: 1px solid rgba(248,81,73,0.30); }

@keyframes fadeInSlide {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Citation card ───────────────────────────────────────────────────────── */
.citation-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 16px;
    margin: 6px 0;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.citation-card:hover {
    border-color: var(--accent);
    box-shadow: 0 0 12px var(--accent-glow);
}
.citation-card a {
    color: var(--accent) !important;
    text-decoration: none;
    font-weight: 600;
    font-size: 0.9rem;
}
.citation-card a:hover { text-decoration: underline; }
.citation-card .domain {
    color: var(--text-muted);
    font-size: 0.78rem;
    margin-top: 2px;
}

/* ── Session card ────────────────────────────────────────────────────────── */
.session-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.83rem;
    color: var(--text-muted);
    word-break: break-all;
}

/* ── Pulse animation for "thinking" dot ──────────────────────────────────── */
.pulse-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    animation: pulse 1.2s infinite ease-in-out;
}
.pulse-dot.gold   { background: #d29922; }
.pulse-dot.blue   { background: #58a6ff; }
.pulse-dot.purple { background: #bc8cff; }
.pulse-dot.green  { background: #7ee787; }
.pulse-dot.red    { background: #ff7b72; }

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.4; transform: scale(0.7); }
}

/* ── Streamlit tweaks ────────────────────────────────────────────────────── */
.stChatInput textarea { font-family: 'Inter', sans-serif !important; }

/* Hero header */
.hero-title {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #58a6ff 0%, #bc8cff 50%, #7ee787 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.hero-sub {
    color: var(--text-muted);
    font-size: 0.92rem;
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: render a status badge
# ─────────────────────────────────────────────────────────────────────────────
STAGE_META = {
    "planning":   ("🧠", "gold",   "Planning research strategy…"),
    "searching":  ("🔍", "blue",   "Searching the web…"),
    "fetching":   ("🌐", "purple", "Fetching sources…"),
    "selecting":  ("📊", "green",  "Selecting relevant context…"),
    "generating": ("✨", "red",    "Generating grounded answer…"),
    "done":       ("✅", "green",  "Research complete"),
    "error":      ("❌", "red",    "An error occurred"),
}


def _status_html(stage: str, detail: str = "") -> str:
    """Return an HTML badge for the given pipeline stage."""
    icon, color, default_text = STAGE_META.get(stage, ("⚙️", "blue", stage))
    text = detail or default_text
    return (
        f'<div class="status-badge {stage}">'
        f'<span class="pulse-dot {color}"></span>'
        f'{icon}&ensp;{text}</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper: render source citations
# ─────────────────────────────────────────────────────────────────────────────
def _render_citations(sources: list):
    """Render citation cards for each source used."""
    if not sources:
        return
    st.markdown("#### 📚 Sources Used")
    cols = st.columns(min(len(sources), 3))
    for idx, src in enumerate(sources):
        with cols[idx % 3]:
            title = src.get("title", "Untitled")
            url   = src.get("url", "#")
            domain = src.get("domain", "")
            st.markdown(
                f'<div class="citation-card">'
                f'<a href="{url}" target="_blank">{title}</a>'
                f'<div class="domain">{domain}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Helper: update rolling summary (mirrors app.py logic)
# ─────────────────────────────────────────────────────────────────────────────
def _update_rolling_summary(session_id: str, current_summary: str, query: str, answer_text: str):
    """Generate and persist an updated conversation summary."""
    prompt = f"""You are maintaining a concise summary of a research conversation.
Current Summary:
{current_summary if current_summary else "No previous summary."}

New Interaction:
User: {query}
Agent: {answer_text}

Task: Write a highly concise updated summary capturing the main facts discussed so far. Do not exceed 2 paragraphs."""
    try:
        new_summary = generate_answer(prompt)
        memory.update_session_summary(session_id, new_summary)
    except Exception:
        pass  # Non-critical; swallow silently


# ─────────────────────────────────────────────────────────────────────────────
# Core: run pipeline with live status streaming
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline_streaming(session_id: str, question: str, status_container):
    """
    Execute the research pipeline and stream status updates into
    *status_container* (a Streamlit container/placeholder).

    Returns:
        (final_answer, sources, selected_chunks)
    """
    try:
        # ── Stage 1: Planning ────────────────────────────────────────────
        status_container.markdown(_status_html("planning", "Loading session context & planning…"), unsafe_allow_html=True)
        time.sleep(0.3)

        summary = memory.get_session_summary(session_id)
        recent_history = memory.get_session_history(session_id, limit=2)

        # ── Stage 2: Web search ──────────────────────────────────────────
        status_container.markdown(
            _status_html("planning", "Session loaded") + _status_html("searching"),
            unsafe_allow_html=True,
        )
        search_results = web_search(query=question, max_results=3)

        opened_urls = []
        sources = []

        # ── Stage 3: Fetching ────────────────────────────────────────────
        if search_results:
            badges = (
                _status_html("planning", "Session loaded")
                + _status_html("searching", f"Found {len(search_results)} results")
            )
            for i, result in enumerate(search_results):
                badges_with_fetch = badges + _status_html("fetching", f"Fetching {i+1}/{len(search_results)}: {result['title'][:50]}…")
                status_container.markdown(badges_with_fetch, unsafe_allow_html=True)

                fetch_data = fetch_and_extract(result["url"])
                opened_urls.append(result["url"])
                sources.append({
                    "title": result["title"],
                    "url": result["url"],
                    "domain": result["domain"],
                    "snippet": result["snippet"],
                    "extracted_text": fetch_data.get("extracted_text", ""),
                    "retrieved_at": fetch_data.get("retrieved_at", ""),
                })
        else:
            badges = (
                _status_html("planning", "Session loaded")
                + _status_html("searching", "No results found — using existing context")
            )
            status_container.markdown(badges, unsafe_allow_html=True)

        # ── Stage 4: Context selection ───────────────────────────────────
        base_badges = (
            _status_html("planning", "Session loaded")
            + _status_html("searching", f"Found {len(search_results)} results")
            + _status_html("fetching", f"Fetched {len(sources)} sources")
        )
        status_container.markdown(base_badges + _status_html("selecting"), unsafe_allow_html=True)

        selected_chunks = build_context(question, sources, max_chars=8000)
        selected_snippets = [c["text"] for c in selected_chunks]

        # ── Stage 5: Answer generation ───────────────────────────────────
        status_container.markdown(
            base_badges
            + _status_html("selecting", f"Selected {len(selected_chunks)} context chunks")
            + _status_html("generating"),
            unsafe_allow_html=True,
        )

        prompt = _build_answer_prompt(summary, recent_history, selected_chunks, question)
        final_answer = generate_answer(prompt)

        # ── Stage 6: Persist to memory ───────────────────────────────────
        search_queries = [question]
        memory.save_turn(session_id, question, search_queries, opened_urls, selected_snippets, final_answer)
        _update_rolling_summary(session_id, summary, question, final_answer)

        # ── Done badge ───────────────────────────────────────────────────
        status_container.markdown(
            base_badges
            + _status_html("selecting", f"Selected {len(selected_chunks)} chunks")
            + _status_html("generating", "Answer ready")
            + _status_html("done"),
            unsafe_allow_html=True,
        )

        return final_answer, sources, selected_chunks

    except Exception as exc:
        status_container.markdown(_status_html("error", str(exc)[:120]), unsafe_allow_html=True)
        return f"⚠️ Pipeline error: {exc}", [], []


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build the grounded-answer prompt (mirrors app.py)
# ─────────────────────────────────────────────────────────────────────────────
def _build_answer_prompt(summary, recent_history, selected_chunks, question):
    prompt = """You are a Deep Research Agent. Your task is to answer the user's question based STRICTLY on the provided context.

REQUIREMENTS:
1. NO HALLUCINATIONS: You must base your answer ONLY on the provided context. Do not use outside knowledge.
2. EXPLICIT UNCERTAINTY: If the context does not contain enough evidence to fully answer the question, state that clearly. If the evidence is weak, note that it is weak.
3. CITATIONS: You MUST include inline citations for every claim. Use the exact format: [Title — Domain](URL).
4. CONFLICTS: If different sources provide conflicting information, explicitly mention the conflict.
5. FORMAT: Use clean Markdown for readability.

"""
    if summary:
        prompt += f"CONVERSATION SUMMARY SO FAR:\n{summary}\n\n"

    if recent_history:
        prompt += "RECENT HISTORY:\n"
        for turn in recent_history:
            prompt += f"User: {turn['query']}\nAgent: {turn['final_answer']}\n\n"

    prompt += "NEW SOURCE CONTEXT:\n"
    for i, chunk in enumerate(selected_chunks):
        prompt += f"--- Source {i+1} ---\n"
        prompt += f"Title: {chunk['source_title']}\n"
        prompt += f"Domain: {chunk['source_domain']}\n"
        prompt += f"URL: {chunk['source_url']}\n"
        prompt += f"Content Snippet:\n{chunk['text']}\n\n"

    prompt += f"USER QUESTION: {question}\n\nANSWER:\n"
    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────────────────────────────────────
def _init_session_state():
    """Ensure all session-state keys exist."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = memory.create_session()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "sources_log" not in st.session_state:
        st.session_state.sources_log = []  # list of source-lists, one per turn


_init_session_state()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title">🔬 Deep Research</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Framework-free AI Research Agent</div>', unsafe_allow_html=True)

    st.divider()

    # Session info
    st.markdown("### 🆔 Session")
    st.markdown(
        f'<div class="session-card"><b>ID:</b> <code>{st.session_state.session_id[:8]}…</code></div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # Previous turns
    st.markdown("### 📝 Conversation Turns")
    if st.session_state.messages:
        user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
        if user_msgs:
            for idx, msg in enumerate(user_msgs, 1):
                preview = msg["content"][:60] + ("…" if len(msg["content"]) > 60 else "")
                st.markdown(
                    f'<div class="session-card"><b>Turn {idx}:</b> {preview}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No turns yet — ask a question to begin.")
    else:
        st.caption("No turns yet — ask a question to begin.")

    st.divider()

    # Clear chat
    if st.button("🗑️  Clear Chat & Start New Session", use_container_width=True):
        st.session_state.session_id = memory.create_session()
        st.session_state.messages = []
        st.session_state.sources_log = []
        st.rerun()

    st.divider()

    # Tech stack
    st.markdown("### ⚙️ Stack")
    st.markdown("""
    - 🔎 **Search** — Tavily API
    - 🤖 **LLM** — Gemini 2.5 Flash
    - 🗄️ **Memory** — SQLite
    - 📜 **Scraper** — Trafilatura
    - 🚫 No LangChain / CrewAI / LlamaIndex
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Main Chat Area
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Deep Research Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Ask any research question — I\'ll search the web, read sources, '
    'and generate a grounded, cited answer in real time.</div>',
    unsafe_allow_html=True,
)

# Render chat history
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🔬"):
        st.markdown(msg["content"], unsafe_allow_html=True)

        # If assistant message and we have sources for this turn, show them
        if msg["role"] == "assistant":
            turn_index = msg.get("turn_index")
            if turn_index is not None and turn_index < len(st.session_state.sources_log):
                srcs = st.session_state.sources_log[turn_index]
                if srcs:
                    with st.expander("📚 View Sources", expanded=False):
                        _render_citations(srcs)


# ─────────────────────────────────────────────────────────────────────────────
# Chat Input & Pipeline Execution
# ─────────────────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask a research question…"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    # Display assistant response area
    with st.chat_message("assistant", avatar="🔬"):
        status_placeholder = st.empty()
        answer_placeholder = st.empty()

        # Run pipeline with live status
        final_answer, sources, chunks = run_pipeline_streaming(
            st.session_state.session_id,
            user_input,
            status_placeholder,
        )

        # Render the final answer
        answer_placeholder.markdown(final_answer)

        # Show citation cards
        if sources:
            _render_citations(sources)

    # Persist to session state
    turn_idx = len(st.session_state.sources_log)
    st.session_state.sources_log.append(sources)
    st.session_state.messages.append({
        "role": "assistant",
        "content": final_answer,
        "turn_index": turn_idx,
    })

    # Force rerun so sidebar updates with new turn
    st.rerun()
