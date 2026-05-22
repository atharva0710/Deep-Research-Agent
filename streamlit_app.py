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


def _status_html(stage: str, detail: str = "", active: bool = True) -> str:
    """Return an HTML badge for the given pipeline stage."""
    icon, color, default_text = STAGE_META.get(stage, ("⚙️", "blue", stage))
    text = detail or default_text
    pulse = f'<span class="pulse-dot {color}"></span>' if active else ""
    return (
        f'<div class="status-badge {stage}">'
        f'{pulse}'
        f'{icon}&ensp;{text}</div>'
    )


class ProgressStreamer:
    """Reusable progress manager for Streamlit operational updates."""
    def __init__(self, container):
        self.container = container
        self.history = []
        self.current_stage_start = 0.0

    def start_step(self, stage: str, detail: str):
        """Starts a new step and marks previous as inactive with timing."""
        now = time.time()
        if self.history:
            # Finalize previous step with elapsed time
            prev_stage, prev_detail, _ = self.history[-1]
            elapsed = now - self.current_stage_start
            self.history[-1] = (prev_stage, f"{prev_detail} ({elapsed:.1f}s)", False)
        
        self.current_stage_start = now
        self.history.append((stage, detail, True))
        self._render()

    def update_step(self, stage: str, detail: str):
        """Updates the current active step without changing timing."""
        if self.history:
            self.history[-1] = (stage, detail, True)
            self._render()

    def finish(self):
        """Marks the final step as complete."""
        now = time.time()
        if self.history:
            prev_stage, prev_detail, _ = self.history[-1]
            elapsed = now - self.current_stage_start
            self.history[-1] = (prev_stage, f"{prev_detail} ({elapsed:.1f}s)", False)
        
        self.history.append(("done", "Research complete", False))
        self._render()

    def error(self, err_msg: str):
        self.history.append(("error", err_msg, False))
        self._render()

    def _render(self):
        html_content = "".join([_status_html(s, d, a) for s, d, a in self.history])
        self.container.markdown(html_content, unsafe_allow_html=True)


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


def generate_research_plan(session_id: str, current_query: str) -> dict:
    """
    Generates a structured research plan and standalone search query based on
    conversation history.
    """
    summary = memory.get_session_summary(session_id)
    recent_history = memory.get_session_history(session_id, limit=2)
    
    prompt = f"""You are an expert research planner and search optimizer.
Your task is to analyze the conversation history and the user's new query, and produce a research plan and standalone search query.

INSTRUCTIONS:
1. Rewrite the query to resolve all pronouns/coreferences, producing a standalone query suitable for Tavily.
2. Outline a 1-2 sentence research strategy describing what you will search for and analyze.
3. Your output must be formatted EXACTLY as follows:
QUERY: <standalone search query>
PLAN: <research strategy>

Do not include any other conversational filler, markdown formatting, or labels.

CONVERSATION SUMMARY:
{summary if summary else "None"}

RECENT HISTORY:
"""
    for turn in recent_history:
        prompt += f"User: {turn['query']}\nAgent: {turn['final_answer']}\n\n"
        
    prompt += f"USER'S NEW QUERY: {current_query}\n\nRESEARCH PLAN:"
    
    plan_obj = {
        "search_query": current_query,
        "plan_text": f"Investigating '{current_query}' directly by fetching top search results."
    }
    
    try:
        response = generate_answer(prompt).strip()
        
        # Parse the structured response
        query_match = re.search(r'QUERY:\s*(.*)', response, re.IGNORECASE)
        plan_match = re.search(r'PLAN:\s*(.*)', response, re.IGNORECASE)
        
        if query_match:
            plan_obj["search_query"] = query_match.group(1).strip()
        if plan_match:
            plan_obj["plan_text"] = plan_match.group(1).strip()
            
        return plan_obj
    except Exception:
        return plan_obj


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
    streamer = ProgressStreamer(status_container)
    try:
        # ── Stage 1: Planning ────────────────────────────────────────────
        streamer.start_step("planning", "Loading session context & planning…")

        summary = memory.get_session_summary(session_id)
        recent_history = memory.get_session_history(session_id, limit=2)

        # Stage 1.5: Query & Plan Generation
        streamer.update_step("planning", "Formulating strategic research plan…")
        plan_data = generate_research_plan(session_id, question)
        search_query = plan_data["search_query"]
        research_plan = plan_data["plan_text"]
        
        # Display the strategic plan to the user in real-time
        streamer.update_step("planning", f"**Plan**: {research_plan}  \n**Optimized Query**: '{search_query}'")

        # ── Stage 2: Web search ──────────────────────────────────────────
        streamer.start_step("searching", "Searching the web…")
        search_results = web_search(query=search_query, max_results=7)

        opened_urls = []
        sources = []

        # ── Stage 3: Fetching ────────────────────────────────────────────
        if search_results:
            streamer.update_step("searching", f"Found {len(search_results)} potential sources")
            for i, result in enumerate(search_results):
                streamer.start_step("fetching", f"Fetching {i+1}/{len(search_results)}: {result['title'][:40]}…")

                # Fetch page content and build structured metadata
                fetch_data = fetch_and_extract(result["url"], result["title"], result["domain"])
                opened_urls.append(result["url"])
                
                # Append relevance score
                fetch_data["score"] = result.get("score", 0.0)
                sources.append(fetch_data)
                
            streamer.update_step("fetching", f"Fetched {len(sources)} sources")
        else:
            streamer.update_step("searching", "No results found — using existing context")

        # ── Stage 4: Context selection ───────────────────────────────────
        streamer.start_step("selecting", "Selecting & ranking relevant context…")

        selected_chunks = build_context(question, sources, max_chars=8000)
        selected_snippets = [c["text"] for c in selected_chunks]
        
        streamer.update_step("selecting", f"Selected {len(selected_chunks)} diverse context chunks")

        # ── Stage 5: Answer generation ───────────────────────────────────
        streamer.start_step("generating", "Generating grounded answer…")

        from context_builder import prepare_context_for_prompt
        prompt = prepare_context_for_prompt(
            query=question,
            selected_chunks=selected_chunks,
            rolling_summary=summary,
            recent_history=recent_history,
            max_chars=12000
        )
        final_answer = generate_answer(prompt)

        # ── Stage 6: Persist to memory ───────────────────────────────────
        search_queries = [search_query]
        memory.save_turn(session_id, question, search_queries, opened_urls, selected_snippets, final_answer)
        _update_rolling_summary(session_id, summary, question, final_answer)

        # ── Done badge ───────────────────────────────────────────────────
        streamer.finish()

        return final_answer, sources, selected_chunks

    except Exception as exc:
        streamer.error(f"Pipeline error: {str(exc)[:120]}")
        return f"⚠️ Pipeline error: {exc}", [], []


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
    - 🤖 **LLM** — Groq (openai/gpt-oss-120b)
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
