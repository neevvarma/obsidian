import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# Page config + theme
# =========================================================
st.set_page_config(page_title="Obsidian", page_icon="🪨", layout="centered")

st.markdown(
    """
    <style>
      html, body, [class*="stApp"] {
        background: radial-gradient(1200px 600px at 50% 0%, #121318 0%, #07070A 60%, #000000 100%) !important;
        color: #E7E7E7 !important;
      }
      .block-container { max-width: 980px; padding-top: 1.1rem; }
      section.main > div { padding-bottom: 4.5rem; }

      .opc-header {
        display:flex; align-items:center; justify-content:space-between;
        gap:14px; margin: 0.25rem 0 1rem 0;
      }
      .opc-title { font-size: 1.25rem; font-weight: 800; letter-spacing: 0.6px; }
      .opc-sub { font-size: 0.92rem; color: rgba(220,220,220,0.7); margin-top: 0.15rem; }
      .badge {
        border: 1px solid rgba(255,255,255,0.18);
        background: rgba(255,255,255,0.06);
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 0.82rem;
        color: rgba(240,240,240,0.85);
      }

      [data-testid="stChatMessage"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 12px 14px;
        margin-bottom: 10px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
      }
      [data-testid="stChatMessage"][aria-label="user"] {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.16);
      }

      a { color: #D7D7D7 !important; text-decoration: none; }
      a:hover { text-decoration: underline; }

      [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02)) !important;
        border-right: 1px solid rgba(255,255,255,0.08);
      }

      .hint { color: rgba(230,230,230,0.72); font-size: 0.92rem; }
      .small { color: rgba(230,230,230,0.70); font-size: 0.86rem; }
      .divider { height: 1px; background: rgba(255,255,255,0.08); margin: 10px 0; }

      .stButton button {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.14) !important;
        color: #F0F0F0 !important;
        border-radius: 999px !important;
        padding: 0.4rem 0.9rem !important;
        width: 100%;
      }
      .stButton button:hover {
        background: rgba(255,255,255,0.12) !important;
        border: 1px solid rgba(255,255,255,0.22) !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
LOGO_PATH = APP_DIR / "logo.png"  # optional

QA_CSV = DATA_DIR / "kb_qa.csv"
CHUNKS_CSV = DATA_DIR / "kb_chunks.csv"


# =========================================================
# Small helpers
# =========================================================
def safe_str(x) -> str:
    return "" if x is None else str(x)

def parse_pipe_list(value: str):
    if not isinstance(value, str):
        return []
    return [x.strip() for x in value.split("|") if x.strip()]

def is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())

def split_sentences(text: str):
    # simple + robust enough for PDFs
    text = normalize(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) > 18]

# Simple "human intent"
GREETING_PAT = re.compile(r"\b(hi|hello|hey|yo|good morning|good afternoon|good evening)\b", re.I)
HELP_PAT = re.compile(r"\b(help|what can you do|how do i use this|what should i ask)\b", re.I)
THANKS_PAT = re.compile(r"\b(thanks|thank you|thx|appreciate it)\b", re.I)

def detect_intent(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "empty"
    if THANKS_PAT.search(t):
        return "thanks"
    if HELP_PAT.search(t):
        return "help"
    if GREETING_PAT.search(t):
        return "greeting"
    return "query"


# =========================================================
# Load data
# =========================================================
@st.cache_data(show_spinner=False)
def load_qa_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).fillna("")
    for col in ["id", "category", "question", "answer", "sources", "source_pages", "followups"]:
        if col not in df.columns:
            df[col] = ""
    return df

@st.cache_data(show_spinner=False)
def load_chunks_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).fillna("")
    for col in ["chunk_id", "doc_title", "section", "page_start", "page_end", "text", "source_url"]:
        if col not in df.columns:
            df[col] = ""
    return df

if not QA_CSV.exists() or not CHUNKS_CSV.exists():
    st.error(
        "Missing CSV files.\n\nExpected:\n- data/kb_qa.csv\n- data/kb_chunks.csv\n\n"
        "Fix: put both files in the repo under /data and restart."
    )
    st.stop()

qa_df = load_qa_df(QA_CSV)
chunks_df = load_chunks_df(CHUNKS_CSV)


# =========================================================
# Build vector indexes (fast + cached)
# =========================================================
@st.cache_resource(show_spinner=False)
def build_index_for_qa(df: pd.DataFrame):
    texts = (df["category"].astype(str) + " " + df["question"].astype(str) + " " + df["answer"].astype(str)).tolist()
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True, max_df=0.95)
    mat = vec.fit_transform(texts)
    return vec, mat

@st.cache_resource(show_spinner=False)
def build_index_for_chunks(df: pd.DataFrame):
    texts = (df["doc_title"].astype(str) + " " + df["text"].astype(str)).tolist()
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True, max_df=0.95)
    mat = vec.fit_transform(texts)
    return vec, mat

qa_vec, qa_mat = build_index_for_qa(qa_df)
chunk_vec, chunk_mat = build_index_for_chunks(chunks_df)

def retrieve(vec, mat, query: str, top_k: int):
    query = normalize(query)
    if not query:
        return []
    qv = vec.transform([query])
    sims = cosine_similarity(qv, mat)[0]
    order = np.argsort(-sims)
    hits = [(int(i), float(sims[int(i)])) for i in order[:top_k]]
    return hits


# =========================================================
# Answer formatting
# =========================================================
def render_sources_block(sources: list[str], pages: str = "") -> str:
    if not sources and not pages:
        return ""
    out = "**Sources**\n"
    for s in sources:
        if is_url(s):
            out += f"- [{s}]({s})\n"
        else:
            out += f"- {s}\n"
    if pages.strip():
        out += f"\n<span class='small'>Pages: {pages.strip()}</span>"
    return out

def qa_answer(row: pd.Series) -> tuple[str, str]:
    answer = safe_str(row["answer"]).strip()
    sources = parse_pipe_list(safe_str(row.get("sources", "")))
    pages = safe_str(row.get("source_pages", ""))
    return answer, render_sources_block(sources, pages)

def doc_summarize_answer(query: str, chunk_hits: list[tuple[int, float]], max_chunks: int = 5) -> tuple[str, str]:
    """
    Create a coherent answer that addresses the question:
    - take top chunks
    - extract top sentences relevant to query
    - format as step-by-step
    """
    if not chunk_hits:
        return (
            "I couldn’t find anything relevant in the current documents yet. Try rephrasing your question.",
            ""
        )

    picked = [i for i, _ in chunk_hits[:max_chunks]]
    texts = []
    cite_lines = []

    for rank, idx in enumerate(picked, start=1):
        row = chunks_df.iloc[idx]
        txt = safe_str(row["text"])
        texts.append(txt)

        doc = safe_str(row["doc_title"]).strip()
        p1 = safe_str(row.get("page_start", "")).strip()
        p2 = safe_str(row.get("page_end", "")).strip()
        if p1 and p2 and p1 != p2:
            cite_lines.append(f"[{rank}] {doc} (pp. {p1}-{p2})")
        elif p1:
            cite_lines.append(f"[{rank}] {doc} (p. {p1})")
        else:
            cite_lines.append(f"[{rank}] {doc}")

    # Sentence scoring using a small TF-IDF over candidate sentences
    sentences = []
    for t in texts:
        sentences.extend(split_sentences(t))

    # Guard
    sentences = [s for s in sentences if 30 <= len(s) <= 260]
    if not sentences:
        return (
            "I found relevant material, but it wasn’t extractable cleanly. Try asking more specifically (e.g., “LLC steps in Texas”).",
            "**Sources (internal)**\n" + "\n".join([f"- {c}" for c in cite_lines])
        )

    local_vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True, max_df=0.98)
    smat = local_vec.fit_transform(sentences + [query])
    qv = smat[-1]
    sent_mat = smat[:-1]
    sims = cosine_similarity(qv, sent_mat)[0]
    order = np.argsort(-sims)

    # Pick best distinct sentences
    chosen = []
    used = set()
    for i in order:
        s = sentences[int(i)]
        key = s[:60]
        if key in used:
            continue
        if sims[int(i)] < 0.05:
            break
        chosen.append(s)
        used.add(key)
        if len(chosen) >= 6:
            break

    # Turn into steps that actually address the query
    steps = []
    for n, s in enumerate(chosen[:5], start=1):
        # make it slightly more instructional
        s_clean = s
        if not s_clean.endswith((".", "!", "?")):
            s_clean += "."
        steps.append(f"**Step {n}:** {s_clean}")

    answer = "\n\n".join(steps)
    cites = "**Sources (internal citations)**\n" + "\n".join([f"- {c}" for c in cite_lines])
    return answer, cites


# =========================================================
# Suggestions (prompt chips)
# =========================================================
@st.cache_data(show_spinner=False)
def get_recommended_prompts(df: pd.DataFrame, n=8):
    qs = [q for q in df["question"].astype(str).tolist() if q.strip()]
    # pick short, high-coverage prompts
    qs = sorted(qs, key=lambda x: len(x))[: max(n * 3, 10)]
    priority = [
        "How do I properly set up an LLC in Texas?",
        "What are the main steps to start a business in Texas?",
        "Do I need a registered agent in Texas?",
        "What is the Texas franchise tax and when is it due?",
    ]
    out = []
    for p in priority:
        if p in qs and p not in out:
            out.append(p)
    for q in qs:
        if q not in out:
            out.append(q)
        if len(out) >= n:
            break
    return out

RECOMMENDED = get_recommended_prompts(qa_df, n=8)

def render_prompt_chips(prompts: list[str]):
    if not prompts:
        return
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='hint'><b>Try one of these:</b></div>", unsafe_allow_html=True)
    cols = st.columns(2)
    for i, p in enumerate(prompts):
        with cols[i % 2]:
            if st.button(p, key=f"chip_{i}"):
                st.session_state.pending_prompt = p


# =========================================================
# Header + Sidebar
# =========================================================
left, right = st.columns([1, 3], vertical_alignment="center")
with left:
    if LOGO_PATH.exists():
        st.image(Image.open(LOGO_PATH), width="stretch")
with right:
    st.markdown(
        """
        <div class="opc-header">
          <div>
            <div class="opc-title">Obsidian</div>
            <div class="opc-sub">Hybrid Advisor • Curated Q/A + Document RAG • Fast, helpful answers</div>
          </div>
          <div class="badge">🧠 Obsidian-Reasoner</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.markdown("### Retrieval Settings")
    qa_top_k = st.slider("Q/A top_k", 1, 8, 4)
    docs_top_k = st.slider("Docs top_k", 1, 10, 5)
    use_docs_fallback = st.toggle("Use docs fallback (RAG)", value=True)
    show_debug = st.toggle("Show debug (scores)", value=False)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("### What I’m good at")
    st.markdown(
        "- Texas business startup steps\n"
        "- LLC basics + registered agent\n"
        "- Tax/compliance basics (franchise tax, sales tax)\n"
        "- Planning + roadmap concepts",
    )

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.pending_prompt = None
        st.session_state.context = ""
        st.rerun()


# =========================================================
# Session state
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if "context" not in st.session_state:
    st.session_state.context = ""


# =========================================================
# Input collection (typed OR clicked)
# =========================================================
typed = st.chat_input("Message Obsidian…")

prompt = None
if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
elif typed:
    prompt = typed


# =========================================================
# If we have a prompt, generate a GOOD answer (fast)
# =========================================================
if prompt:
    prompt = normalize(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    intent = detect_intent(prompt)

    # Minimal “thinking” (fast)
    with st.spinner("Thinking…"):
        time.sleep(0.15)

    # Friendly handling
    if intent in ("greeting", "help"):
        msg = (
            "Hey 👋 I can help you with Texas LLC formation and practical small-business setup.\n\n"
            "Pick a prompt below, or ask in your own words (example: *“How do I set up an LLC in Texas?”*)."
        )
        st.session_state.messages.append({"role": "assistant", "content": msg})
    elif intent == "thanks":
        msg = "Anytime 🙂 Want to form an LLC, handle taxes, or plan your first 30 days? Pick a prompt below."
        st.session_state.messages.append({"role": "assistant", "content": msg})
    else:
        # 1) Q/A retrieval (best for accuracy)
        qa_hits = retrieve(qa_vec, qa_mat, prompt, qa_top_k)
        best_idx, best_score = qa_hits[0]

        # Tuned thresholds: prefer QA when we have a reasonable match
        QA_USE_THRESHOLD = 0.16

        if best_score >= QA_USE_THRESHOLD:
            row = qa_df.iloc[best_idx]
            answer, sources_block = qa_answer(row)

            final = answer
            if sources_block:
                final += "\n\n" + sources_block

            st.session_state.messages.append({"role": "assistant", "content": final})

        else:
            # 2) Docs fallback (better summarization, not random chunk dumping)
            if use_docs_fallback:
                doc_hits = retrieve(chunk_vec, chunk_mat, prompt, docs_top_k)
                answer, cites = doc_summarize_answer(prompt, doc_hits, max_chunks=min(6, docs_top_k))

                final = answer
                if cites:
                    final += "\n\n" + cites

                st.session_state.messages.append({"role": "assistant", "content": final})
            else:
                st.session_state.messages.append(
                    {"role": "assistant", "content": "I’m not confident enough to answer from the current Q/A. Try turning on docs fallback in the sidebar."}
                )

    # Rerun once to render chat cleanly after appending messages
    st.rerun()


# =========================================================
# Render conversation
# =========================================================
if not st.session_state.messages:
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": (
                "Hey — I’m **Obsidian**.\n\n"
                "Ask a question or click a prompt below. I’ll answer **step-by-step** and cite sources when available."
            ),
        }
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# =========================================================
# Prompt chips (always visible)
# =========================================================
render_prompt_chips(RECOMMENDED)

# Optional debug
if show_debug and st.session_state.messages:
    # show last QA retrieval info if last user message exists
    last_user = next((m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"), "")
    if last_user:
        qa_hits = retrieve(qa_vec, qa_mat, last_user, qa_top_k)
        dbg = []
        for r, (idx, score) in enumerate(qa_hits, start=1):
            dbg.append({
                "Rank": r,
                "Score": round(score, 3),
                "Question": safe_str(qa_df.iloc[idx]["question"]),
            })
        st.markdown("**Debug (Top Q/A matches)**")
        st.dataframe(pd.DataFrame(dbg), width="stretch", hide_index=True)