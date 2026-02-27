import time
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Obsidian", page_icon="🪨", layout="centered")

# -----------------------------
# Obsidian theme (CSS)
# -----------------------------
st.markdown(
    """
    <style>
      /* page background */
      html, body, [class*="stApp"] {
        background: radial-gradient(1200px 600px at 50% 0%, #121318 0%, #07070A 60%, #000000 100%) !important;
        color: #E7E7E7 !important;
      }

      /* container width */
      .block-container { max-width: 980px; padding-top: 1.2rem; }

      /* remove some default padding around chat input */
      section.main > div { padding-bottom: 3.5rem; }

      /* headers */
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

      /* chat bubbles: make them feel like ChatGPT */
      [data-testid="stChatMessage"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 12px 14px;
        margin-bottom: 10px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
      }

      /* user message slightly different */
      [data-testid="stChatMessage"][aria-label="user"] {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.14);
      }

      /* links */
      a { color: #D7D7D7 !important; text-decoration: none; }
      a:hover { text-decoration: underline; }

      /* sidebar */
      [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02)) !important;
        border-right: 1px solid rgba(255,255,255,0.08);
      }

      /* buttons */
      .stButton button {
        background: rgba(255,255,255,0.10) !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
        color: #F0F0F0 !important;
        border-radius: 12px !important;
      }
      .stButton button:hover {
        background: rgba(255,255,255,0.14) !important;
        border: 1px solid rgba(255,255,255,0.24) !important;
      }

      /* inputs */
      input, textarea {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.14) !important;
        color: #F0F0F0 !important;
        border-radius: 12px !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Load logo (optional, but recommended)
# -----------------------------
LOGO_PATH = Path(__file__).parent / "logo.png"

# -----------------------------
# Knowledge base (your current prompts)
# -----------------------------
KB = [
    {
        "id": "llc-texas",
        "category": "Business Setup & Management",
        "question": "How do I properly set up an LLC in Texas?",
        "answer": (
            "Step 1: Choose a unique business name (including an LLC designator) and check availability.\n"
            "Step 2: File a Certificate of Formation – Limited Liability Company (Form 205) with the Texas Secretary of State (SOS).\n"
            "Step 3: Designate a registered agent and registered office in Texas.\n"
            "Step 4: Draft a Company/Operating Agreement to govern internal affairs (strongly recommended).\n"
            "Step 5: Maintain statutory records and meet ongoing obligations such as Texas franchise tax filings."
        ),
        "sources": [
            "https://www.sos.state.tx.us/corp/forms/205_boc.pdf?utm_source=chatgpt.com",
            "https://www.sos.state.tx.us/corp/businessstructure.shtml?utm_source=chatgpt.com",
        ],
    },
    {
        "id": "legal-steps-start-business",
        "category": "Business Setup & Management",
        "question": "What legal steps do I need to take to start my business?",
        "answer": (
            "Pick the right entity type, file the formation paperwork if needed, maintain a registered agent/office, "
            "prepare governance documents (operating agreement/bylaws), obtain an EIN, and complete applicable tax registrations, "
            "licenses/permits, and ongoing filings."
        ),
        "sources": [
            "https://www.sos.state.tx.us/corp/businessstructure.shtml?utm_source=chatgpt.com",
        ],
    },
    {
        "id": "business-license-dfw",
        "category": "Business Setup & Management",
        "question": "Do I need a business license in DFW?",
        "answer": (
            "It depends on the city/county and your industry. Texas entity formation is separate from local licensing, "
            "so check the specific city (Dallas/Fort Worth/etc.) and any state agency rules tied to your business activity."
        ),
        "sources": [],
    },
    {
        "id": "management-system-team-organized",
        "category": "Business Setup & Management",
        "question": "What management system is best for keeping my team organized?",
        "answer": (
            "There’s no single required system. Keep required legal records organized, and operationally use a project tracker "
            "(Asana/Trello), team comms (Slack), and a secure document repository for official records."
        ),
        "sources": [],
    },
    {
        "id": "stay-compliant-state-federal",
        "category": "Business Setup & Management",
        "question": "How do I make sure my business stays compliant with state and federal laws?",
        "answer": (
            "Keep your registered agent/office current, document key decisions, maintain required records, file required state reports "
            "(including franchise tax where applicable), and comply with federal tax/employment rules relevant to your business."
        ),
        "sources": [
            "https://www.sos.state.tx.us/corp/businessstructure.shtml?utm_source=chatgpt.com",
        ],
    },
    {
        "id": "insurance-needed-start-business",
        "category": "Business Setup & Management",
        "question": "What insurance do I need when starting a business?",
        "answer": (
            "It depends on operations and risk. Common coverages include general liability, professional liability, property coverage, "
            "workers’ comp (if applicable), and cyber coverage if you handle sensitive data."
        ),
        "sources": [],
    },
    {
        "id": "business-start-checklist",
        "category": "Business Setup & Management",
        "question": "Can you create a checklist for starting my business the right way?",
        "answer": (
            "Checklist: choose entity → choose/check name → appoint registered agent/office → file formation (LLC: Form 205) → "
            "draft operating agreement/bylaws → get EIN → open bank account → set up accounting → tax registrations "
            "(franchise/sales if applicable) → licenses/permits → insurance → maintain records & ongoing filings."
        ),
        "sources": [
            "https://www.sos.state.tx.us/corp/businessstructure.shtml?utm_source=chatgpt.com",
            "https://www.sos.state.tx.us/corp/forms/205_boc.pdf?utm_source=chatgpt.com",
        ],
    },
]

# -----------------------------
# Retrieval (TF-IDF)
# -----------------------------
texts = [f"{x['question']} {x['answer']}" for x in KB]
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True,
    max_df=0.95,
)
doc_vectors = vectorizer.fit_transform(texts)

def retrieve(query: str, top_k: int = 3):
    if not query.strip():
        return []
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, doc_vectors)[0]
    order = np.argsort(-sims)
    return [(int(i), float(sims[int(i)])) for i in order[:top_k] if sims[int(i)] > 0]

def thinking_dots(seconds: float = 0.9):
    spot = st.empty()
    start = time.time()
    dots = ["", ".", "..", "..."]
    i = 0
    while time.time() - start < seconds:
        spot.markdown(f"**Thinking{dots[i % 4]}**")
        time.sleep(0.18)
        i += 1
    spot.empty()

def stream_response(text: str, delay: float = 0.018):
    box = st.empty()
    out = ""
    for token in text.split(" "):
        out += token + " "
        box.markdown(out)
        time.sleep(delay)

# -----------------------------
# Session state chat history
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hi — I’m **Obsidian**. Ask a Texas business formation/compliance question and I’ll answer from our curated knowledge base."
    }]

# -----------------------------
# Header + logo
# -----------------------------
left, right = st.columns([1, 3], vertical_alignment="center")
with left:
    if LOGO_PATH.exists():
        img = Image.open(LOGO_PATH)
        st.image(img, use_container_width=True)
with right:
    st.markdown(
        """
        <div class="opc-header">
          <div>
            <div class="opc-title">Obsidian</div>
            <div class="opc-sub">Obsidian Partners & Co. • AI-style advisor • Curated answers • Not legal advice</div>
          </div>
          <div class="badge">🧠 Obsidian-Reasoner</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.markdown("### Controls")
    top_k = st.slider("Retriever: top_k", 1, 5, 3)
    show_debug = st.toggle("Show retrieval details", value=False)
    st.markdown("---")
    if st.button("Clear chat"):
        st.session_state.messages = st.session_state.messages[:1]
        st.rerun()

# -----------------------------
# Render chat messages
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# Chat input (ChatGPT style)
# -----------------------------
prompt = st.chat_input("Message Obsidian…")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Thinking indicator
        with st.spinner("Thinking…"):
            thinking_dots(0.9)

        hits = retrieve(prompt, top_k=top_k)
        if not hits:
            final = "I couldn’t find a close match in the current knowledge base. Try rephrasing or ask one of the supported prompts."
            st.markdown(final)
            st.session_state.messages.append({"role": "assistant", "content": final})
        else:
            best_idx, best_score = hits[0]
            item = KB[best_idx]

            # Stream response like ChatGPT
            stream_response(item["answer"], delay=0.016)

            # Sources footer
            sources = item.get("sources", [])
            if sources:
                st.markdown("**Sources**")
                for url in sources:
                    st.markdown(f"- [{url}]({url})")

            # Debug (optional)
            if show_debug:
                rows = []
                for r, (idx, score) in enumerate(hits, start=1):
                    rows.append({
                        "Rank": r,
                        "Question": KB[idx]["question"],
                        "Category": KB[idx].get("category", ""),
                        "Score": round(score, 3),
                    })
                st.markdown("**Retrieval details**")
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Save assistant message to history
            stored = item["answer"]
            if sources:
                stored += "\n\n**Sources**\n" + "\n".join([f"- {s}" for s in sources])
            st.session_state.messages.append({"role": "assistant", "content": stored})