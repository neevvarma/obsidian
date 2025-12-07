import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Obsidian – Business Q&A (Texas)",
    page_icon="📘",
    layout="centered",
)

# -----------------------------
# In-code knowledge base (from your PDF)
# -----------------------------
KB = [
    {
        "id": "llc-texas",
        "category": "Business Setup & Management",
        "question": "How do I properly set up an LLC in Texas?",
        "answer": """1. Choose a unique business name (including an LLC designator) and check availability.
2. File a Certificate of Formation – Limited Liability Company (Form 205) with the Texas Secretary of State (“SOS”). The form is designed to satisfy the requirements of the Texas Business Organizations Code (TBOC), Title 3, Chapter 101.
3. Designate a registered agent and registered office in Texas, as required by TBOC (Title 1, Chapter 5).
4. (Strongly recommended though not always legally required) Draft a Company Agreement (operating agreement) to govern internal affairs and structure of the LLC.
5. After filing and approval, maintain statutory records (e.g., membership records) and meet ongoing obligations (such as franchise tax filings with the Texas Comptroller of Public Accounts).""",
        "sources": [
            "https://www.sos.state.tx.us/corp/forms/205_boc.pdf?utm_source=chatgpt.com",
            "https://www.sos.state.tx.us/corp/businessstructure.shtml?utm_source=chatgpt.com",
        ],
    },
    {
        "id": "legal-steps-start-business",
        "category": "Business Setup & Management",
        "question": "What legal steps do I need to take to start my business?",
        "answer": """At a high level, the legal steps include:
- Decide on the appropriate business entity (LLC, corporation, partnership). The SOS provides guidance distinguishing structures.
- If forming a domestic entity, file the required formation document (e.g., Certificate of Formation) with the Texas SOS for the chosen entity type.
- Ensure you have a registered agent and registered office in Texas (TBOC Title 1, Chapter 5).
- For corporations, adopt bylaws; for LLCs, draft a company/operating agreement.
- Issue evidences of ownership (e.g., membership interest, stock) as required by the entity type.
- Obtain any required federal employer identification number (EIN) from the IRS.
- Comply with state filings, tax registrations (e.g., sales tax, franchise tax), local licenses/permits, and other regulatory obligations.""",
        "sources": [],
    },
    {
        "id": "business-license-dfw",
        "category": "Business Setup & Management",
        "question": "Do I need a business license in DFW?",
        "answer": """TBOC does not govern local business licenses. Whether you need a business license in the Dallas–Fort Worth (DFW) area depends on the city or county jurisdiction and on your industry type (e.g., health services, food, construction). You will need to check:
- The specific city’s (e.g., Dallas or Fort Worth) business license/permit requirements.
- Any state regulatory licensing if your business activity triggers a state-licensing agency (e.g., professional services).""",
        "sources": [],
    },
    {
        "id": "management-system-team-organized",
        "category": "Business Setup & Management",
        "question": "What management system is best for keeping my team organized?",
        "answer": """There is no single “best” system mandated by law. From a legal perspective under TBOC you must maintain certain records and governance procedures:
- For LLCs: maintain records of members, capital contributions, distributions, and amendments as required by TBOC Title 3, Chapter 101.
- For corporations: maintain minute books, shareholder records, and stock transfer records as required under TBOC Title 2.

In practice, many businesses use project-management or collaboration tools (e.g., Asana, Trello, Slack) together with a secure document repository for legal records.""",
        "sources": [],
    },
    {
        "id": "stay-compliant-state-federal",
        "category": "Business Setup & Management",
        "question": "How do I make sure my business stays compliant with state and federal laws?",
        "answer": """From the state entity-law side (TBOC):
- Keep your entity’s registration current, including the registered agent and registered office (TBOC Title 1, Chapter 5).
- File amendments or changes when required (e.g., change of registered agent, change of entity name).
- Hold any required meetings (corporation) or document decisions (LLC) and keep records.
- Ensure you act within the powers granted to the entity by its governing documents and the law (TBOC Title 1, Chapter 2 covers general powers).

From the federal side:
- Comply with IRS rules for federal taxes (income tax, employment tax, self-employment tax).
- Comply with federal labor laws (e.g., Fair Labor Standards Act), employment tax withholding, and worker-classification rules.""",
        "sources": [],
    },
    {
        "id": "insurance-needed-start-business",
        "category": "Business Setup & Management",
        "question": "What insurance do I need when starting a business?",
        "answer": """TBOC does not specify insurance requirements. Insurance needs depend on your specific operations, risk profile, industry, and jurisdiction. Common insurance types to consider include:
- General liability
- Professional liability (errors & omissions)
- Workers’ compensation (if you have employees and state law requires it)
- Property insurance
- Business interruption insurance

You should also check:
- Whether your industry is regulated and mandates certain coverages.
- Lease or contract obligations that may require insurance.
- Any state or local laws (e.g., workers’ compensation rules) that mandate coverage.""",
        "sources": [],
    },
    {
        "id": "business-start-checklist",
        "category": "Business Setup & Management",
        "question": "Can you create a checklist for starting my business the right way?",
        "answer": """Here is a business start checklist focused on legal formation and compliance (you should add operations, marketing, etc., as needed):

1. Choose a business structure (LLC, corporation, partnership) and understand legal differences.
2. Choose a business name and check availability in Texas (SOS entity name search).
3. Designate a registered agent and registered office in Texas.
4. File the appropriate Certificate of Formation with the Texas SOS (for LLC: Form 205) under TBOC Title 3, Chapter 101.
5. Draft internal governance documents:
   - LLC: Company/Operating Agreement
   - Corporation: Bylaws and any shareholder agreements
6. Issue ownership interests (membership interests or shares) as required by the entity type.
7. Obtain an EIN from the IRS.
8. Register for relevant state taxes (e.g., franchise tax) and sales tax if applicable.
9. Apply for all required local and state licenses and permits.
10. Open a business bank account.
11. Set up bookkeeping and accounting systems.
12. Purchase business insurance appropriate for your operations.
13. Set up payroll (if you will have employees) and related compliance processes.
14. Maintain entity compliance:
   - Keep the registered agent and office up to date
   - File amendments as needed
   - Keep required records and books
   - Follow entity-governance rules under TBOC.""",
        "sources": [],
    },
]

# -----------------------------
# Build TF–IDF index over question + answer text
# -----------------------------
from typing import List, Tuple

texts = [item["question"] + " " + item["answer"] for item in KB]

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True,
    max_df=0.95,
)

doc_vectors = vectorizer.fit_transform(texts)


def retrieve(query: str, top_k: int = 3) -> List[Tuple[int, float]]:
    """Return indices + scores for the top_k most similar Q&A entries."""
    if not query.strip():
        return []
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, doc_vectors)[0]
    order = np.argsort(-sims)
    hits = [(int(i), float(sims[int(i)])) for i in order[:top_k] if sims[int(i)] > 0]
    return hits


# -----------------------------
# UI
# -----------------------------
st.title("ATLAS (Advisory, Texas, Legal, Assistance, System)")
st.caption(
    "Ask a question. The app finds the closest curated prompt and returns the exact answer from your knowledge base."
)

default_q = "How do I properly set up an LLC in Texas?"
q = st.text_input("Ask a question:", value=default_q)

top_k = st.slider("Show top matches", min_value=1, max_value=5, value=3)

if st.button("Generate answer", type="primary"):
    hits = retrieve(q, top_k=top_k)

    if not hits:
        st.warning("No close matches found in the knowledge base. Try rephrasing your question.")
    else:
        best_idx, best_score = hits[0]
        best = KB[best_idx]

        st.markdown("### Matched prompt")
        st.markdown(f"**{best['question']}**")
        if best.get("category"):
            st.caption(f"Category: {best['category']} · similarity score: {best_score:.3f}")
        else:
            st.caption(f"Similarity score: {best_score:.3f}")

        st.markdown("### Answer")
        st.markdown(best["answer"])

        st.markdown("### Sources")
        if best["sources"]:
            for url in best["sources"]:
                st.markdown(f"- [{url}]({url})")
        else:
            st.write("_No specific source links stored for this prompt._")

        if len(hits) > 1:
            rows = []
            for rank, (idx, score) in enumerate(hits, start=1):
                item = KB[idx]
                rows.append(
                    {
                        "Rank": rank,
                        "Question": item["question"],
                        "Category": item["category"],
                        "Score": round(score, 3),
                    }
                )
            df = pd.DataFrame(rows)
            st.markdown("### Other close matches")
            st.dataframe(df, use_container_width=True)

st.markdown("---")
st.caption("To add more prompts and answers, extend the KB list at the top of `app.py`.")
