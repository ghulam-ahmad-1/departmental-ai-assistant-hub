from __future__ import annotations

import os, time, hashlib
from dotenv import load_dotenv
import pandas as pd
import streamlit as st

from utils.logger import log_usage, LOG_FILE
from ingest import ingest_streamlit_uploads, ingest_paths, get_persist_dir
from dtms import build_rag_chain, DEFAULT_TEMPLATES

load_dotenv()  # Loads GOOGLE_API_KEY

st.set_page_config(page_title="AI Departmental Assistant Hub", page_icon="🤖", layout="wide")

# -------------- Sidebar Controls --------------
st.sidebar.title("AI DTM Hub ⚙️")
department = st.sidebar.selectbox("Department", ["HR", "Finance", "Sales"], index=0)
embedding_backend = st.sidebar.selectbox("Embeddings backend", ["google", "local"], index=0, help="'google' uses text-embedding-004 via Gemini; 'local' uses SentenceTransformers (can use CPU or GPU)." )

device_choice = st.sidebar.selectbox("Device (local backend)", ["auto", "cpu", "cuda"], index=0, help="Only applies to 'local' backend.")
device = None
if embedding_backend == "local":
    if device_choice == "auto":
        # Let SentenceTransformers decide; or user can override
        device = None
    else:
        device = device_choice

with st.sidebar.expander("Department Prompt Template", expanded=True):
    default = DEFAULT_TEMPLATES.get(department, "You are a helpful assistant.")
    system_template = st.text_area("System prompt prefix", value=default, height=120)
    st.caption("Customize how the assistant speaks for this department.")

uploaded_files = st.sidebar.file_uploader("Upload department files (PDF/TXT/CSV)", type=["pdf", "txt", "csv"], accept_multiple_files=True)
if st.sidebar.button("Ingest Uploaded Files", use_container_width=True) and uploaded_files:
    rep = ingest_streamlit_uploads(uploaded_files, department=department, backend=embedding_backend, device=device)
    st.sidebar.success(f"Ingested {rep.n_docs} chunks into {rep.persist_dir}")

# Demo data loader
demo_paths = [
    os.path.join("sample_data", "hr", "leave_policy.txt"),
    os.path.join("sample_data", "finance", "monthly_finance.csv"),
    os.path.join("sample_data", "sales", "product_FAQ.txt"),
]
if st.sidebar.button("Load Demo Data", use_container_width=True):
    # Load only relevant department's sample
    dept_to_path = {
        "HR": demo_paths[0],
        "Finance": demo_paths[1],
        "Sales": demo_paths[2],
    }
    p = dept_to_path[department]
    rep = ingest_paths([p], department=department, backend=embedding_backend, device=device)
    st.sidebar.success(f"Loaded demo doc for {department} → {rep.persist_dir}")

# -------------- Tabs --------------
tab_chat, tab_metrics = st.tabs(["💬 Chat", "📈 Metrics"])

# Session-state chat history per department
key_hist = f"history_{department}"
if key_hist not in st.session_state:
    st.session_state[key_hist] = []  # list of BaseMessage

with tab_chat:
    st.header(f"{department} Assistant")
    st.caption(f"Embeddings: {embedding_backend} | Store: {get_persist_dir(department, embedding_backend)}")

    chain, retriever, final_template = build_rag_chain(department, backend=embedding_backend, device=device, system_prefix=system_template)

    # Render conversation
    for msg in st.session_state[key_hist]:
        if msg.type == "human":
            st.chat_message("user").markdown(msg.content)
        else:
            st.chat_message("assistant").markdown(msg.content)

    user_q = st.chat_input("Ask a question about your department documents…")
    if user_q:
        st.chat_message("user").markdown(user_q)
        t0 = time.time()
        # Convert history to plain list of dicts for the chain
        history_msgs = st.session_state[key_hist]
        answer = chain.invoke({"question": user_q, "history": history_msgs})
        dt_ms = int((time.time() - t0) * 1000)

        st.chat_message("assistant").markdown(answer)
        # Append to history
        from langchain_core.messages import HumanMessage, AIMessage
        st.session_state[key_hist].append(HumanMessage(content=user_q))
        st.session_state[key_hist].append(AIMessage(content=answer))

        # Log usage
        import hashlib
        template_hash = hashlib.sha256((final_template or "").encode("utf-8")).hexdigest()[:10]
        # Try to count retrieved docs
        ctx_docs = retriever.get_relevant_documents(user_q)
        log_usage({
            "department": department,
            "query": user_q,
            "response_preview": answer[:300],
            "latency_ms": dt_ms,
            "n_context_docs": len(ctx_docs),
            "template_hash": template_hash,
        })

with tab_metrics:
    st.header("Adoption Metrics")
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        total = len(df)
        avg_lat = df["latency_ms"].mean() if total else 0
        by_dept = df.groupby("department").size().reset_index(name="queries")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total queries", int(total))
        c2.metric("Avg latency (ms)", int(avg_lat))
        c3.metric("Departments", df["department"].nunique())
        st.subheader("Queries by department")
        st.bar_chart(by_dept.set_index("department"))
        st.subheader("Recent interactions")
        st.dataframe(df.tail(50), use_container_width=True)
    else:
        st.info("No logs yet. Interact with a DTM to populate metrics.")
