"""DTM (Departmental Team Member) chain builders."""
from __future__ import annotations

import os, time
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from ingest import get_embeddings, ensure_vectorstore, get_persist_dir

load_dotenv()

DEFAULT_TEMPLATES = {
    "HR": "You are the HR Assistant. Answer based on company policies, benefits, and HR procedures.",
    "Finance": "You are the Finance Assistant. Answer using uploaded financial reports and explain calculations step-by-step when needed.",
    "Sales": "You are the Sales Assistant. Answer using product sheets, pricing, and FAQs with concise, client-friendly wording.",
}


def get_llm(model: str = "gemini-2.5-pro", temperature: float = 0.2):
    """Create a Gemini chat model via langchain-google-genai."""
    return ChatGoogleGenerativeAI(model=model, temperature=temperature)


def format_docs(docs):
    return "\n\n".join([f"[Source: {d.metadata.get('source','?')}]\n{d.page_content}" for d in docs])


def build_rag_chain(department: str, *, backend: str = "google", device: str | None = None, system_prefix: str | None = None):
    """Create a retrieval-augmented chat chain for a department."""
    embeddings = get_embeddings(backend=backend, device=device)
    vs = ensure_vectorstore(department, backend, embeddings)
    retriever = vs.as_retriever(search_kwargs={"k": 4})

    template = system_prefix or DEFAULT_TEMPLATES.get(department, "You are a helpful departmental assistant.")
    prompt = ChatPromptTemplate.from_messages([
        ("system", template + "\nUse the following context if relevant to answer. If unsure, say you are unsure.\n\n{context}"),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])

    llm = get_llm()
    chain = (
        RunnableParallel(
            context=RunnableLambda(lambda x: retriever.get_relevant_documents(x["question"])) | RunnableLambda(format_docs),
            history=RunnableLambda(lambda x: x.get("history", [])),
            question=RunnableLambda(lambda x: x["question"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever, template
