"""Ingestion helpers for AI Departmental Assistant Hub.

Functions here are imported by the Streamlit app to index uploaded files
(or demo data) into a persistent Chroma store per department.
"""
from __future__ import annotations

import os, hashlib, tempfile, shutil
from typing import Iterable, List, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
import asyncio
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sentence_transformers import SentenceTransformer

from langchain.schema import Document
from langchain_community.vectorstores import Chroma

# Embeddings backends
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Google
from langchain_community.embeddings import SentenceTransformerEmbeddings  # Local (PyTorch)

from utils.loaders import load_any, chunk_docs

load_dotenv()

CHROMA_PARENT = os.path.join(os.path.dirname(__file__), "stores")


@dataclass
class IngestReport:
    department: str
    n_files: int
    n_docs: int
    n_chunks: int
    persist_dir: str
    backend: str


def _template_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]


# def get_embeddings(backend: str = "google", device: str | None = None):
#     """Return an embeddings object for the chosen backend.

#     - backend="google": uses Gemini embeddings (text-embedding-004)
#     - backend="local":  uses SentenceTransformers (all-MiniLM-L6-v2), device can be 'cpu' or 'cuda'
#     """
#     if backend == "google":
#         return GoogleGenerativeAIEmbeddings(model="text-embedding-004")  # uses GOOGLE_API_KEY
#     elif backend == "local":
#         model_name = "all-MiniLM-L6-v2"
#         kw = {}
#         if device:
#             kw["model_kwargs"] = {"device": device}
#         return SentenceTransformerEmbeddings(model_name=model_name, **kw)
#     else:
#         raise ValueError(f"Unknown embeddings backend: {backend}")

def get_embeddings(backend="google", device="cpu"):
    """
    Returns embedding model: either Google's Gemini embeddings or local SentenceTransformer.
    Ensures asyncio loop exists for Google client.
    """
    if backend == "google":
        try:
            # Ensure there's an event loop (fix for Streamlit thread issue)
            asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return GoogleGenerativeAIEmbeddings(model="text-embedding-004")

    elif backend == "local":
        model = SentenceTransformer("all-MiniLM-L6-v2")
        model = model.to(device)
        return model

    else:
        raise ValueError(f"Unknown embedding backend: {backend}")


def get_persist_dir(department: str, backend: str) -> str:
    safe = department.lower().strip().replace(" ", "_")
    return os.path.join(CHROMA_PARENT, f"{safe}_{backend}")


def ensure_vectorstore(department: str, backend: str, embeddings):
    persist = get_persist_dir(department, backend)
    os.makedirs(persist, exist_ok=True)
    vs = Chroma(collection_name=f"{department}_{backend}", embedding_function=embeddings, persist_directory=persist)
    return vs


def ingest_paths(paths: Iterable[str], *, department: str, backend: str = "google", device: str | None = None) -> IngestReport:
    embeddings = get_embeddings(backend=backend, device=device)
    vs = ensure_vectorstore(department, backend, embeddings)

    all_docs: List[Document] = []
    n_files = 0
    for p in paths:
        res = load_any(p, department=department)
        chunks = chunk_docs(res.docs)
        all_docs.extend(chunks)
        n_files += 1

    if all_docs:
        vs.add_documents(all_docs)
        vs.persist()

    return IngestReport(
        department=department,
        n_files=n_files,
        n_docs=len(all_docs),
        n_chunks=len(all_docs),
        persist_dir=get_persist_dir(department, backend),
        backend=backend,
    )


def ingest_streamlit_uploads(files, *, department: str, backend: str = "google", device: str | None = None) -> IngestReport:
    """Save Streamlit UploadedFile objects to temp paths and reuse ingest_paths."""
    tmpdir = tempfile.mkdtemp(prefix=f"uploads_{department}_")
    saved = []
    try:
        for uf in files:
            path = os.path.join(tmpdir, uf.name)
            with open(path, "wb") as f:
                f.write(uf.getbuffer())
            saved.append(path)
        return ingest_paths(saved, department=department, backend=backend, device=device)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
