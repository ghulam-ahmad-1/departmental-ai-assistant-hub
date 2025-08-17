"""Utility loaders & text splitters.

Provides lightweight loaders for TXT, PDF, and CSV without relying on
version-sensitive LangChain loaders. Also exposes a chunker to prepare
documents for the vector store.
"""
from __future__ import annotations

import os
from typing import List
from dataclasses import dataclass
from pypdf import PdfReader
import pandas as pd

try:
    # Prefer modern splitters package if available
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover
    # Fallback to legacy import path
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore

from langchain.schema import Document


@dataclass
class LoadResult:
    docs: List[Document]
    n_bytes: int


def load_txt(path: str, *, department: str) -> LoadResult:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    meta = {"source": os.path.basename(path), "department": department, "type": "txt"}
    return LoadResult([Document(page_content=text, metadata=meta)], n_bytes=len(text.encode("utf-8")))


def load_pdf(path: str, *, department: str) -> LoadResult:
    reader = PdfReader(path)
    docs, total = [], 0
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        meta = {
            "source": f"{os.path.basename(path)}#page={i+1}",
            "page": i + 1,
            "department": department,
            "type": "pdf",
        }
        docs.append(Document(page_content=text, metadata=meta))
        total += len(text.encode("utf-8"))
    return LoadResult(docs, n_bytes=total)


def load_csv(path: str, *, department: str, max_rows: int | None = None) -> LoadResult:
    df = pd.read_csv(path)
    if max_rows is not None:
        df = df.head(max_rows)
    rows = []
    for i, row in df.iterrows():
        text = "; ".join([f"{c}: {row[c]}" for c in df.columns])
        meta = {
            "source": f"{os.path.basename(path)}#row={i}",
            "row": int(i),
            "department": department,
            "type": "csv",
        }
        rows.append(Document(page_content=text, metadata=meta))
    nbytes = df.to_csv(index=False).encode("utf-8").__len__()
    return LoadResult(rows, n_bytes=nbytes)


def load_any(path: str, *, department: str) -> LoadResult:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt", ".md"):
        return load_txt(path, department=department)
    if ext in (".pdf",):
        return load_pdf(path, department=department)
    if ext in (".csv",):
        return load_csv(path, department=department)
    raise ValueError(f"Unsupported file type: {ext}")


def chunk_docs(docs: List[Document], *, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)
