"""CSV logger for usage metrics."""
from __future__ import annotations

import os, csv, time, threading
from typing import Dict

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
LOG_FILE = os.path.join(LOG_DIR, "usage.csv")
_LOCK = threading.Lock()

HEADER = ["ts", "department", "query", "response_preview", "latency_ms", "n_context_docs", "template_hash"]

def _ensure_file() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(HEADER)

def log_usage(row: Dict[str, str | int | float]) -> None:
    _ensure_file()
    with _LOCK:
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                row.get("ts", int(time.time())),
                row.get("department", "unknown"),
                (row.get("query", "") or "").strip().replace("\n", " ")[:500],
                (row.get("response_preview", "") or "").strip().replace("\n", " ")[:500],
                int(row.get("latency_ms", 0) or 0),
                int(row.get("n_context_docs", 0) or 0),
                row.get("template_hash", "-"),
            ])
