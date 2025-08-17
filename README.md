# AI Departmental Assistant Hub (Gemini Edition)

A Streamlit app that provides **Departmental Team Member (DTM)** assistants for **HR**, **Finance**, and **Sales**.  
Each assistant uses **RAG (retrieval‑augmented generation)** on your department documents stored in a **Chroma** vector store and answers using **Google Gemini**.

> LLM: `gemini-2.5-pro` via `langchain-google-genai`  
> Embeddings: choose **Google `text-embedding-004`** *(default)* or **local SentenceTransformers** (`all-MiniLM-L6-v2`).

---

## 🚀 Quickstart

```bash
# 1) Clone or unzip this project
cd ai-dtm-hub

# 2) Create a virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Add your Google Gemini API key
cp .env.example .env
# then edit .env and set GOOGLE_API_KEY=your_key_here

# 5) Run the app
streamlit run app.py
```

Open the URL shown in the terminal (usually http://localhost:8501).

---

## 🔑 Environment

The app reads `GOOGLE_API_KEY` from `.env` (or your shell env).  
You can get a key from Google AI Studio / Gemini API.

---

## 🧠 Features

- **Department selector** (HR, Finance, Sales)
- **File upload** for PDF/TXT/CSV → ingest into **Chroma** with embeddings
- **Per‑department prompt template** (SOPs/policies)
- **RAG chat** with conversation history
- **Metrics dashboard** (query count, avg latency, usage by department)
- **CSV logging** of every interaction at `logs/usage.csv`

---

## ⚙️ Embeddings Backends (CPU/GPU)

You can choose the embeddings backend in the sidebar:

1. **google** *(default)* → `text-embedding-004` via Gemini  
   - Pros: no PyTorch install, lightweight, great quality
   - Cons: uses API credits

2. **local** → `sentence-transformers` (`all-MiniLM-L6-v2`)  
   - Pros: fully local, zero API cost for embeddings  
   - Cons: requires `torch` + downloads the model on first run

### Device Selection (for local backend)

- **auto**: lets SentenceTransformers pick. If CUDA is available, it uses GPU, else CPU.
- **cpu**: force CPU
- **cuda**: force GPU

> Note: the **LLM** (Gemini) runs remotely via API; device selection here only affects **embeddings**.

---

## 🗂️ Data & Stores

- Vector stores persist in `stores/<department>_<backend>/`
- Example: `stores/hr_google/`
- Logs are appended to `logs/usage.csv`

---

## 🧪 Demo Data

Already included:
- `sample_data/hr/leave_policy.txt`
- `sample_data/finance/monthly_finance.csv`
- `sample_data/sales/product_FAQ.txt`

Click **“Load Demo Data”** for the selected department in the sidebar.

---

## 🧩 Code Map

- `app.py` — Streamlit UI + chat & metrics
- `dtms.py` — RAG chain builder (Gemini + retriever + prompt)
- `ingest.py` — file ingest helpers, embeddings backends, Chroma
- `utils/loaders.py` — simple TXT/PDF/CSV loaders + chunking
- `utils/logger.py` — CSV logger
- `requirements.txt` — dependencies
- `stores/` — Chroma persistence
- `logs/usage.csv` — interaction logs (auto‑created)

---

## 🛠️ Troubleshooting

- **No module named `langchain_google_genai`**  
  Run: `pip install -U langchain-google-genai google-generativeai`

- **CUDA not available** (for local embeddings)  
  Use **google** backend or force **cpu** device in the sidebar.

- **Chroma reset**  
  Delete the `stores/` folder to rebuild from scratch.

---
