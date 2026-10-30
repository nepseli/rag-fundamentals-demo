"""Central configuration for the RAG demo.

Every value below can be overridden with an environment variable, so nothing
needs to be edited in the code to point the demo at a different Ollama server,
different models, or your own documents.

Things you may want to change:
    OLLAMA_BASE_URL  - where your Ollama server is listening
    EMBED_MODEL      - Ollama model used to turn text into vectors
    CHAT_MODEL       - Ollama model used to write answers
    RAG_DOCS_DIR     - folder containing the .txt files to index
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# --- Paths -------------------------------------------------------------------
# Folder with the source documents (every *.txt file inside is ingested).
DOCS_DIR = Path(os.getenv("RAG_DOCS_DIR", BASE_DIR / "data" / "policies"))
# SQLite file holding documents and chunks. Created automatically on first run.
DB_PATH = Path(os.getenv("RAG_DB_PATH", BASE_DIR / "rag_demo.db"))

# --- Ollama ------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
# Note: the code turns off the model's "thinking" output (see conversation.py).
# Use a chat model that supports this. Some builds, such as certain `qwen3:4b`
# versions, are thinking-only and will print their reasoning in the answer.
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3:8b")
# Models offered in the Streamlit sidebar. Only models you have pulled will work.
CHAT_MODEL_OPTIONS = [m.strip() for m in os.getenv("CHAT_MODEL_OPTIONS", CHAT_MODEL).split(",")]

# --- Chunking (stage 2) ------------------------------------------------------
# Sizes are measured in characters.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

# --- Vector store (stage 3) --------------------------------------------------
COLLECTION_NAME = "policy_chunks"

# --- Retrieval (stage 4) -----------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "4"))

# --- Conversation (multi-turn chat) ------------------------------------------
# Number of most recent messages (user + assistant) kept as chat history.
HISTORY_MESSAGES = int(os.getenv("HISTORY_MESSAGES", "6"))

# --- Output ------------------------------------------------------------------
# How many rows the command-line stage printouts show.
PREVIEW_ROWS = 5
