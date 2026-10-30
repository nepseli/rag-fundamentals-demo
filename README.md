# RAG Fundamentals Demo

A small, fully local Retrieval-Augmented Generation (RAG) pipeline for learning the basics. It answers questions about a fictitious company's leave policy, and lets you see the output of every stage along the way.

- **Models:** [Ollama](https://ollama.com) for both embeddings and chat
- **Orchestration:** LangChain
- **Document storage:** SQLite
- **Vector store:** Chroma, in memory
- **Interface:** command line and Streamlit

Nothing leaves your machine, and no API keys are needed.

## The five stages

```
 .txt files
     |
 [1 Ingestion]   read files, store full text in SQLite                      stage1_ingestion.py
     |
 [2 Chunking]    split text into overlapping chunks, store in SQLite        stage2_chunking.py
     |
 [3 Embedding    turn each chunk into a vector, load into Chroma            stage3_embedding_vectorstore.py
    + vector store]
     |
 question --> (multi-turn: rewrite follow-up as a standalone question)      conversation.py
     |
 [4 Retrieval]   embed the question, find the closest chunks                stage4_retrieval.py
     |
 [5 Generation]  prompt = instructions + chunks + history + question        stage5_generation.py
     |           -> chat model streams a cited answer
   answer
```

Chroma is in memory, so the vectors are rebuilt from SQLite every time the program starts. SQLite is the source of truth.

## Prerequisites

- Python 3.10 or newer (developed and tested on 3.13)
- Ollama, with the two models pulled. Step-by-step instructions are in **[OLLAMA_SETUP.md](OLLAMA_SETUP.md)**.

## Setup

```bash
git clone https://github.com/nepseli/rag-fundamentals-demo.git
cd rag-fundamentals-demo

python -m venv .venv
```

Activate the virtual environment:

```powershell
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Make sure Ollama is running and the models are installed (see [OLLAMA_SETUP.md](OLLAMA_SETUP.md)):

```bash
ollama pull nomic-embed-text
ollama pull qwen3:8b
```

## Run it

### Option A: Streamlit interface

```bash
streamlit run streamlit_app.py
```

One tab per stage, then a multi-turn chat. The sidebar controls the chat model, chunk size and overlap, number of chunks retrieved, and a switch to turn RAG off (so you can compare grounded answers with the model's own knowledge).

### Option B: command line, whole pipeline

```bash
python main.py
```

Builds the index, printing the output of stages 1 to 3, then runs a short scripted conversation that prints retrieval and generation for each question. For an interactive terminal chat:

```bash
python main.py --chat
```

### Option C: one stage at a time

Each stage file runs on its own and prints a small sample of its output. This is the best way to walk through the code in an editor.

```bash
python stage1_ingestion.py
python stage2_chunking.py
python stage3_embedding_vectorstore.py
python stage4_retrieval.py "How much annual leave can I carry over?"
python conversation.py
python stage5_generation.py
```

## Project structure

| File | Role |
|---|---|
| `main.py` | Command-line entry point; runs all stages |
| `streamlit_app.py` | Streamlit entry point |
| `stage1_ingestion.py` | Stage 1: load documents into SQLite |
| `stage2_chunking.py` | Stage 2: split documents into chunks |
| `stage3_embedding_vectorstore.py` | Stage 3: embed chunks, build the Chroma collection |
| `stage4_retrieval.py` | Stage 4: similarity search with scores |
| `stage5_generation.py` | Stage 5: build the prompt, stream the answer |
| `conversation.py` | Chat history and follow-up question rewriting |
| `db.py` | SQLite helper (schema and queries) |
| `config.py` | All settings in one place |
| `data/policies/` | Sample documents (fictitious leave policy) |
| `OLLAMA_SETUP.md` | Installing Ollama and the models |

## Configuration

Defaults live in [config.py](config.py). Each can be overridden with an environment variable, so no code edits are needed:

| Variable | Default | What it does |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `CHAT_MODEL` | `qwen3:8b` | Chat model |
| `CHAT_MODEL_OPTIONS` | same as `CHAT_MODEL` | Models listed in the Streamlit dropdown (comma-separated) |
| `RAG_DOCS_DIR` | `data/policies` | Folder of `.txt` files to index |
| `RAG_DB_PATH` | `rag_demo.db` | SQLite file location |
| `CHUNK_SIZE` | `500` | Maximum characters per chunk |
| `CHUNK_OVERLAP` | `80` | Characters shared between neighbouring chunks |
| `TOP_K` | `4` | Chunks retrieved per question |
| `HISTORY_MESSAGES` | `6` | Recent messages kept as chat history |

PowerShell example:

```powershell
$env:CHUNK_SIZE = "300"; python stage2_chunking.py
```

macOS / Linux example:

```bash
CHUNK_SIZE=300 python stage2_chunking.py
```

## Using your own documents

1. Put `.txt` files in a folder (or replace the files in `data/policies/`).
2. Point the demo at it: set `RAG_DOCS_DIR`, or edit `DOCS_DIR` in `config.py`.
3. The prompt in `stage5_generation.py` (`SYSTEM_PROMPT`) mentions the example company and the "leave policy". Edit that text to describe your content.
4. Run `python main.py`, or click **Rebuild index** in Streamlit.

Other file types (PDF, Word) are not loaded by this demo. To add one, convert the files to text or extend `stage1_ingestion.py` with a LangChain document loader.

## Suggested demo script

| Step | What to do | What to point out |
|---|---|---|
| 1 | Run `stage1_ingestion.py` | Five documents, stored as plain rows in SQLite |
| 2 | Run `stage2_chunking.py`, then change `CHUNK_SIZE` | How chunk size changes chunk count. In the sample data the annual-leave entitlement list is split across two chunks, which shows why overlap matters. |
| 3 | Run `stage3_embedding_vectorstore.py` | A chunk becomes 768 numbers; the same dimension for every chunk |
| 4 | Run `stage4_retrieval.py "How long is paternity leave?"` | Scores, and the right chunk ranking first |
| 5 | Run `stage5_generation.py` | The exact prompt the model sees |
| 6 | Streamlit chat: the questions below | Rewritten follow-up, retrieved chunks, prompt |
| 7 | Turn **Use RAG** off and repeat a question | The model guesses without the documents |

Sample questions:

- "How many days of annual leave do I get after 3 years of service?" then "And after 5 years?" (follow-up needs rewriting)
- "How much annual leave can I carry over, and when does it expire?"
- "How long is paternity leave?" then "Can it be split into blocks?"
- "Do I need a medical certificate for 2 days of sick leave?"
- "What is the dental insurance coverage?" (not in the documents; the answer should say it could not be found)

## Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot reach Ollama` | Start Ollama; see [OLLAMA_SETUP.md](OLLAMA_SETUP.md) |
| `model ... not found` | `ollama pull <model>` |
| Answers start with long reasoning text | The chat model is thinking-only; choose another `CHAT_MODEL` (see [OLLAMA_SETUP.md](OLLAMA_SETUP.md)) |
| First request is slow | Ollama is loading the model into memory; later requests are faster |
| Streamlit port already in use | `streamlit run streamlit_app.py --server.port 8600` |
| Changed documents are not picked up | Click **Rebuild index** in Streamlit, or re-run the stage scripts |

## Notes

- The sample company and policy text are fictitious.
- `rag_demo.db` is created on first run and is excluded from version control.
- The index is not saved between runs by design. To persist it, create the Chroma collection with a `persist_directory` in `stage3_embedding_vectorstore.py`.
