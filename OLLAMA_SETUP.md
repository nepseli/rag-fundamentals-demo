# Ollama setup

[Ollama](https://ollama.com) runs large language models and embedding models locally. This demo uses it for both:

| Purpose | Model | Approx. size |
|---|---|---|
| Embeddings (turn text into vectors) | `nomic-embed-text` | 0.3 GB |
| Chat (write answers, rewrite follow-up questions) | `qwen3:8b` | 5 GB |

Everything runs on your machine; no account or API key is needed.

## 1. Install Ollama

**Windows**
1. Download the installer from <https://ollama.com/download> and run it.
2. Ollama starts automatically in the background (look for the llama icon in the system tray).

**macOS**
1. Download the app from <https://ollama.com/download>, move it to Applications and open it.
2. Or with Homebrew: `brew install ollama`, then `ollama serve` in a terminal.

**Linux**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
This installs Ollama and starts it as a service. If it is not running, start it with `ollama serve`.

## 2. Check that it is running

```bash
ollama --version
```

Ollama listens on `http://localhost:11434` by default. Opening that address in a browser should show "Ollama is running".

## 3. Download the models

```bash
ollama pull nomic-embed-text
ollama pull qwen3:8b
```

Confirm they are installed:

```bash
ollama list
```

## 4. Try a model (optional)

```bash
ollama run qwen3:8b "Say hello in one sentence."
```

## Using different models or a different server

The demo reads its settings from environment variables (defaults are in [config.py](config.py)):

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Address of the Ollama server |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `CHAT_MODEL` | `qwen3:8b` | Chat model |
| `CHAT_MODEL_OPTIONS` | same as `CHAT_MODEL` | Comma-separated list shown in the Streamlit sidebar |

Example (PowerShell):

```powershell
$env:CHAT_MODEL = "llama3.1:8b"
python main.py
```

Example (macOS / Linux):

```bash
CHAT_MODEL=llama3.1:8b python main.py
```

Notes on choosing a chat model:
- The code asks the model not to print its internal "thinking" text. Models that support this switch (for example `qwen3:8b`) work as intended. Some builds are thinking-only (certain `qwen3:4b` versions, for example) and will include their reasoning in the answer; avoid those or use a different model.
- If you change `EMBED_MODEL`, the vectors change size, so restart the app (the index is rebuilt on start).
- Larger models answer better but need more RAM. As a guide, 8B models want about 8 GB of free memory; if your machine is smaller, try a 3B-4B model that supports disabling thinking.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot reach Ollama at http://localhost:11434` | Start Ollama (open the app, or run `ollama serve`). |
| `model "..." not found` | Run `ollama pull <model>` for the model named in the message. |
| First question is very slow | Ollama loads the model into memory on first use. Later questions are faster. |
| Port 11434 already in use | Another Ollama instance is running. Stop it, or set `OLLAMA_BASE_URL` to the address you use. |
| Out-of-memory errors | Use a smaller chat model. |
