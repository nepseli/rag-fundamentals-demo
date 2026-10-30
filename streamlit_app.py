"""STREAMLIT INTERFACE

Browser UI for the RAG demo. One tab per pipeline stage, in order, plus a
multi-turn chat. The tabs call the same functions as the command-line stage
files, so what you see here matches what `python stageN_*.py` prints.

Run:  streamlit run streamlit_app.py

Prerequisite: Ollama running with the embedding and chat models pulled
(see OLLAMA_SETUP.md).
"""

import ollama
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA

import config
import conversation
import db
import stage1_ingestion
import stage2_chunking
import stage3_embedding_vectorstore
import stage4_retrieval
import stage5_generation

st.set_page_config(page_title="RAG Fundamentals Demo", layout="wide")


# --- Helpers -------------------------------------------------------------------
def check_ollama() -> None:
    """Stop with a helpful message if Ollama is unreachable or a model is missing."""
    try:
        installed = [m.model for m in ollama.Client(host=config.OLLAMA_BASE_URL).list().models]
    except Exception:
        st.error(
            f"Cannot reach Ollama at {config.OLLAMA_BASE_URL}. "
            "Start it (see OLLAMA_SETUP.md) and reload this page."
        )
        st.stop()
    missing = [
        name for name in {config.EMBED_MODEL, *config.CHAT_MODEL_OPTIONS}
        if not any(m == name or m.startswith(f"{name}:") for m in installed)
    ]
    if missing:
        st.error("Missing Ollama models: " + ", ".join(missing) + ". Install with `ollama pull <model>`.")
        st.stop()


def build_index(chunk_size: int, chunk_overlap: int) -> None:
    """Run stages 1-3 and keep the results in the session."""
    with st.spinner("Ingesting, chunking and embedding documents..."):
        st.session_state.documents = stage1_ingestion.ingest()
        st.session_state.chunks = stage2_chunking.chunk_documents(chunk_size, chunk_overlap)
        st.session_state.store = stage3_embedding_vectorstore.build_vector_store()
        st.session_state.vectors = stage3_embedding_vectorstore.get_vectors(st.session_state.store)
        st.session_state.built_with = (chunk_size, chunk_overlap)
        st.session_state.last_query = None


def preview(text: str, n: int = 90) -> str:
    return text[:n].replace("\n", " ") + ("..." if len(text) > n else "")


def results_table(results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"rank": r["rank"], "score": r["score"], "file": r["filename"],
             "chunk": r["chunk_index"], "text": preview(r["text"], 120)}
            for r in results
        ]
    )


def show_chunks(results: list[dict]) -> None:
    for r in results:
        with st.expander(f"#{r['rank']}  score {r['score']:.3f}  -  {r['filename']} (chunk {r['chunk_index']})"):
            st.text(r["text"])


# --- Startup -------------------------------------------------------------------
check_ollama()
st.session_state.setdefault("messages", [])
if "store" not in st.session_state:
    build_index(config.CHUNK_SIZE, config.CHUNK_OVERLAP)

# --- Sidebar -------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    chat_model = st.selectbox("Chat model", config.CHAT_MODEL_OPTIONS)
    use_rag = st.toggle("Use RAG (retrieve from documents)", value=True,
                        help="Off: the model answers from its own knowledge only.")
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 10, config.TOP_K)

    st.subheader("Chunking")
    chunk_size = st.slider("Chunk size (characters)", 100, 1500, config.CHUNK_SIZE, step=50)
    chunk_overlap = st.slider("Chunk overlap (characters)", 0, 300, config.CHUNK_OVERLAP, step=10)
    pending = (chunk_size, chunk_overlap) != st.session_state.built_with
    if pending:
        st.warning("Chunk settings changed. Rebuild the index to apply them.")
    if st.button("Rebuild index", type="primary" if pending else "secondary"):
        if chunk_overlap >= chunk_size:
            st.error("Overlap must be smaller than chunk size.")
        else:
            build_index(chunk_size, chunk_overlap)
            st.rerun()

    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.last_query = None
        st.rerun()
    st.caption(f"Embedding model: {config.EMBED_MODEL}")

st.title("RAG Fundamentals Demo")
st.caption("Fictitious company leave policy, answered by local Ollama models. "
           "Stages: Ingestion, Chunking, Embedding, Retrieval, Generation.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["1. Ingestion", "2. Chunking", "3. Embeddings & vector store", "4. Retrieval", "5. Chat (generation)"]
)

# --- Stage 1: ingestion ----------------------------------------------------------
with tab1:
    st.caption("Source files are read and stored as full text in SQLite (table: documents).")
    documents = st.session_state.documents
    st.dataframe(
        pd.DataFrame(
            [{"id": d["id"], "file": d["filename"], "title": d["title"], "characters": len(d["content"]),
              "sha256": d["sha256"][:12], "ingested_at": d["ingested_at"]} for d in documents]
        ),
        hide_index=True, width="stretch",
    )
    chosen = st.selectbox("View document text", [d["filename"] for d in documents])
    st.text_area("Raw text", next(d["content"] for d in documents if d["filename"] == chosen),
                 height=300, disabled=True)

# --- Stage 2: chunking -----------------------------------------------------------
with tab2:
    st.caption("Each document is split into overlapping chunks (table: chunks). "
               "Change the chunk settings in the sidebar and rebuild to see the effect.")
    chunks = st.session_state.chunks
    stats = stage2_chunking.summarize(chunks)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Chunks", stats["chunks"])
    c2.metric("Average size (chars)", stats["avg_chars"])
    c3.metric("Smallest / largest", f"{stats['min_chars']} / {stats['max_chars']}")
    c4.metric("Built with size / overlap", f"{st.session_state.built_with[0]} / {st.session_state.built_with[1]}")

    file_filter = st.selectbox("File", ["All files"] + [d["filename"] for d in st.session_state.documents], key="chunk_file")
    shown = [c for c in chunks if file_filter in ("All files", c["filename"])]
    st.dataframe(
        pd.DataFrame(
            [{"id": c["id"], "file": c["filename"], "chunk": c["chunk_index"], "chars": len(c["text"]),
              "start": c["start_char"], "end": c["end_char"], "preview": preview(c["text"])} for c in shown]
        ),
        hide_index=True, width="stretch", height=300,
    )

    st.subheader("Inspect a chunk and its overlap")
    pick = st.selectbox("Chunk", [f"{c['filename']} #{c['chunk_index']}" for c in shown], key="chunk_pick")
    idx = [f"{c['filename']} #{c['chunk_index']}" for c in shown].index(pick)
    current = shown[idx]
    st.text_area("Chunk text", current["text"], height=160, disabled=True)
    if idx > 0 and shown[idx - 1]["filename"] == current["filename"]:
        overlap_chars = shown[idx - 1]["end_char"] - current["start_char"]
        if overlap_chars > 0:
            st.caption(f"Overlap with the previous chunk: {overlap_chars} characters")
            st.code(current["text"][:overlap_chars], language=None)
        else:
            st.caption("No text is shared with the previous chunk.")

# --- Stage 3: embeddings and vector store ---------------------------------------------
with tab3:
    st.caption("Each chunk is converted to a vector by the embedding model and stored in an in-memory Chroma collection. "
               "Similar meaning means nearby vectors.")
    data = st.session_state.vectors
    dim = len(data["embeddings"][0])
    c1, c2, c3 = st.columns(3)
    c1.metric("Embedding model", config.EMBED_MODEL)
    c2.metric("Vectors in store", len(data["ids"]))
    c3.metric("Dimensions per vector", dim)

    ids = list(data["ids"])
    labels = [f"chunk {i}: {m['filename']} #{m['chunk_index']}" for i, m in zip(ids, data["metadatas"])]
    pick_vec = st.selectbox("Show the vector for", labels)
    vec = data["embeddings"][labels.index(pick_vec)]
    st.write("First 12 values: " + ", ".join(f"{v:+.3f}" for v in vec[:12]) + ", ...")
    st.bar_chart(pd.DataFrame({"value": list(vec[:64])}), height=160)
    st.caption("Bar chart: first 64 of the vector's numbers.")

    st.subheader("All chunks in 2D")
    st.caption("A projection (PCA) of the vectors down to two dimensions. Ask a question in the Chat or Retrieval tab "
               "to see where it lands (red star).")
    pca = PCA(n_components=2).fit(data["embeddings"])
    coords = pca.transform(data["embeddings"])
    plot_df = pd.DataFrame(
        {"x": coords[:, 0], "y": coords[:, 1],
         "file": [m["filename"] for m in data["metadatas"]],
         "chunk": [m["chunk_index"] for m in data["metadatas"]],
         "text": [preview(t, 80) for t in
                  (db.get_chunks_by_ids([int(i) for i in ids])[int(i)]["text"] for i in ids)]}
    )
    fig = px.scatter(plot_df, x="x", y="y", color="file", hover_data=["chunk", "text"])
    last_query = st.session_state.get("last_query")
    if last_query:
        qx, qy = pca.transform([stage3_embedding_vectorstore.get_embeddings().embed_query(last_query)])[0]
        fig.add_scatter(x=[qx], y=[qy], mode="markers+text", text=["question"], textposition="top center",
                        marker={"symbol": "star", "size": 16, "color": "red"}, name="question")
    st.plotly_chart(fig, width="stretch")

# --- Stage 4: retrieval --------------------------------------------------------------
with tab4:
    st.caption("The question is embedded with the same model, and the closest chunks are returned. "
               "No chat model is involved in this step.")
    query = st.text_input("Question", "How much annual leave can I carry over?")
    if st.button("Search"):
        st.session_state.last_query = query
        st.session_state.search_results = stage4_retrieval.retrieve(st.session_state.store, query, top_k)
    if st.session_state.get("search_results"):
        st.dataframe(results_table(st.session_state.search_results), hide_index=True, width="stretch")
        show_chunks(st.session_state.search_results)

# --- Stage 5: chat (multi-turn) ----------------------------------------------------------
with tab5:
    st.caption("Multi-turn chat. Follow-up questions are rewritten into standalone questions, used to retrieve chunks, "
               "and the chunks plus recent history are sent to the chat model.")

    def render_details(msg: dict) -> None:
        with st.expander("How this answer was produced"):
            if not msg["rag"]:
                st.write("RAG was off: no documents were retrieved.")
            else:
                if msg["rewritten"] != msg["question"]:
                    st.markdown(f"**Question rewritten for retrieval:** {msg['rewritten']}")
                else:
                    st.markdown("**Question used for retrieval:** unchanged")
                st.markdown("**Retrieved chunks**")
                st.dataframe(results_table(msg["chunks"]), hide_index=True, width="stretch")
            st.markdown("**Prompt sent to the model**")
            st.code(msg["prompt"], language=None)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_details(msg)

    if question := st.chat_input("Ask about the leave policy"):
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            rewritten, chunks_found = question, []
            if use_rag:
                with st.spinner("Rewriting question and retrieving chunks..."):
                    rewritten = conversation.rewrite_question(question, history, chat_model)
                    chunks_found = stage4_retrieval.retrieve(st.session_state.store, rewritten, top_k)
                st.session_state.last_query = rewritten
            messages = stage5_generation.build_messages(question, chunks_found, history)
            answer = st.write_stream(stage5_generation.stream_answer(messages, chat_model))
            record = {"role": "assistant", "content": answer, "question": question, "rewritten": rewritten,
                      "chunks": chunks_found, "prompt": stage5_generation.render_prompt(messages), "rag": use_rag}
            render_details(record)
        st.session_state.messages += [{"role": "user", "content": question}, record]
