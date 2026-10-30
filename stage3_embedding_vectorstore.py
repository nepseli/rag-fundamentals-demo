"""STAGE 3 - EMBEDDING AND VECTOR STORE

What it does
    Turns every chunk into an embedding (a list of numbers that captures its
    meaning) using an Ollama embedding model, then loads the vectors into an
    in-memory Chroma collection. Chunks with similar meaning end up close
    together in vector space, which is what makes semantic search possible.

    The vector store only keeps embeddings plus a little metadata (chunk id,
    file name, chunk number). The chunk text itself stays in SQLite. Because
    Chroma runs in memory here, the index is rebuilt from SQLite every time the
    program starts.

Input   : rows in SQLite `chunks`
Output  : a populated Chroma collection (returned as a LangChain vector store)

Settings (see config.py): EMBED_MODEL, OLLAMA_BASE_URL, COLLECTION_NAME

Run on its own:  python stage3_embedding_vectorstore.py
"""

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

import config
import db
import stage2_chunking


def get_embeddings() -> OllamaEmbeddings:
    """Embedding client. Requires the Ollama server to be running."""
    return OllamaEmbeddings(model=config.EMBED_MODEL, base_url=config.OLLAMA_BASE_URL)


def _new_collection(embeddings: OllamaEmbeddings) -> Chroma:
    # Cosine distance is the usual choice for text embeddings.
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )


def build_vector_store() -> Chroma:
    """Embed all chunks from SQLite and load them into a fresh in-memory collection."""
    chunks = db.list_chunks() or stage2_chunking.chunk_documents()
    embeddings = get_embeddings()

    # Drop any collection left over from an earlier build in this process.
    _new_collection(embeddings).delete_collection()
    store = _new_collection(embeddings)

    store.add_texts(
        texts=[c["text"] for c in chunks],
        metadatas=[
            {"chunk_id": c["id"], "filename": c["filename"], "chunk_index": c["chunk_index"]}
            for c in chunks
        ],
        ids=[str(c["id"]) for c in chunks],  # same id as the SQLite row
    )
    return store


def get_vectors(store: Chroma) -> dict:
    """Return {'ids': [...], 'embeddings': [[...], ...], 'metadatas': [...]} for the whole collection."""
    return store.get(include=["embeddings", "metadatas"])


def show(store: Chroma, rows: int = 3) -> None:
    """Print the vector store size, the embedding dimension and a few vectors."""
    data = get_vectors(store)
    vectors = data["embeddings"]
    print(f"\nSTAGE 3 - EMBEDDING AND VECTOR STORE")
    print(f"Embedding model : {config.EMBED_MODEL}")
    print(f"Vectors stored  : {len(vectors)} (in-memory Chroma collection '{config.COLLECTION_NAME}')")
    print(f"Dimensions      : {len(vectors[0])} numbers per chunk\n")
    print(f"First {rows} vectors (first 6 of {len(vectors[0])} values):")
    for chunk_id, meta, vec in list(zip(data["ids"], data["metadatas"], vectors))[:rows]:
        values = ", ".join(f"{v:+.3f}" for v in vec[:6])
        print(f"  chunk {chunk_id} [{meta['filename']} #{meta['chunk_index']}]: [{values}, ...]")


if __name__ == "__main__":
    vector_store = build_vector_store()
    show(vector_store)
    print("\nStage 3 complete: chunks embedded and loaded into the in-memory vector store.")
    print("Next: run stage4_retrieval.py to search the vector store with a question.")
