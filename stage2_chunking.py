"""STAGE 2 - CHUNKING

What it does
    Splits each document into smaller overlapping pieces ("chunks"). Models
    and vector search work best on focused passages, not whole documents.
    The splitter tries paragraph breaks first, then lines, then sentences,
    then words, so chunks end at natural boundaries where possible.

Input   : rows in SQLite `documents`
Output  : rows in SQLite `chunks` (also returned as a list of dicts)

Settings (see config.py):
    CHUNK_SIZE     - maximum characters per chunk
    CHUNK_OVERLAP  - characters repeated between neighbouring chunks, so a fact
                     that sits on a boundary is not lost

Run on its own:  python stage2_chunking.py
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
import db
import stage1_ingestion


def chunk_documents(
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[dict]:
    """Split every stored document and save the chunks to SQLite."""
    documents = db.list_documents()
    if not documents:
        documents = stage1_ingestion.ingest()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,  # records where each chunk starts in the source text
    )

    rows = []
    for doc in documents:
        pieces = splitter.create_documents([doc["content"]])
        for index, piece in enumerate(pieces):
            start = piece.metadata["start_index"]
            rows.append(
                {
                    "document_id": doc["id"],
                    "chunk_index": index,
                    "text": piece.page_content,
                    "start_char": start,
                    "end_char": start + len(piece.page_content),
                }
            )

    db.replace_chunks(rows)
    return db.list_chunks()


def summarize(chunks: list[dict]) -> dict:
    """Basic statistics about a set of chunks."""
    sizes = [len(c["text"]) for c in chunks]
    return {
        "chunks": len(chunks),
        "avg_chars": round(sum(sizes) / len(sizes), 1) if sizes else 0,
        "min_chars": min(sizes, default=0),
        "max_chars": max(sizes, default=0),
    }


def show(chunks: list[dict], rows: int = config.PREVIEW_ROWS) -> None:
    """Print chunk statistics and the first few chunks."""
    stats = summarize(chunks)
    print(
        f"\nSTAGE 2 - CHUNKING: {stats['chunks']} chunks "
        f"(avg {stats['avg_chars']} chars, min {stats['min_chars']}, max {stats['max_chars']})"
    )
    print(f"Settings: chunk_size={config.CHUNK_SIZE}, chunk_overlap={config.CHUNK_OVERLAP}\n")
    print(f"First {rows} chunks:")
    for c in chunks[:rows]:
        preview = c["text"][:90].replace("\n", " ")
        print(f"  [{c['filename']} #{c['chunk_index']}] {len(c['text'])} chars, "
              f"chars {c['start_char']}-{c['end_char']}: {preview}...")


if __name__ == "__main__":
    all_chunks = chunk_documents()
    show(all_chunks)
    print(f"\nStage 2 complete: {len(all_chunks)} chunks saved to SQLite.")
    print("Next: run stage3_embedding_vectorstore.py to embed the chunks and build the vector store.")
