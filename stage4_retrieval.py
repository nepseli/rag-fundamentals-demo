"""STAGE 4 - RETRIEVAL

What it does
    Embeds the user's question with the same model used for the chunks, then
    asks the vector store for the `k` chunks whose vectors are closest to the
    question vector. Each result comes with a similarity score (1.0 means
    identical direction; higher is more relevant). The chunk text is read back
    from SQLite using the chunk id stored in the vector metadata.

Input   : a question (text) and the Chroma vector store from stage 3
Output  : a ranked list of chunks with scores

Settings (see config.py): TOP_K

Run on its own:  python stage4_retrieval.py "How many days of annual leave after 3 years?"
"""

import sys

from langchain_chroma import Chroma

import config
import db
import stage3_embedding_vectorstore


def retrieve(store: Chroma, question: str, k: int = config.TOP_K) -> list[dict]:
    """Return the top-k chunks for `question`, best match first."""
    hits = store.similarity_search_with_score(question, k=k)  # (document, cosine distance)
    chunk_ids = [int(doc.metadata["chunk_id"]) for doc, _ in hits]
    stored = db.get_chunks_by_ids(chunk_ids)

    results = []
    for rank, (doc, distance) in enumerate(hits, start=1):
        chunk = stored[int(doc.metadata["chunk_id"])]
        results.append(
            {
                "rank": rank,
                "chunk_id": chunk["id"],
                "filename": chunk["filename"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "score": round(1.0 - distance, 4),  # cosine similarity
            }
        )
    return results


def show(question: str, results: list[dict]) -> None:
    """Print the retrieved chunks with their scores."""
    print(f"\nSTAGE 4 - RETRIEVAL")
    print(f"Question: {question}")
    print(f"Top {len(results)} chunks:")
    for r in results:
        preview = r["text"][:110].replace("\n", " ")
        print(f"  #{r['rank']} score={r['score']:.3f} [{r['filename']} #{r['chunk_index']}] {preview}...")


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "How many days of annual leave do I get after 3 years of service?"
    vector_store = stage3_embedding_vectorstore.build_vector_store()
    found = retrieve(vector_store, query)
    show(query, found)
    print(f"\nStage 4 complete: {len(found)} chunks retrieved for the question.")
    print("Next: run conversation.py to see how follow-up questions are rewritten (multi-turn),")
    print("      then stage5_generation.py to generate the answer.")
