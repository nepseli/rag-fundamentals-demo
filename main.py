"""MAIN ENTRY POINT (command line)

Runs the whole RAG pipeline end to end and prints the output of every stage:

    Stage 1  ingestion                  -> stage1_ingestion.py
    Stage 2  chunking                   -> stage2_chunking.py
    Stage 3  embedding + vector store   -> stage3_embedding_vectorstore.py
    Stage 4  retrieval                  -> stage4_retrieval.py
    Stage 5  generation                 -> stage5_generation.py
    (multi-turn question rewriting)     -> conversation.py

Usage
    python main.py            # build the index, then run a scripted 3-question conversation
    python main.py --chat     # build the index, then chat interactively (type 'quit' to exit)

Prerequisite: Ollama running with the embedding and chat models pulled (see OLLAMA_SETUP.md).
For the browser interface run:  streamlit run streamlit_app.py
"""

import argparse
import sys

import config
import conversation
import stage1_ingestion
import stage2_chunking
import stage3_embedding_vectorstore
import stage4_retrieval
import stage5_generation

# Scripted conversation: the second question is a follow-up that needs rewriting,
# the third is not covered by the documents.
DEMO_QUESTIONS = [
    "How many days of annual leave do I get after 3 years of service?",
    "And after 5 years?",
    "What is the dental insurance coverage?",
]


def build_index():
    """Run stages 1-3 and return the vector store."""
    stage1_ingestion.show(stage1_ingestion.ingest())
    stage2_chunking.show(stage2_chunking.chunk_documents())
    store = stage3_embedding_vectorstore.build_vector_store()
    stage3_embedding_vectorstore.show(store)
    return store


def ask(store, question: str, history: list[dict]) -> str:
    """Answer one question: rewrite (if follow-up) -> retrieve -> generate. Returns the answer."""
    print("\n" + "=" * 78)
    print(f"USER: {question}")

    standalone = conversation.rewrite_question(question, history)
    if standalone != question:
        print(f"\n[Rewritten for retrieval] {standalone}")

    results = stage4_retrieval.retrieve(store, standalone)
    stage4_retrieval.show(standalone, results)

    messages = stage5_generation.build_messages(question, results, history)
    print("\nSTAGE 5 - GENERATION\nANSWER: ", end="", flush=True)
    answer = "".join(_print_stream(stage5_generation.stream_answer(messages)))
    print()
    return answer


def _print_stream(tokens):
    for token in tokens:
        print(token, end="", flush=True)
        yield token


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG demo pipeline in the terminal.")
    parser.add_argument("--chat", action="store_true", help="interactive chat instead of the scripted demo")
    args = parser.parse_args()

    # Make sure special characters in model output print correctly on every terminal.
    sys.stdout.reconfigure(encoding="utf-8")

    print(f"Embedding model: {config.EMBED_MODEL} | Chat model: {config.CHAT_MODEL} | Ollama: {config.OLLAMA_BASE_URL}")
    store = build_index()

    history: list[dict] = []
    questions = iter(DEMO_QUESTIONS) if not args.chat else None
    while True:
        if questions is not None:
            question = next(questions, None)
        else:
            question = input("\nYou (or 'quit'): ").strip()
            if question.lower() in {"quit", "exit", ""}:
                question = None
        if question is None:
            break
        answer = ask(store, question, history)
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


if __name__ == "__main__":
    main()
