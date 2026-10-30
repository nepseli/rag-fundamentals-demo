"""STAGE 5 - GENERATION

What it does
    Builds the final prompt (instructions + retrieved chunks + recent chat
    history + the user's question) and streams the answer from the Ollama chat
    model. The instructions tell the model to answer only from the supplied
    context and to say so when the answer is not there; this is what keeps the
    response grounded in the documents.

Input   : the question, retrieved chunks (stage 4) and optional chat history
Output  : an answer, streamed token by token

Settings (see config.py): CHAT_MODEL, OLLAMA_BASE_URL

Run on its own:  python stage5_generation.py
"""

from collections.abc import Iterator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

import config
import conversation

SYSTEM_PROMPT = """You are an HR assistant for Lumora Labs. Answer the employee's question using ONLY the context below.

Rules:
- If the context does not contain the answer, reply exactly: "I could not find this in the leave policy."
- Cite the source for each fact in square brackets, for example [01_annual_leave.txt].
- Be concise. Use the exact numbers from the context.

Context:
{context}"""

NO_CONTEXT_PROMPT = "You are a helpful assistant. Answer the question as best you can. Be concise."


def format_context(chunks: list[dict]) -> str:
    """Label each chunk with its source so the model can cite it."""
    return "\n\n".join(
        f"[{c['filename']}, chunk {c['chunk_index']}]\n{c['text']}" for c in chunks
    )


def build_messages(question: str, chunks: list[dict], history: list[dict]) -> list[BaseMessage]:
    """Assemble the chat messages sent to the model.

    Pass an empty `chunks` list to ask the model without any retrieved context
    (useful to show the difference between plain LLM answers and RAG answers).
    """
    system = SYSTEM_PROMPT.format(context=format_context(chunks)) if chunks else NO_CONTEXT_PROMPT
    messages: list[BaseMessage] = [SystemMessage(system)]
    for m in conversation.recent_history(history):
        messages.append(HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"]))
    messages.append(HumanMessage(question))
    return messages


def render_prompt(messages: list[BaseMessage]) -> str:
    """Readable text version of the messages, for display."""
    labels = {"system": "SYSTEM", "human": "USER", "ai": "ASSISTANT"}
    return "\n\n".join(f"--- {labels.get(m.type, m.type.upper())} ---\n{m.content}" for m in messages)


def stream_answer(messages: list[BaseMessage], model: str = config.CHAT_MODEL) -> Iterator[str]:
    """Yield the answer text piece by piece."""
    for part in conversation.get_chat_model(model).stream(messages):
        if part.content:
            yield part.content


if __name__ == "__main__":
    demo_chunks = [
        {
            "filename": "01_annual_leave.txt",
            "chunk_index": 2,
            "text": "- 2 years but less than 5 years of service: 16 days per year",
        }
    ]
    demo_question = "How many annual leave days do I get after 3 years?"
    msgs = build_messages(demo_question, demo_chunks, history=[])
    print("STAGE 5 - GENERATION\n")
    print(render_prompt(msgs))
    print("\n--- MODEL ANSWER ---")
    for token in stream_answer(msgs):
        print(token, end="", flush=True)
    print("\n\nStage 5 complete: answer generated from the retrieved context.")
    print("Next: run main.py to see all stages together, or `streamlit run streamlit_app.py` for the browser interface.")
