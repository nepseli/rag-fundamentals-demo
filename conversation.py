"""CONVERSATION MEMORY (multi-turn chat support)

Why it exists
    A follow-up such as "And after 5 years?" means nothing on its own. If it is
    embedded as-is, retrieval finds unrelated chunks. Before retrieval, the chat
    model rewrites the follow-up into a standalone question using the recent
    conversation, e.g. "How many days of annual leave does an employee with
    5 years of service get?". Retrieval then runs on the rewritten question.

Messages are plain dicts: {"role": "user" | "assistant", "content": "..."}.

Settings (see config.py): HISTORY_MESSAGES, CHAT_MODEL

Run on its own:  python conversation.py
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

import config

REWRITE_INSTRUCTIONS = (
    "You rewrite follow-up questions. Given a chat history and the latest user "
    "question, rewrite the latest question so it can be understood without the "
    "history. Keep the original meaning, include the specific details from the "
    "history that the question depends on, and do NOT answer it. If the question "
    "is already self-contained, return it unchanged. Output only the rewritten question."
)


def recent_history(messages: list[dict], limit: int = config.HISTORY_MESSAGES) -> list[dict]:
    """Keep only the last `limit` messages so the prompt stays small."""
    return messages[-limit:] if limit > 0 else []


def get_chat_model(model: str = config.CHAT_MODEL, temperature: float = 0.0) -> ChatOllama:
    """Chat client. `reasoning=False` turns off the model's hidden thinking step for faster replies."""
    return ChatOllama(
        model=model,
        base_url=config.OLLAMA_BASE_URL,
        temperature=temperature,
        reasoning=False,
    )


def rewrite_question(question: str, history: list[dict], model: str = config.CHAT_MODEL) -> str:
    """Return a standalone version of `question`. With no history, the question is returned as-is."""
    history = recent_history(history)
    if not history:
        return question

    transcript = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in history)
    prompt = f"Chat history:\n{transcript}\n\nLatest question: {question}\n\nStandalone question:"
    reply = get_chat_model(model).invoke([SystemMessage(REWRITE_INSTRUCTIONS), HumanMessage(prompt)])
    rewritten = reply.content.split("</think>")[-1].strip().strip('"')  # drop reasoning text if a model emits it
    return rewritten or question


if __name__ == "__main__":
    demo_history = [
        {"role": "user", "content": "How many days of annual leave do I get after 3 years?"},
        {"role": "assistant", "content": "After 3 years of service you get 16 days of annual leave per year."},
    ]
    follow_up = "And after 5 years?"
    print(f"History   : {len(demo_history)} messages")
    print(f"Follow-up : {follow_up}")
    print(f"Rewritten : {rewrite_question(follow_up, demo_history)}")
    print("\nConversation step complete: the follow-up is now a standalone question that retrieval can use.")
    print("Next: run stage5_generation.py to generate the answer.")
