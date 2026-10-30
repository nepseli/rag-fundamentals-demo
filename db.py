"""SQLite helper used by the pipeline stages.

SQLite is the system of record for the demo:

    documents(id, filename, title, content, sha256, ingested_at)
    chunks(id, document_id, chunk_index, text, start_char, end_char)

The vector store (Chroma) only holds embeddings and can always be rebuilt from
the `chunks` table.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT NOT NULL UNIQUE,
    title       TEXT NOT NULL,
    content     TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    ingested_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    start_char  INTEGER NOT NULL,
    end_char    INTEGER NOT NULL
);
"""


@contextmanager
def connect():
    """Open a connection, create tables if needed, and commit on exit."""
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def upsert_document(filename: str, title: str, content: str, sha256: str) -> None:
    """Insert a document, or update it if the file name already exists."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO documents (filename, title, content, sha256, ingested_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(filename) DO UPDATE SET
                title = excluded.title,
                content = excluded.content,
                sha256 = excluded.sha256,
                ingested_at = excluded.ingested_at
            """,
            (filename, title, content, sha256, now),
        )


def remove_missing_documents(keep_filenames: list[str]) -> None:
    """Delete documents (and their chunks) whose files are no longer on disk."""
    with connect() as conn:
        if keep_filenames:
            marks = ",".join("?" * len(keep_filenames))
            conn.execute(f"DELETE FROM documents WHERE filename NOT IN ({marks})", keep_filenames)
        else:
            conn.execute("DELETE FROM documents")


def list_documents() -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM documents ORDER BY filename").fetchall()
    return [dict(r) for r in rows]


def replace_chunks(chunks: list[dict]) -> None:
    """Delete all existing chunks and store the given ones.

    Each chunk dict needs: document_id, chunk_index, text, start_char, end_char.
    """
    with connect() as conn:
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM sqlite_sequence WHERE name = 'chunks'")  # restart ids at 1
        conn.executemany(
            """
            INSERT INTO chunks (document_id, chunk_index, text, start_char, end_char)
            VALUES (:document_id, :chunk_index, :text, :start_char, :end_char)
            """,
            chunks,
        )


def list_chunks() -> list[dict]:
    """All chunks in reading order, joined with their source file name."""
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT c.id, c.document_id, d.filename, c.chunk_index,
                   c.text, c.start_char, c.end_char
            FROM chunks c JOIN documents d ON d.id = c.document_id
            ORDER BY d.filename, c.chunk_index
            """
        ).fetchall()
    return [dict(r) for r in rows]


def get_chunks_by_ids(ids: list[int]) -> dict[int, dict]:
    """Fetch chunks by primary key. Returns {chunk_id: chunk_dict}."""
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    with connect() as conn:
        rows = conn.execute(
            f"""
            SELECT c.id, d.filename, c.chunk_index, c.text, c.start_char, c.end_char
            FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE c.id IN ({marks})
            """,
            ids,
        ).fetchall()
    return {r["id"]: dict(r) for r in rows}
