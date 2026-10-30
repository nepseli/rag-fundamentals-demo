"""STAGE 1 - INGESTION

What it does
    Reads every .txt file from the documents folder and stores the full text in
    the SQLite `documents` table, together with a content hash so changed files
    can be detected.

Input   : .txt files in config.DOCS_DIR
Output  : rows in SQLite `documents` (also returned as a list of dicts)

Run on its own:  python stage1_ingestion.py
"""

import hashlib
from pathlib import Path

import config
import db


def _title_from_filename(path: Path) -> str:
    """'01_annual_leave.txt' -> 'Annual Leave'."""
    words = path.stem.split("_")
    if words and words[0].isdigit():
        words = words[1:]
    return " ".join(words).title()


def ingest(docs_dir: Path = config.DOCS_DIR) -> list[dict]:
    """Load all .txt files from `docs_dir` into SQLite and return the stored rows."""
    files = sorted(Path(docs_dir).glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {docs_dir}")

    for path in files:
        text = path.read_text(encoding="utf-8")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        db.upsert_document(path.name, _title_from_filename(path), text, digest)

    # Keep the database in sync if a file was deleted from the folder.
    db.remove_missing_documents([p.name for p in files])
    return db.list_documents()


def show(documents: list[dict]) -> None:
    """Print a small summary table of the ingested documents."""
    print(f"\nSTAGE 1 - INGESTION: {len(documents)} documents stored in {config.DB_PATH.name}\n")
    print(f"{'id':<3} {'file':<36} {'chars':>6}  sha256")
    for d in documents:
        print(f"{d['id']:<3} {d['filename']:<36} {len(d['content']):>6}  {d['sha256'][:12]}...")
    first = documents[0]
    print(f"\nFirst 300 characters of {first['filename']}:\n")
    print(first["content"][:300])


if __name__ == "__main__":
    docs = ingest()
    show(docs)
    print(f"\nStage 1 complete: {len(docs)} documents ingested into SQLite.")
    print("Next: run stage2_chunking.py to split them into chunks.")
