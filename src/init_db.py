import sqlite3
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "portfolio.db"
SCHEMA_PATH = DATA_DIR / "schema.sql"


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        n = conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
    print(f"Database ready at {DB_PATH} ({n} assets)")


if __name__ == "__main__":
    init_db()