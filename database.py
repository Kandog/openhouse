"""SQLite database for storing visitor face embeddings and metadata."""

import sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime
import config


def _conn() -> sqlite3.Connection:
    db_dir = config.DB_PATH.parent
    db_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS visitors (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            embedding   BLOB NOT NULL,
            first_seen  TEXT NOT NULL,
            last_seen   TEXT NOT NULL,
            visit_count INTEGER NOT NULL DEFAULT 1
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            visitor_id    INTEGER REFERENCES visitors(id),
            event_type    TEXT NOT NULL,
            message       TEXT,
            created_at    TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def add_visitor(name: str, embedding: np.ndarray) -> int:
    conn = _conn()
    now = datetime.utcnow().isoformat()
    cur = conn.execute(
        "INSERT INTO visitors (name, embedding, first_seen, last_seen, visit_count) VALUES (?, ?, ?, ?, 1)",
        (name, embedding.tobytes(), now, now),
    )
    conn.commit()
    vid = cur.lastrowid
    conn.execute(
        "INSERT INTO events (visitor_id, event_type, message, created_at) VALUES (?, ?, ?, ?)",
        (vid, "first_seen", f"New visitor registered as {name}", now),
    )
    conn.commit()
    conn.close()
    return vid


def update_visitor_last_seen(visitor_id: int) -> None:
    conn = _conn()
    now = datetime.utcnow().isoformat()
    conn.execute(
        "UPDATE visitors SET last_seen = ?, visit_count = visit_count + 1 WHERE id = ?",
        (now, visitor_id),
    )
    conn.commit()
    conn.close()


def get_all_visitors() -> list[dict]:
    conn = _conn()
    rows = conn.execute("SELECT id, name, embedding, first_seen, last_seen, visit_count FROM visitors").fetchall()
    conn.close()
    return [
        {
            "id": r["id"],
            "name": r["name"],
            "embedding": np.frombuffer(r["embedding"], dtype=np.float64),
            "first_seen": r["first_seen"],
            "last_seen": r["last_seen"],
            "visit_count": r["visit_count"],
        }
        for r in rows
    ]


def get_visitor_by_id(visitor_id: int) -> dict | None:
    conn = _conn()
    row = conn.execute("SELECT * FROM visitors WHERE id = ?", (visitor_id,)).fetchone()
    conn.close()
    if row is None:
        return None
    return {
        "id": row["id"],
        "name": row["name"],
        "embedding": np.frombuffer(row["embedding"], dtype=np.float64),
        "first_seen": row["first_seen"],
        "last_seen": row["last_seen"],
        "visit_count": row["visit_count"],
    }


def log_event(visitor_id: int | None, event_type: str, message: str | None = None) -> None:
    conn = _conn()
    now = datetime.utcnow().isoformat()
    conn.execute(
        "INSERT INTO events (visitor_id, event_type, message, created_at) VALUES (?, ?, ?, ?)",
        (visitor_id, event_type, message, now),
    )
    conn.commit()
    conn.close()
