"""
SQLite persistence layer for chat history.

Schema:
  sessions  (session_id TEXT PRIMARY KEY, created_at TEXT)
  messages  (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT,
             text TEXT, artifact_path TEXT, timestamp TEXT)

All timestamps are ISO-8601 UTC strings (no external dependency needed).

Why SQLite?
  - Zero server setup, single file, ships with Python's stdlib (sqlite3).
  - Survives process restarts: the orchestrator is stateless across launches;
    the DB is the durable store.
  - On startup the frontend (Phase 7) calls get_history() to rehydrate the
    chat UI from previous sessions.

Thread safety:
  sqlite3 connections are not thread-safe by default. We use
  check_same_thread=False and a module-level lock so the gRPC thread pool
  (4 workers) can call us concurrently without corruption. Every write is
  wrapped in a transaction that commits immediately.
"""

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

_DB_PATH = Path(__file__).parent.parent.parent / "chat_history.db"
_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    """Return (lazily creating) the shared SQLite connection."""
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(
            str(_DB_PATH),
            check_same_thread=False,  # guarded by _lock
        )
        _conn.row_factory = sqlite3.Row  # rows behave like dicts
        _conn.execute("PRAGMA foreign_keys = ON")
        _init_schema(_conn)
    return _conn


def _init_schema(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist. Safe to call on every startup."""
    with conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id  TEXT PRIMARY KEY,
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id    TEXT    NOT NULL,
                role          TEXT    NOT NULL,
                text          TEXT    NOT NULL,
                artifact_path TEXT    NOT NULL DEFAULT '',
                timestamp     TEXT    NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages (session_id, id);
        """)


def _now() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


# Public write API

def ensure_session(session_id: str) -> None:
    """
    Insert the session row if it doesn't exist yet.
    Called before the first message write for a session.
    INSERT OR IGNORE means subsequent calls for the same session are no-ops.
    """
    with _lock:
        conn = _get_conn()
        with conn:
            conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id, created_at) VALUES (?, ?)",
                (session_id, _now()),
            )


def save_message(session_id: str, role: str, text: str,
                 artifact_path: str = "") -> None:
    """
    Persist one chat message.

    Args:
      session_id    - identifies the conversation
      role          - "user" or "assistant"
      text          - message content
      artifact_path - path to generated file, or "" if none
    """
    ensure_session(session_id)
    with _lock:
        conn = _get_conn()
        with conn:
            conn.execute(
                """INSERT INTO messages
                   (session_id, role, text, artifact_path, timestamp)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, role, text, artifact_path, _now()),
            )


# Public read API

def get_history(session_id: str) -> list[dict]:
    """
    Return all messages for a session in chronological order.

    Each dict has: id, session_id, role, text, artifact_path, timestamp.
    Returns an empty list if the session doesn't exist.
    """
    with _lock:
        conn = _get_conn()
        rows = conn.execute(
            """SELECT id, session_id, role, text, artifact_path, timestamp
               FROM messages
               WHERE session_id = ?
               ORDER BY id ASC""",
            (session_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def list_sessions() -> list[dict]:
    """
    Return all sessions ordered by creation time, most recent first.
    Each dict has: session_id, created_at.
    Used by the frontend on startup to populate the session list.
    """
    with _lock:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT session_id, created_at FROM sessions ORDER BY created_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def close() -> None:
    """Close the database connection. Call on clean shutdown."""
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
            _conn = None
