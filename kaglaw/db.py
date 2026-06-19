from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from typing import Iterator

from .config import DB_PATH

_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    nick TEXT PRIMARY KEY,
    username TEXT NOT NULL,
    config_dir TEXT NOT NULL,
    added_at TEXT DEFAULT (datetime('now')),
    notes TEXT
);

CREATE TABLE IF NOT EXISTS notebooks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    local_path TEXT NOT NULL,
    language TEXT DEFAULT 'python',
    kernel_type TEXT DEFAULT 'notebook',
    enable_gpu INTEGER DEFAULT 0,
    enable_tpu INTEGER DEFAULT 0,
    enable_internet INTEGER DEFAULT 1,
    dataset_sources TEXT,
    competition_sources TEXT,
    kernel_sources TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    notebook_id INTEGER NOT NULL,
    account_nick TEXT NOT NULL,
    slug TEXT NOT NULL,
    version_number INTEGER,
    version_notes TEXT,
    status TEXT,
    pushed_at TEXT DEFAULT (datetime('now')),
    completed_at TEXT,
    runtime_seconds REAL,
    used_gpu INTEGER DEFAULT 0,
    used_tpu INTEGER DEFAULT 0,
    output_path TEXT,
    log_summary TEXT,
    error TEXT,
    FOREIGN KEY(notebook_id) REFERENCES notebooks(id),
    FOREIGN KEY(account_nick) REFERENCES accounts(nick)
);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_account ON runs(account_nick);

CREATE TABLE IF NOT EXISTS submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    competition TEXT NOT NULL,
    account_nick TEXT NOT NULL,
    file_name TEXT,
    description TEXT,
    submitted_at TEXT,
    public_score TEXT,
    private_score TEXT,
    status TEXT,
    rank_public INTEGER,
    rank_private INTEGER,
    leaderboard_size INTEGER,
    last_synced TEXT DEFAULT (datetime('now')),
    UNIQUE(competition, account_nick, submitted_at, file_name)
);
CREATE INDEX IF NOT EXISTS idx_submissions_comp ON submissions(competition);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    provider TEXT,
    model TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    role TEXT NOT NULL,          -- user / assistant / tool / system
    text TEXT,                    -- text payload (assistant final answer, user prompt)
    tool_calls TEXT,              -- JSON: [{id, name, args}] for assistant
    tool_call_id TEXT,            -- for role=tool
    tool_name TEXT,               -- for role=tool
    is_error INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY(conversation_id) REFERENCES conversations(id)
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);

-- Job queue for batch sweeps (Phase 2). One row = one experiment to run.
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT NOT NULL,
    notebook_id INTEGER NOT NULL,
    competition TEXT,
    params TEXT,                 -- JSON {param: value} (clean, for the matrix)
    replacements TEXT,           -- JSON {old: new} actually substituted in source
    allowed_nicks TEXT,          -- JSON list; dispatcher picks one with free quota
    slug_suffix TEXT,
    version_notes TEXT,
    tags TEXT,
    need_gpu INTEGER DEFAULT 0,
    priority INTEGER DEFAULT 0,
    status TEXT DEFAULT 'queued',-- queued | running | done | failed | canceled
    run_id INTEGER,              -- set once dispatched/pushed
    nick TEXT,                   -- chosen nick
    attempts INTEGER DEFAULT 0,
    max_attempts INTEGER DEFAULT 2,
    error TEXT,
    enqueued_at TEXT DEFAULT (datetime('now')),
    dispatched_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_batch ON jobs(batch_id);

-- In-app notifications (e.g. "batch X finished").
CREATE TABLE IF NOT EXISTS notifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT,
    ref TEXT,                    -- e.g. batch_id (used to dedupe)
    title TEXT,
    body TEXT,
    seen INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Long-term memory: facts/preferences the agent recalls across ALL chats.
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT DEFAULT 'fact',    -- preference | fact | note
    text TEXT NOT NULL,
    tags TEXT,
    pinned INTEGER DEFAULT 1,    -- 1 = injected into the system prompt every turn
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);
"""


# Columns added to `conversations` after the fact (rolling summary for compaction).
_CONV_MIGRATIONS: dict[str, str] = {
    "summary": "TEXT",
    "summary_upto": "INTEGER DEFAULT 0",  # max message id already folded into summary
}


# Columns added after the original schema. (no alembic — manual, idempotent.)
# name -> SQL type/decl used in ALTER TABLE ADD COLUMN.
_RUNS_MIGRATIONS: dict[str, str] = {
    "competition": "TEXT",
    "params": "TEXT",                 # JSON: {param: value} for this experiment
    "metric_name": "TEXT",            # e.g. 'cv', 'auc' parsed from the log
    "metric_value": "REAL",
    "lb_public": "TEXT",              # public LB score if this run was submitted
    "lb_private": "TEXT",
    "batch_id": "TEXT",               # groups a sweep / multi-nick push together
    "tags": "TEXT",                   # free-text comma list
    "code_snapshot_path": "TEXT",     # data/runs/<id>/ — exact code that ran
}


def _migrate(con: sqlite3.Connection) -> None:
    existing = {r["name"] for r in con.execute("PRAGMA table_info(runs)").fetchall()}
    for col, decl in _RUNS_MIGRATIONS.items():
        if col not in existing:
            con.execute(f"ALTER TABLE runs ADD COLUMN {col} {decl}")
    con.execute("CREATE INDEX IF NOT EXISTS idx_runs_competition ON runs(competition)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_runs_batch ON runs(batch_id)")

    conv_existing = {r["name"] for r in con.execute("PRAGMA table_info(conversations)").fetchall()}
    for col, decl in _CONV_MIGRATIONS.items():
        if col not in conv_existing:
            con.execute(f"ALTER TABLE conversations ADD COLUMN {col} {decl}")


def init_db() -> None:
    with connect() as con:
        con.executescript(SCHEMA)
        _migrate(con)


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    with _lock:
        con = sqlite3.connect(DB_PATH, isolation_level=None, timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        try:
            yield con
        finally:
            con.close()
