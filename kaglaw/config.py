from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("KAGLAW_DATA", ROOT / "data"))

ACCOUNTS_DIR = DATA_DIR / "accounts"
NOTEBOOKS_DIR = DATA_DIR / "notebooks"
OUTPUTS_DIR = DATA_DIR / "outputs"
EXPORTS_DIR = DATA_DIR / "exports"
RUNS_DIR = DATA_DIR / "runs"  # per-run code snapshots (reproducibility)
LOCAL_RUNS_DIR = DATA_DIR / "local_runs"  # working dirs for local test-runs
DB_PATH = DATA_DIR / "kaglaw.sqlite3"

for p in (DATA_DIR, ACCOUNTS_DIR, NOTEBOOKS_DIR, OUTPUTS_DIR, EXPORTS_DIR, RUNS_DIR, LOCAL_RUNS_DIR):
    p.mkdir(parents=True, exist_ok=True)

HOST = os.environ.get("KAGLAW_HOST", "127.0.0.1")
PORT = int(os.environ.get("KAGLAW_PORT", "8765"))

POLL_INTERVAL_SECONDS = int(os.environ.get("KAGLAW_POLL_INTERVAL", "60"))

# ---- Orchestration / quota (Phase 2+3) ----
# Max kernels kaglaw will have queued/running per nick at once (Kaggle limits
# concurrent GPU sessions; keep low to be safe).
MAX_CONCURRENT_PER_NICK = int(os.environ.get("KAGLAW_MAX_CONCURRENT_PER_NICK", "1"))
# Kaggle weekly GPU quota per account (hours). Used only as a budget estimate.
GPU_WEEKLY_HOURS = float(os.environ.get("KAGLAW_GPU_WEEKLY_HOURS", "30"))
# How often the dispatcher tries to launch queued jobs.
DISPATCH_INTERVAL_SECONDS = int(os.environ.get("KAGLAW_DISPATCH_INTERVAL", "30"))
# Hard cap on combos a single sweep may enqueue.
MAX_SWEEP_JOBS = int(os.environ.get("KAGLAW_MAX_SWEEP_JOBS", "200"))

# Interpreter + default timeout for local test-runs (smoke run before pushing).
LOCAL_PYTHON = os.environ.get("KAGLAW_LOCAL_PYTHON", "")  # "" = same interpreter running kaglaw
LOCAL_RUN_TIMEOUT = int(os.environ.get("KAGLAW_LOCAL_RUN_TIMEOUT", "300"))

# ---- Agent memory + context compaction ----
# Max chars of saved memories injected into the system prompt each turn.
MEMORY_INJECT_MAX_CHARS = int(os.environ.get("KAGLAW_MEMORY_MAX_CHARS", "8000"))
# When a chat's history exceeds this many chars, compact the old part into a summary.
CONTEXT_BUDGET_CHARS = int(os.environ.get("KAGLAW_CONTEXT_BUDGET_CHARS", "48000"))
# Keep at least this many chars of the most-recent messages verbatim (rest summarized).
CONTEXT_RECENT_CHARS = int(os.environ.get("KAGLAW_CONTEXT_RECENT_CHARS", "20000"))
