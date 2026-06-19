"""Per-nick quota / budget awareness (Phase 3).

Kaggle does not expose quota via API, so these are ESTIMATES used to spread work
across accounts sensibly:
  - GPU hours used in the last 7 days  = sum(runtime_seconds) of GPU runs in window
  - running_now                        = runs currently queued/running for the nick
  - submissions today (per competition)

`pick_nick` is what the dispatcher calls: among the allowed nicks, keep only those
with a free concurrency slot, then choose the one with the most remaining GPU budget
(falling back to the least-loaded when GPU isn't needed).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from . import db
from .config import GPU_WEEKLY_HOURS, MAX_CONCURRENT_PER_NICK

_RUNNING_STATES = ("queued", "running", "queueing", "pending")


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace(" ", "T"))
    except Exception:
        return None


def nick_usage(nick: str) -> dict[str, Any]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).replace(tzinfo=None)
    gpu_seconds = 0.0
    running = 0
    with db.connect() as con:
        rows = con.execute(
            "SELECT status, used_gpu, runtime_seconds, pushed_at FROM runs WHERE account_nick=?",
            (nick,),
        ).fetchall()
        subs_today = con.execute(
            "SELECT COUNT(*) FROM submissions WHERE account_nick=? AND substr(submitted_at,1,10)=?",
            (nick, datetime.now().strftime("%Y-%m-%d")),
        ).fetchone()[0]
    for r in rows:
        st = (r["status"] or "").lower()
        if st in _RUNNING_STATES:
            running += 1
        if r["used_gpu"] and r["runtime_seconds"]:
            dt = _parse_dt(r["pushed_at"])
            if dt is None or dt.replace(tzinfo=None) >= cutoff:
                gpu_seconds += float(r["runtime_seconds"])
    gpu_hours = gpu_seconds / 3600.0
    return {
        "nick": nick,
        "running_now": running,
        "free_slots": max(0, MAX_CONCURRENT_PER_NICK - running),
        "gpu_hours_7d": round(gpu_hours, 2),
        "gpu_hours_remaining": round(max(0.0, GPU_WEEKLY_HOURS - gpu_hours), 2),
        "submissions_today": subs_today,
    }


def all_usages(nicks: list[str]) -> list[dict[str, Any]]:
    return [nick_usage(n) for n in nicks]


def pick_nick(
    allowed: list[str],
    *,
    need_gpu: bool = False,
    in_flight: dict[str, int] | None = None,
) -> str | None:
    """Choose the best nick from `allowed` that still has a free concurrency slot.
    `in_flight` lets the caller account for jobs it already dispatched this tick."""
    in_flight = in_flight or {}
    best: tuple[float, int, str] | None = None  # (remaining_gpu, -running, nick)
    for nick in allowed:
        u = nick_usage(nick)
        running = u["running_now"] + in_flight.get(nick, 0)
        if running >= MAX_CONCURRENT_PER_NICK:
            continue
        # prefer most remaining GPU budget when GPU is needed, else least-loaded
        key_gpu = u["gpu_hours_remaining"] if need_gpu else 0.0
        cand = (key_gpu, -running, nick)
        if best is None or cand > best:
            best = cand
    return best[2] if best else None
