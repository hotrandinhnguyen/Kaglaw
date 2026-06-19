"""Batch orchestration (Phase 2): turn a parameter grid into many experiments and
launch them across accounts, respecting per-nick concurrency + GPU budget.

Flow:
  launch_sweep(...)  -> expands {{PARAM}} grid into N jobs (status=queued)
  dispatch_jobs()    -> (called by the scheduler) picks a nick with a free slot for
                        each queued job and pushes it, creating a run; retries failures
  _reconcile()       -> flips job->done when its run is no longer active
  detect_finished_batches() -> writes a notification when a whole batch completes

Templates: the notebook source must contain literal `{{name}}` placeholders for each
grid key, e.g. `lr = {{lr}}`. Each combo substitutes those, so params stay queryable.
"""

from __future__ import annotations

import itertools
import json
import random
from typing import Any

from . import accounts as account_mod
from . import actions, budgets, db, notebook_builder
from .config import MAX_SWEEP_JOBS

_RUNNING_STATES = ("queued", "running", "queueing", "pending", None)


# --------------------------------------------------------------------------- #
# grid expansion
# --------------------------------------------------------------------------- #

def expand_grid(
    grid: dict[str, list[Any]],
    *,
    search: str = "grid",
    n: int | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Return a list of param combos from a grid.

    grid:   {"lr": [0.1, 0.05], "seed": [1, 2]}
    search: 'grid'  -> full cartesian product (capped)
            'random'-> n random combos (deduped)
    """
    keys = list(grid.keys())
    if not keys:
        return []
    value_lists = [list(grid[k]) for k in keys]
    if search == "random":
        rng = random.Random(seed)
        n = n or 10
        seen: set[tuple] = set()
        combos: list[dict[str, Any]] = []
        attempts = 0
        max_attempts = n * 50 + 100
        while len(combos) < n and attempts < max_attempts:
            attempts += 1
            picked = tuple(rng.choice(v) for v in value_lists)
            if picked in seen:
                continue
            seen.add(picked)
            combos.append(dict(zip(keys, picked)))
        return combos
    # grid
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*value_lists)]
    return combos[:MAX_SWEEP_JOBS]


def _replacements_for(combo: dict[str, Any]) -> dict[str, str]:
    """Map each grid key to its {{key}} placeholder substitution."""
    return {"{{" + k + "}}": str(v) for k, v in combo.items()}


# --------------------------------------------------------------------------- #
# launch
# --------------------------------------------------------------------------- #

def launch_sweep(
    notebook_id: int,
    grid: dict[str, list[Any]],
    *,
    competition: str | None = None,
    nicks: list[str] | None = None,
    search: str = "grid",
    n: int | None = None,
    seed: int = 0,
    tags: str | None = None,
    version_notes: str = "",
) -> dict[str, Any]:
    """Validate the template, expand the grid, and enqueue one job per combo.
    Jobs are dispatched asynchronously by the scheduler."""
    nb = actions.get_notebook(notebook_id)
    if not nb:
        return {"ok": False, "error": f"notebook {notebook_id} not found"}

    # Validate every grid key has a {{placeholder}} in the source.
    try:
        source = notebook_builder.read_source(nb["local_path"])
    except Exception as exc:
        return {"ok": False, "error": f"cannot read notebook source: {exc}"}
    missing = [k for k in grid if ("{{" + k + "}}") not in source]
    if missing:
        return {
            "ok": False,
            "error": f"notebook source has no placeholder(s) for: {missing}. "
                     f"Add literal {{{{name}}}} where the value should go (e.g. lr = {{{{lr}}}}).",
        }

    combos = expand_grid(grid, search=search, n=n, seed=seed)
    if not combos:
        return {"ok": False, "error": "grid expanded to 0 combos"}
    if len(combos) > MAX_SWEEP_JOBS:
        combos = combos[:MAX_SWEEP_JOBS]

    competition = competition or actions._first_competition(nb)
    allowed = nicks or [a.nick for a in account_mod.list_accounts()]
    if not allowed:
        return {"ok": False, "error": "no accounts available to run the sweep"}

    batch_id = actions._new_batch_id()
    allowed_json = json.dumps(allowed)
    need_gpu = int(bool(nb["enable_gpu"]))
    with db.connect() as con:
        for idx, combo in enumerate(combos):
            con.execute(
                """INSERT INTO jobs
                (batch_id, notebook_id, competition, params, replacements, allowed_nicks,
                 slug_suffix, version_notes, tags, need_gpu, status)
                VALUES(?,?,?,?,?,?,?,?,?,?, 'queued')""",
                (
                    batch_id, notebook_id, competition,
                    json.dumps(combo), json.dumps(_replacements_for(combo)),
                    allowed_json, f"-{batch_id[-4:]}-{idx}", version_notes, tags, need_gpu,
                ),
            )
    return {
        "ok": True,
        "batch_id": batch_id,
        "n_jobs": len(combos),
        "allowed_nicks": allowed,
        "competition": competition,
        "note": "Jobs queued. The dispatcher launches them respecting per-nick concurrency "
                "and GPU budget. Track with batch_status(batch_id) or the /batches page.",
    }


# --------------------------------------------------------------------------- #
# dispatch (called by the scheduler)
# --------------------------------------------------------------------------- #

def _reconcile_running_jobs() -> int:
    """Flip jobs whose run has finished from 'running' to 'done'."""
    flipped = 0
    with db.connect() as con:
        rows = con.execute(
            "SELECT j.id, r.status AS rstatus FROM jobs j "
            "JOIN runs r ON r.id=j.run_id WHERE j.status='running'"
        ).fetchall()
        for r in rows:
            st = (r["rstatus"] or "").lower()
            if st and st not in ("queued", "running", "queueing", "pending"):
                con.execute("UPDATE jobs SET status='done' WHERE id=?", (r["id"],))
                flipped += 1
    return flipped


def dispatch_jobs(max_dispatch: int | None = None) -> dict[str, Any]:
    """Try to launch queued jobs. Returns a small summary. Safe to call repeatedly."""
    _reconcile_running_jobs()
    with db.connect() as con:
        jobs = con.execute(
            "SELECT * FROM jobs WHERE status='queued' ORDER BY priority DESC, id ASC"
        ).fetchall()
    if not jobs:
        return {"dispatched": 0, "queued_remaining": 0}

    in_flight: dict[str, int] = {}
    dispatched = 0
    errors: list[str] = []
    for job in jobs:
        if max_dispatch is not None and dispatched >= max_dispatch:
            break
        allowed = json.loads(job["allowed_nicks"] or "[]") or [
            a.nick for a in account_mod.list_accounts()
        ]
        nick = budgets.pick_nick(allowed, need_gpu=bool(job["need_gpu"]), in_flight=in_flight)
        if nick is None:
            continue  # everyone busy for this job right now; try next tick

        variant = {
            "nick": nick,
            "replacements": json.loads(job["replacements"] or "{}"),
            "slug_suffix": job["slug_suffix"] or "",
            "params": json.loads(job["params"] or "{}"),
        }
        try:
            results = actions.push_variants(
                job["notebook_id"], [variant],
                version_notes=job["version_notes"] or "",
                competition=job["competition"], tags=job["tags"],
                batch_id=job["batch_id"],
            )
            res = results[0] if results else {"ok": False, "error": "no result"}
        except Exception as exc:  # noqa: BLE001
            res = {"ok": False, "error": str(exc)}

        if res.get("ok"):
            in_flight[nick] = in_flight.get(nick, 0) + 1
            dispatched += 1
            with db.connect() as con:
                con.execute(
                    "UPDATE jobs SET status='running', run_id=?, nick=?, attempts=attempts+1, "
                    "dispatched_at=datetime('now'), error=NULL WHERE id=?",
                    (res.get("run_id"), nick, job["id"]),
                )
        else:
            err = str(res.get("error") or "push failed")
            # deterministic errors (bad template) shouldn't be retried forever
            deterministic = "not found" in err.lower()
            new_attempts = (job["attempts"] or 0) + 1
            terminal = deterministic or new_attempts >= (job["max_attempts"] or 2)
            with db.connect() as con:
                con.execute(
                    "UPDATE jobs SET attempts=?, error=?, status=? WHERE id=?",
                    (new_attempts, err, "failed" if terminal else "queued", job["id"]),
                )
            errors.append(f"job {job['id']}: {err}")

    with db.connect() as con:
        remaining = con.execute("SELECT COUNT(*) FROM jobs WHERE status='queued'").fetchone()[0]
    return {"dispatched": dispatched, "queued_remaining": remaining, "errors": errors[:5]}


# --------------------------------------------------------------------------- #
# batch reporting
# --------------------------------------------------------------------------- #

def batch_status(batch_id: str) -> dict[str, Any]:
    with db.connect() as con:
        jobs = con.execute(
            "SELECT id, status, nick, run_id, params, error FROM jobs WHERE batch_id=? ORDER BY id",
            (batch_id,),
        ).fetchall()
    totals: dict[str, int] = {}
    job_list = []
    for j in jobs:
        totals[j["status"]] = totals.get(j["status"], 0) + 1
        job_list.append({
            "job_id": j["id"], "status": j["status"], "nick": j["nick"],
            "run_id": j["run_id"], "params": json.loads(j["params"] or "{}"),
            "error": (j["error"] or "")[:200] or None,
        })
    matrix = actions.compare_experiments(batch_id=batch_id, limit=MAX_SWEEP_JOBS)
    return {
        "batch_id": batch_id,
        "totals": totals,
        "n_jobs": len(jobs),
        "best": matrix.get("best"),
        "param_keys": matrix.get("param_keys"),
        "jobs": job_list,
        "results": matrix.get("rows"),
    }


def list_batches(limit: int = 50) -> list[dict[str, Any]]:
    """Unified view over sweep batches (jobs) and direct-push batches (runs)."""
    with db.connect() as con:
        job_batches = con.execute(
            """SELECT batch_id,
                      MIN(enqueued_at) AS created,
                      COUNT(*) AS n_jobs,
                      SUM(status='queued')  AS queued,
                      SUM(status='running') AS running,
                      SUM(status='done')    AS done,
                      SUM(status='failed')  AS failed,
                      SUM(status='canceled')AS canceled,
                      MAX(competition) AS competition
               FROM jobs GROUP BY batch_id""",
        ).fetchall()
        run_stats = con.execute(
            """SELECT batch_id,
                      COUNT(*) AS n_runs,
                      MAX(metric_value) AS best_metric,
                      MIN(pushed_at) AS created,
                      MAX(competition) AS competition
               FROM runs WHERE batch_id IS NOT NULL GROUP BY batch_id""",
        ).fetchall()
    runs_by_batch = {r["batch_id"]: r for r in run_stats}
    out: list[dict[str, Any]] = []
    seen = set()
    for b in job_batches:
        rs = runs_by_batch.get(b["batch_id"])
        seen.add(b["batch_id"])
        out.append({
            "batch_id": b["batch_id"],
            "created": b["created"],
            "competition": b["competition"] or (rs["competition"] if rs else None),
            "n_jobs": b["n_jobs"],
            "queued": b["queued"], "running": b["running"], "done": b["done"],
            "failed": b["failed"], "canceled": b["canceled"],
            "n_runs": rs["n_runs"] if rs else 0,
            "best_metric": rs["best_metric"] if rs else None,
            "kind": "sweep",
        })
    # direct-push batches that have no jobs
    for bid, rs in runs_by_batch.items():
        if bid in seen:
            continue
        out.append({
            "batch_id": bid, "created": rs["created"], "competition": rs["competition"],
            "n_jobs": 0, "queued": 0, "running": 0, "done": 0, "failed": 0, "canceled": 0,
            "n_runs": rs["n_runs"], "best_metric": rs["best_metric"], "kind": "push",
        })
    out.sort(key=lambda x: x["created"] or "", reverse=True)
    return out[:limit]


def nick_status(nicks: list[str] | None = None) -> list[dict[str, Any]]:
    """Per-nick live status: quota estimate + WHICH runs are active right now
    (notebook, slug, status, competition, batch) + queued jobs waiting for it."""
    nicks = nicks or [a.nick for a in account_mod.list_accounts()]
    active = actions.list_runs(active_only=True, limit=500)
    by_nick: dict[str, list[dict[str, Any]]] = {}
    for r in active:
        by_nick.setdefault(r["account_nick"], []).append({
            "run_id": r["id"],
            "notebook": r.get("notebook_title"),
            "slug": r["slug"],
            "status": r["status"],
            "competition": r.get("competition"),
            "version": r.get("version_number"),
            "batch_id": r.get("batch_id"),
            "pushed_at": r["pushed_at"],
        })
    # queued jobs whose allowed_nicks include each nick (waiting to be dispatched)
    with db.connect() as con:
        qjobs = con.execute(
            "SELECT allowed_nicks FROM jobs WHERE status='queued'"
        ).fetchall()
    waiting: dict[str, int] = {}
    for j in qjobs:
        try:
            allowed = json.loads(j["allowed_nicks"] or "[]")
        except Exception:
            allowed = []
        for nk in (allowed or nicks):
            waiting[nk] = waiting.get(nk, 0) + 1

    out: list[dict[str, Any]] = []
    for nick in nicks:
        u = budgets.nick_usage(nick)
        out.append({
            **u,
            "active_runs": by_nick.get(nick, []),
            "queued_waiting": waiting.get(nick, 0),
        })
    return out


def cancel_batch(batch_id: str) -> dict[str, Any]:
    with db.connect() as con:
        cur = con.execute(
            "UPDATE jobs SET status='canceled' WHERE batch_id=? AND status='queued'",
            (batch_id,),
        )
    return {"ok": True, "batch_id": batch_id, "canceled_queued": cur.rowcount}


# --------------------------------------------------------------------------- #
# notifications
# --------------------------------------------------------------------------- #

def detect_finished_batches() -> int:
    """Emit a notification for each sweep batch that just fully finished.

    Note: gather candidates with the connection CLOSED before calling
    compare_experiments (which opens its own connection) — the db lock is not
    reentrant, so nesting connections would deadlock.
    """
    with db.connect() as con:
        batches = con.execute(
            """SELECT batch_id,
                      SUM(status IN ('queued','running')) AS active,
                      COUNT(*) AS total
               FROM jobs GROUP BY batch_id""",
        ).fetchall()
        done_ids = [b["batch_id"] for b in batches if b["total"] and not b["active"]]
        already = {
            r["ref"] for r in con.execute(
                "SELECT ref FROM notifications WHERE kind='batch_done'"
            ).fetchall()
        }
        totals = {b["batch_id"]: b["total"] for b in batches}
    pending = [bid for bid in done_ids if bid not in already]

    created = 0
    for bid in pending:
        matrix = actions.compare_experiments(batch_id=bid, limit=MAX_SWEEP_JOBS)
        best = matrix.get("best")
        best_txt = (
            f"best {best['metric_name']}={best['metric_value']} (run #{best['run_id']}, "
            f"{best['nick']})" if best else "no parsed metric yet"
        )
        with db.connect() as con:
            con.execute(
                "INSERT INTO notifications(kind, ref, title, body) VALUES(?,?,?,?)",
                ("batch_done", bid, f"Batch {bid} hoàn tất",
                 f"{totals[bid]} job xong. {best_txt}."),
            )
        created += 1
    return created


def list_notifications(unseen_only: bool = False, limit: int = 30) -> list[dict[str, Any]]:
    q = "SELECT * FROM notifications"
    if unseen_only:
        q += " WHERE seen=0"
    q += " ORDER BY id DESC LIMIT ?"
    with db.connect() as con:
        rows = con.execute(q, (limit,)).fetchall()
    return [dict(r) for r in rows]


def count_unseen() -> int:
    with db.connect() as con:
        return con.execute("SELECT COUNT(*) FROM notifications WHERE seen=0").fetchone()[0]


def mark_notifications_seen(ids: list[int] | None = None) -> dict[str, Any]:
    with db.connect() as con:
        if ids:
            con.executemany("UPDATE notifications SET seen=1 WHERE id=?", [(i,) for i in ids])
        else:
            con.execute("UPDATE notifications SET seen=1 WHERE seen=0")
    return {"ok": True}
