"""High-level orchestration: push 1 notebook to N nicks, sync status,
sync submissions, compute leaderboard rank."""

from __future__ import annotations

import json
import re
import shutil
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import db, kaggle_client, metrics
from .accounts import Account, get_account, list_accounts
from .config import NOTEBOOKS_DIR, OUTPUTS_DIR, RUNS_DIR


# ---------- Notebook registration ----------

@dataclass
class NotebookSpec:
    title: str
    local_path: str  # absolute path to a notebook file OR a folder w/ kernel-metadata.json
    language: str = "python"
    kernel_type: str = "notebook"  # or 'script'
    enable_gpu: bool = False
    enable_tpu: bool = False
    enable_internet: bool = True
    dataset_sources: list[str] | None = None
    competition_sources: list[str] | None = None
    kernel_sources: list[str] | None = None


def register_notebook(spec: NotebookSpec) -> int:
    with db.connect() as con:
        cur = con.execute(
            """INSERT INTO notebooks
            (title, local_path, language, kernel_type, enable_gpu, enable_tpu,
             enable_internet, dataset_sources, competition_sources, kernel_sources)
            VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                spec.title,
                spec.local_path,
                spec.language,
                spec.kernel_type,
                int(spec.enable_gpu),
                int(spec.enable_tpu),
                int(spec.enable_internet),
                json.dumps(spec.dataset_sources or []),
                json.dumps(spec.competition_sources or []),
                json.dumps(spec.kernel_sources or []),
            ),
        )
        return int(cur.lastrowid)


def list_notebooks() -> list[dict[str, Any]]:
    with db.connect() as con:
        rows = con.execute("SELECT * FROM notebooks ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


def get_notebook(nb_id: int) -> dict[str, Any] | None:
    with db.connect() as con:
        r = con.execute("SELECT * FROM notebooks WHERE id=?", (nb_id,)).fetchone()
    return dict(r) if r else None


def delete_notebook(nb_id: int) -> None:
    with db.connect() as con:
        con.execute("DELETE FROM notebooks WHERE id=?", (nb_id,))


def _truthy(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "on")
    return default


def import_kernel(
    kernel_ref: str,
    as_nick: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Pull an existing Kaggle kernel (`<user>/<slug>`) — source + metadata — into kaglaw
    as a registered notebook, so it can be edited locally and pushed back as a new version.
    Uses `as_nick`'s credentials (any nick can pull a public kernel)."""
    import tempfile

    account = get_account(as_nick) if as_nick else (list_accounts()[0] if list_accounts() else None)
    if not account:
        return {"ok": False, "error": "no accounts configured to pull with"}

    with tempfile.TemporaryDirectory(prefix="kaglaw_import_") as td:
        tdp = Path(td)
        try:
            kaggle_client.kernel_pull(account, kernel_ref, tdp, metadata=True)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"pull failed: {exc}"}

        meta: dict[str, Any] = {}
        mp = tdp / "kernel-metadata.json"
        if mp.exists():
            try:
                meta = json.loads(mp.read_text(encoding="utf-8"))
            except Exception:
                meta = {}

        src = None
        for ext in (".ipynb", ".py", ".r", ".R", ".Rmd"):
            files = list(tdp.glob(f"*{ext}"))
            if files:
                src = files[0]
                break
        if not src:
            return {"ok": False, "error": "no source file found in the pulled kernel"}

        nb_title = title or meta.get("title") or kernel_ref.split("/")[-1]
        folder = NOTEBOOKS_DIR / slugify(nb_title)
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / src.name
        shutil.copy2(src, target)

    spec = NotebookSpec(
        title=nb_title,
        local_path=str(target),
        language=meta.get("language", "python"),
        kernel_type=meta.get("kernel_type", "notebook"),
        enable_gpu=_truthy(meta.get("enable_gpu")),
        enable_tpu=_truthy(meta.get("enable_tpu")),
        enable_internet=_truthy(meta.get("enable_internet"), True),
        dataset_sources=meta.get("dataset_sources") or [],
        competition_sources=meta.get("competition_sources") or [],
        kernel_sources=meta.get("kernel_sources") or [],
    )
    nb_id = register_notebook(spec)
    return {
        "ok": True,
        "notebook_id": nb_id,
        "title": nb_title,
        "source_file": src.name,
        "language": spec.language,
        "kernel_type": spec.kernel_type,
        "enable_gpu": spec.enable_gpu,
        "from_kernel": kernel_ref,
        "via_nick": account.nick,
        "note": "Sửa code rồi push dưới nick SỞ HỮU kernel (và giữ title để slug khớp) "
                "thì sẽ tạo version mới của chính kernel đó.",
    }


# ---------- Push to N accounts ----------

_SLUG_RE = re.compile(r"[^a-z0-9-]+")


def slugify(s: str) -> str:
    s = s.lower().strip()
    s = _SLUG_RE.sub("-", s).strip("-")
    return s[:60] or "kernel"


# ---------- Experiment helpers ----------

def _new_batch_id() -> str:
    """Short id grouping one sweep / multi-nick push: <timestamp>-<rand4>."""
    return datetime.now().strftime("%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:4]


def _first_competition(nb: dict[str, Any]) -> str | None:
    try:
        comps = json.loads(nb.get("competition_sources") or "[]")
        return comps[0] if comps else None
    except Exception:
        return None


def _snapshot_code(staging: Path, run_id: int) -> str | None:
    """Move the materialized kernel folder to data/runs/<run_id>/ for reproducibility."""
    dest = RUNS_DIR / str(run_id)
    try:
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        shutil.move(str(staging), str(dest))
        return str(dest)
    except Exception:
        return None


def _build_kernel_folder(nb: dict[str, Any], account: Account, dest: Path) -> tuple[Path, str]:
    """Materialize a folder with the notebook file + kernel-metadata.json bound to
    `<account.username>/<slug>`."""
    dest.mkdir(parents=True, exist_ok=True)
    src = Path(nb["local_path"])
    if src.is_dir():
        # Copy contents
        for p in src.iterdir():
            if p.is_file():
                shutil.copy2(p, dest / p.name)
            elif p.is_dir():
                shutil.copytree(p, dest / p.name, dirs_exist_ok=True)
        nb_file = _find_notebook_file(dest, nb["language"], nb["kernel_type"])
    else:
        shutil.copy2(src, dest / src.name)
        nb_file = dest / src.name

    slug = slugify(nb["title"])
    meta = {
        "id": f"{account.username}/{slug}",
        "title": nb["title"],
        "code_file": nb_file.name,
        "language": nb["language"],
        "kernel_type": nb["kernel_type"],
        "is_private": True,
        "enable_gpu": bool(nb["enable_gpu"]),
        "enable_tpu": bool(nb["enable_tpu"]),
        "enable_internet": bool(nb["enable_internet"]),
        "dataset_sources": json.loads(nb["dataset_sources"] or "[]"),
        "competition_sources": json.loads(nb["competition_sources"] or "[]"),
        "kernel_sources": json.loads(nb["kernel_sources"] or "[]"),
    }
    (dest / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return dest, slug


def _find_notebook_file(folder: Path, language: str, kernel_type: str) -> Path:
    if kernel_type == "notebook":
        candidates = list(folder.glob("*.ipynb"))
    else:
        ext = {"python": ".py", "r": ".R", "rmarkdown": ".Rmd"}.get(language.lower(), ".py")
        candidates = list(folder.glob(f"*{ext}"))
    if not candidates:
        raise FileNotFoundError(f"No notebook/script file found in {folder}")
    return candidates[0]


def _apply_replacements(folder: Path, language: str, replacements: dict[str, str]) -> dict[str, int]:
    """Apply literal-string replacements to the source file in the staged folder.

    For .ipynb: parses JSON, walks each code cell's `source`, replaces inside each line.
    For .py / scripts: plain file str.replace.
    Returns {original_pattern: count} for the user to verify.
    """
    counts = {k: 0 for k in replacements}
    if not replacements:
        return counts
    # Find the source file
    nb_files = list(folder.glob("*.ipynb"))
    if nb_files:
        nbf = nb_files[0]
        data = json.loads(nbf.read_text(encoding="utf-8"))
        for cell in data.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            src = cell.get("source", [])
            if isinstance(src, str):
                src_lines = src.splitlines(keepends=True)
            else:
                src_lines = list(src)
            new_lines = []
            for line in src_lines:
                for k, v in replacements.items():
                    if k in line:
                        line = line.replace(k, v)
                        counts[k] += 1
                new_lines.append(line)
            cell["source"] = new_lines
        nbf.write_text(json.dumps(data, indent=1), encoding="utf-8")
        return counts
    # Script
    script_files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in (".py", ".r", ".rmd")]
    if not script_files:
        return counts
    sf = script_files[0]
    text = sf.read_text(encoding="utf-8")
    for k, v in replacements.items():
        cnt = text.count(k)
        if cnt:
            text = text.replace(k, v)
            counts[k] = cnt
    sf.write_text(text, encoding="utf-8")
    return counts


def push_variants(
    notebook_id: int,
    variants: list[dict[str, Any]],
    version_notes: str = "",
    competition: str | None = None,
    tags: str | None = None,
    batch_id: str | None = None,
) -> list[dict[str, Any]]:
    """Push a notebook to multiple nicks, each with its own literal-string replacements
    applied to the notebook source. Variant shape:
        {"nick": "alt1", "replacements": {"SEED = 42": "SEED = 1"}, "slug_suffix": "-s1"}
    Each variant is recorded as an experiment (params=replacements) in a shared batch.
    Pass `batch_id` to attach these runs to an existing batch (e.g. a sweep).
    """
    nb = get_notebook(notebook_id)
    if not nb:
        raise ValueError(f"Notebook {notebook_id} not found")
    competition = competition or _first_competition(nb)
    batch_id = batch_id or _new_batch_id()
    results: list[dict[str, Any]] = []
    for variant in variants:
        nick = variant["nick"]
        account = get_account(nick)
        if not account:
            results.append({"nick": nick, "ok": False, "error": "Account not found"})
            continue
        replacements = variant.get("replacements") or {}
        # Clean params for the matrix; falls back to the replacement map.
        params = variant.get("params") or replacements
        staging = RUNS_DIR / f"_staging_{uuid.uuid4().hex}"
        try:
            folder, base_slug = _build_kernel_folder(nb, account, staging)
            counts = _apply_replacements(folder, nb["language"], replacements)
            if any(c == 0 for c in counts.values()):
                missing = [k for k, c in counts.items() if c == 0]
                shutil.rmtree(staging, ignore_errors=True)
                results.append({
                    "nick": nick, "ok": False,
                    "error": f"Replacement pattern not found in source: {missing}",
                })
                continue
            # Optional slug suffix to distinguish variants
            suffix = variant.get("slug_suffix") or ""
            if suffix:
                final_slug = slugify(f"{base_slug}{suffix}")
                meta_path = folder / "kernel-metadata.json"
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta["id"] = f"{account.username}/{final_slug}"
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            else:
                final_slug = base_slug

            resp = kaggle_client.kernel_push(account, folder)

            version_number = _extract_int(resp, ("versionNumber", "version_number"))
            status = (resp.get("status") or "queued") if resp else "queued"
            error = resp.get("error") if resp else None
            with db.connect() as con:
                cur = con.execute(
                    """INSERT INTO runs
                    (notebook_id, account_nick, slug, version_number, version_notes,
                     status, used_gpu, used_tpu, error, competition, params, batch_id, tags)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        notebook_id, nick, final_slug, version_number, version_notes,
                        str(status), nb["enable_gpu"], nb["enable_tpu"], error,
                        competition, json.dumps(params), batch_id, tags,
                    ),
                )
                run_id = int(cur.lastrowid)
            snap = _snapshot_code(staging, run_id)
            with db.connect() as con:
                con.execute("UPDATE runs SET code_snapshot_path=? WHERE id=?", (snap, run_id))
            results.append({
                "nick": nick, "ok": True, "run_id": run_id, "slug": final_slug,
                "version": version_number, "status": status, "replacements_applied": counts,
                "batch_id": batch_id,
            })
        except Exception as exc:
            shutil.rmtree(staging, ignore_errors=True)
            tb = traceback.format_exc(limit=4)
            with db.connect() as con:
                con.execute(
                    """INSERT INTO runs(notebook_id, account_nick, slug, status, error,
                                        competition, params, batch_id, tags)
                       VALUES(?,?,?,?,?,?,?,?,?)""",
                    (notebook_id, nick, slugify(nb["title"]), "push_failed", f"{exc}\n{tb}",
                     competition, json.dumps(params), batch_id, tags),
                )
            results.append({"nick": nick, "ok": False, "error": str(exc)})
    return results


def push_notebook_to_accounts(
    notebook_id: int,
    nicks: list[str],
    version_notes: str = "",
    competition: str | None = None,
    tags: str | None = None,
) -> list[dict[str, Any]]:
    """Push 1 notebook to several accounts. Records one experiment row per (notebook, account)
    in `runs`, snapshots the exact code, and groups the push under one batch_id."""
    nb = get_notebook(notebook_id)
    if not nb:
        raise ValueError(f"Notebook {notebook_id} not found")

    competition = competition or _first_competition(nb)
    batch_id = _new_batch_id()
    results: list[dict[str, Any]] = []
    for nick in nicks:
        account = get_account(nick)
        if not account:
            results.append({"nick": nick, "ok": False, "error": "Account not found"})
            continue

        staging = RUNS_DIR / f"_staging_{uuid.uuid4().hex}"
        try:
            folder, slug = _build_kernel_folder(nb, account, staging)
            resp = kaggle_client.kernel_push(account, folder)

            version_number = _extract_int(resp, ("versionNumber", "version_number"))
            status = (resp.get("status") or resp.get("invalidTags") or "queued") if resp else "queued"
            error = resp.get("error") if resp else None

            with db.connect() as con:
                cur = con.execute(
                    """INSERT INTO runs
                    (notebook_id, account_nick, slug, version_number, version_notes,
                     status, used_gpu, used_tpu, error, competition, params, batch_id, tags)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        notebook_id, nick, slug, version_number, version_notes,
                        str(status), nb["enable_gpu"], nb["enable_tpu"], error,
                        competition, "{}", batch_id, tags,
                    ),
                )
                run_id = int(cur.lastrowid)
            snap = _snapshot_code(staging, run_id)
            with db.connect() as con:
                con.execute("UPDATE runs SET code_snapshot_path=? WHERE id=?", (snap, run_id))

            results.append(
                {
                    "nick": nick,
                    "ok": True,
                    "run_id": run_id,
                    "slug": slug,
                    "version": version_number,
                    "status": status,
                    "batch_id": batch_id,
                }
            )
        except Exception as exc:
            shutil.rmtree(staging, ignore_errors=True)
            tb = traceback.format_exc(limit=4)
            with db.connect() as con:
                con.execute(
                    """INSERT INTO runs
                    (notebook_id, account_nick, slug, status, error, competition, batch_id, tags)
                    VALUES(?,?,?,?,?,?,?,?)""",
                    (notebook_id, nick, slugify(nb["title"]), "push_failed", f"{exc}\n{tb}",
                     competition, batch_id, tags),
                )
            results.append({"nick": nick, "ok": False, "error": str(exc)})
    return results


# ---------- Sync run statuses ----------

_RUNNING_STATES = {"queued", "running", "queueing", "pending"}


def list_runs(active_only: bool = False, limit: int = 500) -> list[dict[str, Any]]:
    q = "SELECT r.*, n.title AS notebook_title FROM runs r LEFT JOIN notebooks n ON n.id=r.notebook_id"
    params: tuple = ()
    if active_only:
        placeholders = ",".join("?" * len(_RUNNING_STATES))
        q += f" WHERE r.status IN ({placeholders}) OR r.status IS NULL"
        params = tuple(_RUNNING_STATES)
    q += " ORDER BY r.id DESC LIMIT ?"
    params = params + (limit,)
    with db.connect() as con:
        rows = con.execute(q, params).fetchall()
    return [dict(r) for r in rows]


def sync_run_status(run_id: int, pull_output_on_complete: bool = True) -> dict[str, Any]:
    with db.connect() as con:
        row = con.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    if not row:
        return {"ok": False, "error": "run not found"}
    account = get_account(row["account_nick"])
    if not account:
        return {"ok": False, "error": "account missing"}
    kernel_ref = f"{account.username}/{row['slug']}"
    try:
        resp = kaggle_client.kernel_status(account, kernel_ref)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    status = (resp.get("status") or "").lower() or row["status"]
    failure = resp.get("failureMessage") or resp.get("failure_message")
    completed_at = None
    runtime_seconds = row["runtime_seconds"]
    output_path = row["output_path"]
    log_summary = row["log_summary"]
    metric_name = row["metric_name"]
    metric_value = row["metric_value"]

    if status not in _RUNNING_STATES:
        completed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        # crude runtime estimate: completed_at - pushed_at if available
        try:
            pushed = datetime.fromisoformat(row["pushed_at"].replace(" ", "T"))
            delta = (datetime.fromisoformat(completed_at) - pushed.replace(tzinfo=timezone.utc)).total_seconds()
            if delta > 0:
                runtime_seconds = float(delta)
        except Exception:
            pass

        if pull_output_on_complete and status == "complete":
            try:
                dest = OUTPUTS_DIR / row["account_nick"] / row["slug"] / f"v{row['version_number'] or 'latest'}"
                files = kaggle_client.kernel_pull_output(account, kernel_ref, dest)
                output_path = str(dest)
                log_summary = "; ".join(files[:10])
                # Parse a research metric (CV/AUC/score…) out of the pulled log.
                mn, mv, msrc = metrics.extract_from_dir(dest)
                if mv is not None:
                    metric_name, metric_value = mn, mv
                    log_summary = f"[{mn}={mv}] " + log_summary
            except Exception as exc:
                log_summary = f"output pull failed: {exc}"

    with db.connect() as con:
        con.execute(
            """UPDATE runs SET status=?, completed_at=?, runtime_seconds=?,
               output_path=?, log_summary=?, metric_name=?, metric_value=?,
               error=COALESCE(?, error) WHERE id=?""",
            (status, completed_at, runtime_seconds, output_path, log_summary,
             metric_name, metric_value, failure, run_id),
        )
    return {"ok": True, "status": status,
            "metric": ({"name": metric_name, "value": metric_value} if metric_value is not None else None)}


def sync_all_active_runs() -> dict[str, Any]:
    runs = list_runs(active_only=True, limit=200)
    n = 0
    for r in runs:
        sync_run_status(r["id"])
        n += 1
    return {"synced": n}


# ---------- Experiments (runs enriched with params + metric) ----------

def list_experiments(
    *,
    competition: str | None = None,
    batch_id: str | None = None,
    notebook_id: int | None = None,
    nick: str | None = None,
    has_metric: bool = False,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Return runs as experiment records (params parsed, metric included)."""
    q = ("SELECT r.*, n.title AS notebook_title FROM runs r "
         "LEFT JOIN notebooks n ON n.id=r.notebook_id WHERE 1=1")
    params: list[Any] = []
    if competition:
        q += " AND r.competition=?"; params.append(competition)
    if batch_id:
        q += " AND r.batch_id=?"; params.append(batch_id)
    if notebook_id is not None:
        q += " AND r.notebook_id=?"; params.append(notebook_id)
    if nick:
        q += " AND r.account_nick=?"; params.append(nick)
    if has_metric:
        q += " AND r.metric_value IS NOT NULL"
    q += " ORDER BY r.id DESC LIMIT ?"; params.append(limit)
    with db.connect() as con:
        rows = con.execute(q, tuple(params)).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        try:
            d["params"] = json.loads(d.get("params") or "{}")
        except Exception:
            d["params"] = {}
        out.append(d)
    return out


def compare_experiments(
    *,
    competition: str | None = None,
    batch_id: str | None = None,
    notebook_id: int | None = None,
    descending: bool = True,
    limit: int = 100,
) -> dict[str, Any]:
    """Flatten experiments into a param-matrix sorted by metric.
    Returns {param_keys, rows} where each row has run_id/nick/metric + one col per param."""
    exps = list_experiments(
        competition=competition, batch_id=batch_id, notebook_id=notebook_id, limit=limit
    )
    param_keys: list[str] = []
    for e in exps:
        for k in e["params"]:
            if k not in param_keys:
                param_keys.append(k)
    rows: list[dict[str, Any]] = []
    for e in exps:
        row = {
            "run_id": e["id"],
            "nick": e["account_nick"],
            "notebook": e.get("notebook_title"),
            "competition": e.get("competition"),
            "status": e.get("status"),
            "metric_name": e.get("metric_name"),
            "metric_value": e.get("metric_value"),
            "lb_public": e.get("lb_public"),
            "lb_private": e.get("lb_private"),
            "runtime_s": e.get("runtime_seconds"),
            "batch_id": e.get("batch_id"),
        }
        for k in param_keys:
            row[f"param.{k}"] = e["params"].get(k)
        rows.append(row)
    # sort: experiments with a metric first, ordered by value
    rows.sort(
        key=lambda x: (x["metric_value"] is None, -(x["metric_value"] or 0) if descending else (x["metric_value"] or 0)),
    )
    best = next((r for r in rows if r["metric_value"] is not None), None)
    return {"n": len(rows), "param_keys": param_keys, "best": best, "rows": rows}


def set_run_metric(run_id: int, name: str, value: float) -> dict[str, Any]:
    with db.connect() as con:
        cur = con.execute(
            "UPDATE runs SET metric_name=?, metric_value=? WHERE id=?",
            (name, float(value), run_id),
        )
    if cur.rowcount == 0:
        return {"ok": False, "error": f"run {run_id} not found"}
    return {"ok": True, "run_id": run_id, "metric_name": name, "metric_value": float(value)}


def reextract_metrics(competition: str | None = None, only_missing: bool = True) -> dict[str, Any]:
    """Re-scan pulled logs and (re)fill metric_value for completed runs."""
    q = "SELECT id, output_path, metric_value, competition FROM runs WHERE output_path IS NOT NULL"
    params: list[Any] = []
    if competition:
        q += " AND competition=?"; params.append(competition)
    if only_missing:
        q += " AND metric_value IS NULL"
    with db.connect() as con:
        rows = con.execute(q, tuple(params)).fetchall()
    updated = 0
    for r in rows:
        mn, mv, _src = metrics.extract_from_dir(r["output_path"])
        if mv is not None:
            with db.connect() as con:
                con.execute("UPDATE runs SET metric_name=?, metric_value=? WHERE id=?",
                            (mn, mv, r["id"]))
            updated += 1
    return {"scanned": len(rows), "updated": updated}


# ---------- Competitions ----------

def sync_submissions_for_competition(competition: str, nicks: list[str] | None = None) -> dict[str, Any]:
    accounts = (
        [get_account(n) for n in nicks] if nicks else list_accounts()
    )
    accounts = [a for a in accounts if a]
    total_new = 0
    errors: list[str] = []

    leaderboard: list[dict[str, Any]] | None = None
    if accounts:
        try:
            leaderboard = kaggle_client.competition_leaderboard(accounts[0], competition)
        except Exception as exc:
            errors.append(f"leaderboard fetch failed: {exc}")
            leaderboard = None

    lb_size = len(leaderboard) if leaderboard else None
    score_to_rank: dict[str, int] = {}
    if leaderboard:
        for i, entry in enumerate(leaderboard, start=1):
            s = str(entry.get("score") or entry.get("publicScore") or "")
            if s and s not in score_to_rank:
                score_to_rank[s] = i

    for account in accounts:
        try:
            subs = kaggle_client.competition_submissions(account, competition)
        except Exception as exc:
            errors.append(f"{account.nick}: {exc}")
            continue
        for s in subs:
            submitted_at = s.get("date") or s.get("submittedAt") or s.get("submitted_at")
            file_name = s.get("fileName") or s.get("file_name")
            description = s.get("description") or s.get("message")
            public_score = s.get("publicScore") or s.get("public_score")
            private_score = s.get("privateScore") or s.get("private_score")
            status = s.get("status")
            rank_public = score_to_rank.get(str(public_score)) if public_score else None
            with db.connect() as con:
                cur = con.execute(
                    """INSERT INTO submissions
                    (competition, account_nick, file_name, description,
                     submitted_at, public_score, private_score, status,
                     rank_public, leaderboard_size, last_synced)
                    VALUES(?,?,?,?,?,?,?,?,?,?, datetime('now'))
                    ON CONFLICT(competition, account_nick, submitted_at, file_name)
                    DO UPDATE SET public_score=excluded.public_score,
                                  private_score=excluded.private_score,
                                  status=excluded.status,
                                  rank_public=excluded.rank_public,
                                  leaderboard_size=excluded.leaderboard_size,
                                  last_synced=datetime('now')""",
                    (
                        competition,
                        account.nick,
                        file_name,
                        description,
                        str(submitted_at) if submitted_at else None,
                        str(public_score) if public_score is not None else None,
                        str(private_score) if private_score is not None else None,
                        str(status) if status is not None else None,
                        rank_public,
                        lb_size,
                    ),
                )
                if cur.rowcount > 0:
                    total_new += 1
    linked = autolink_runs_to_submissions(competition)
    return {"new_or_updated": total_new, "errors": errors,
            "leaderboard_size": lb_size, "runs_linked": linked.get("linked", 0)}


# ---------- Link runs to the submission they produced (fills lb_public) ----------

def _parse_ts(s: str | None) -> datetime | None:
    """Tolerant timestamp parse → naive UTC datetime."""
    if not s:
        return None
    t = str(s).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(t.replace(" ", "T"))
        return dt.replace(tzinfo=None)
    except Exception:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(s).strip()[:19], fmt)
        except Exception:
            continue
    return None


def autolink_runs_to_submissions(competition: str | None = None, window_hours: float = 24.0) -> dict[str, Any]:
    """Heuristically attach the LB score of the submission a run produced.

    A run and a submission are separate objects on Kaggle, so this matches by
    (competition, nick) and time: the submission whose submitted_at is closest to
    the run's completed_at (within `window_hours`). If only one submission exists
    for that comp+nick, it's used directly. Best-effort — verify if it matters.
    Only fills runs whose lb_public is still empty (won't clobber manual values).
    """
    q = ("SELECT id, account_nick, competition, completed_at FROM runs "
         "WHERE competition IS NOT NULL AND lb_public IS NULL")
    params: list[Any] = []
    if competition:
        q += " AND competition=?"; params.append(competition)
    with db.connect() as con:
        runs = con.execute(q, tuple(params)).fetchall()
        subs = con.execute(
            "SELECT account_nick, competition, submitted_at, public_score, private_score "
            "FROM submissions WHERE public_score IS NOT NULL"
            + (" AND competition=?" if competition else ""),
            (competition,) if competition else (),
        ).fetchall()

    # index submissions by (comp, nick)
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for s in subs:
        by_key.setdefault((s["competition"], s["account_nick"]), []).append(dict(s))

    linked = 0
    for r in runs:
        cands = by_key.get((r["competition"], r["account_nick"]))
        if not cands:
            continue
        chosen: dict[str, Any] | None = None
        if len(cands) == 1:
            chosen = cands[0]
        else:
            rt = _parse_ts(r["completed_at"])
            if rt is None:
                continue
            best_dt = None
            for s in cands:
                st = _parse_ts(s["submitted_at"])
                if st is None:
                    continue
                gap = abs((st - rt).total_seconds())
                if gap <= window_hours * 3600 and (best_dt is None or gap < best_dt):
                    best_dt = gap
                    chosen = s
        if chosen is None:
            continue
        with db.connect() as con:
            con.execute(
                "UPDATE runs SET lb_public=?, lb_private=COALESCE(?, lb_private) WHERE id=?",
                (chosen.get("public_score"), chosen.get("private_score"), r["id"]),
            )
        linked += 1
    return {"linked": linked, "candidates": len(runs)}


def set_run_lb(run_id: int, public: str | None = None, private: str | None = None) -> dict[str, Any]:
    with db.connect() as con:
        cur = con.execute(
            "UPDATE runs SET lb_public=COALESCE(?, lb_public), lb_private=COALESCE(?, lb_private) WHERE id=?",
            (public, private, run_id),
        )
    if cur.rowcount == 0:
        return {"ok": False, "error": f"run {run_id} not found"}
    return {"ok": True, "run_id": run_id, "lb_public": public, "lb_private": private}


def list_submissions(competition: str | None = None) -> list[dict[str, Any]]:
    q = "SELECT * FROM submissions"
    params: tuple = ()
    if competition:
        q += " WHERE competition=?"
        params = (competition,)
    q += " ORDER BY id DESC LIMIT 1000"
    with db.connect() as con:
        rows = con.execute(q, params).fetchall()
    return [dict(r) for r in rows]


def list_tracked_competitions() -> list[str]:
    with db.connect() as con:
        rows = con.execute("SELECT DISTINCT competition FROM submissions ORDER BY competition").fetchall()
    return [r[0] for r in rows]


def submit_file_to_competition(
    nick: str, competition: str, file_path: str, message: str
) -> dict[str, Any]:
    account = get_account(nick)
    if not account:
        return {"ok": False, "error": "account not found"}
    p = Path(file_path)
    if not p.exists():
        return {"ok": False, "error": f"file not found: {file_path}"}
    try:
        resp = kaggle_client.competition_submit(account, competition, p, message)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "response": resp}


# ---------- helpers ----------

def _extract_int(d: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for k in keys:
        v = d.get(k)
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.isdigit():
            return int(v)
    return None
