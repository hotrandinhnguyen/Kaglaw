"""Tool registry for the agent.

Each tool = (JSONSchema for Claude/OpenAI) + (Python handler).

Tools are split into:
  - read-only (auto-execute)
  - destructive (require `confirm=True`; without it, return a preview dict)

Handlers return a JSON-serializable dict. The agent loop stringifies it
back into the conversation.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .. import accounts as account_mod
from .. import actions, budgets, db, excel_export, kaggle_client, memory_store, notebook_builder, orchestrator, runners
from ..config import DB_PATH, EXPORTS_DIR, NOTEBOOKS_DIR
from .llm import ToolSpec


@dataclass
class Tool:
    spec: ToolSpec
    handler: Callable[[dict[str, Any]], Any]
    destructive: bool = False


REGISTRY: dict[str, Tool] = {}


def register(name: str, description: str, schema: dict[str, Any], *, destructive: bool = False):
    def deco(fn: Callable[[dict[str, Any]], Any]) -> Callable[[dict[str, Any]], Any]:
        REGISTRY[name] = Tool(
            spec=ToolSpec(name=name, description=description, input_schema=schema),
            handler=fn,
            destructive=destructive,
        )
        return fn
    return deco


def _obj(props: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required or []}


def _confirm_gate(args: dict, preview: dict) -> dict | None:
    """Return preview dict if confirm not set; None to proceed."""
    if args.get("confirm") is True:
        return None
    return {
        "preview": True,
        "what_would_happen": preview,
        "next_step": "If the user agrees, call this tool again with the same args plus `confirm: true`.",
    }


# ============================================================
# READ-ONLY TOOLS
# ============================================================

@register(
    "list_accounts",
    "List all Kaggle accounts (nicks) configured in kaglaw with their usernames.",
    _obj({}),
)
def t_list_accounts(args):
    return [
        {"nick": a.nick, "username": a.username, "notes": a.notes}
        for a in account_mod.list_accounts()
    ]


@register(
    "list_notebooks",
    "List notebooks registered in kaglaw (id, title, language, flags).",
    _obj({}),
)
def t_list_notebooks(args):
    rows = actions.list_notebooks()
    out = []
    for r in rows:
        out.append({
            "id": r["id"],
            "title": r["title"],
            "language": r["language"],
            "kernel_type": r["kernel_type"],
            "enable_gpu": bool(r["enable_gpu"]),
            "enable_tpu": bool(r["enable_tpu"]),
            "local_path": r["local_path"],
        })
    return out


@register(
    "list_runs",
    "List recent notebook runs across all nicks. Filter by active_only or by nick.",
    _obj(
        {
            "active_only": {"type": "boolean", "default": False, "description": "Only queued/running"},
            "nick": {"type": "string", "description": "Filter by account nick"},
            "limit": {"type": "integer", "default": 50, "minimum": 1, "maximum": 500},
        }
    ),
)
def t_list_runs(args):
    runs = actions.list_runs(active_only=bool(args.get("active_only")), limit=int(args.get("limit", 50)))
    if nick := args.get("nick"):
        runs = [r for r in runs if r["account_nick"] == nick]
    return [
        {
            "id": r["id"], "notebook_title": r["notebook_title"], "nick": r["account_nick"],
            "slug": r["slug"], "version": r["version_number"], "status": r["status"],
            "pushed_at": r["pushed_at"], "completed_at": r["completed_at"],
            "runtime_seconds": r["runtime_seconds"], "used_gpu": bool(r["used_gpu"]),
            "error": (r["error"] or "")[:300] if r["error"] else None,
        }
        for r in runs
    ]


@register(
    "list_submissions",
    "List submissions stored in kaglaw. Filter by competition and/or nick.",
    _obj(
        {
            "competition": {"type": "string"},
            "nick": {"type": "string"},
            "limit": {"type": "integer", "default": 200},
        }
    ),
)
def t_list_submissions(args):
    subs = actions.list_submissions(args.get("competition"))
    if nick := args.get("nick"):
        subs = [s for s in subs if s["account_nick"] == nick]
    return subs[: int(args.get("limit", 200))]


@register(
    "list_tracked_competitions",
    "List all competitions currently tracked (i.e. have submissions in db).",
    _obj({}),
)
def t_list_tracked_competitions(args):
    return actions.list_tracked_competitions()


@register(
    "top_scores_per_nick",
    "For a competition, return the best public score each nick has achieved.",
    _obj({"competition": {"type": "string"}}, required=["competition"]),
)
def t_top_scores_per_nick(args):
    comp = args["competition"]
    with db.connect() as con:
        rows = con.execute(
            """SELECT account_nick, MAX(CAST(public_score AS REAL)) AS best_public
               FROM submissions WHERE competition=? AND public_score IS NOT NULL
               GROUP BY account_nick ORDER BY best_public DESC""",
            (comp,),
        ).fetchall()
    return [dict(r) for r in rows]


@register(
    "kaggle_search_competitions",
    "Search Kaggle for competitions by keyword.",
    _obj(
        {
            "query": {"type": "string"},
            "as_nick": {"type": "string", "description": "Which nick's credentials to use; defaults to first available"},
            "limit": {"type": "integer", "default": 20},
        },
        required=["query"],
    ),
)
def t_kaggle_search_competitions(args):
    account = _pick_account(args.get("as_nick"))
    if not account:
        return {"error": "no accounts configured"}
    items = kaggle_client.competitions_list(account, search=args["query"])
    return items[: int(args.get("limit", 20))]


@register(
    "kaggle_get_leaderboard",
    "Pull the public leaderboard for a Kaggle competition. Returns top N rows (team, score, rank).",
    _obj(
        {
            "competition": {"type": "string"},
            "as_nick": {"type": "string"},
            "top_n": {"type": "integer", "default": 30},
        },
        required=["competition"],
    ),
)
def t_kaggle_get_leaderboard(args):
    account = _pick_account(args.get("as_nick"))
    if not account:
        return {"error": "no accounts configured"}
    lb = kaggle_client.competition_leaderboard(account, args["competition"])
    return lb[: int(args.get("top_n", 30))]


@register(
    "kaggle_search_kernels",
    "Search Kaggle public kernels (notebooks). Useful to study top approaches in a comp.",
    _obj(
        {
            "query": {"type": "string"},
            "competition": {"type": "string", "description": "Optional comp slug filter"},
            "as_nick": {"type": "string"},
            "sort_by": {"type": "string", "enum": ["hotness", "votes", "scoreDescending", "dateRun"], "default": "hotness"},
            "limit": {"type": "integer", "default": 20},
        },
        required=["query"],
    ),
)
def t_kaggle_search_kernels(args):
    account = _pick_account(args.get("as_nick"))
    if not account:
        return {"error": "no accounts configured"}
    from .. import kaggle_client as kc
    with kc.as_account(account) as api:
        try:
            kw = {"search": args["query"], "sort_by": args.get("sort_by", "hotness")}
            if comp := args.get("competition"):
                kw["competition"] = comp
            resp = api.kernels_list(**kw)
        except Exception as exc:
            return {"error": str(exc)}
        items = [kc._resp_to_dict(x) for x in resp]
    return items[: int(args.get("limit", 20))]


@register(
    "kaggle_search_datasets",
    "Search Kaggle datasets.",
    _obj(
        {
            "query": {"type": "string"},
            "as_nick": {"type": "string"},
            "limit": {"type": "integer", "default": 20},
        },
        required=["query"],
    ),
)
def t_kaggle_search_datasets(args):
    account = _pick_account(args.get("as_nick"))
    if not account:
        return {"error": "no accounts configured"}
    from .. import kaggle_client as kc
    with kc.as_account(account) as api:
        resp = api.dataset_list(search=args["query"])
        items = [kc._resp_to_dict(x) for x in resp]
    return items[: int(args.get("limit", 20))]


@register(
    "db_query",
    "Run a READ-ONLY SQL query on the kaglaw SQLite db. Tables: accounts, notebooks, "
    "runs, submissions. Use this for custom aggregates / reports. Only SELECT/WITH is allowed.",
    _obj(
        {"sql": {"type": "string"}, "limit": {"type": "integer", "default": 200}},
        required=["sql"],
    ),
)
def t_db_query(args):
    sql = args["sql"].strip().rstrip(";")
    low = sql.lstrip().lower()
    if not (low.startswith("select") or low.startswith("with")):
        return {"error": "Only SELECT/WITH allowed in db_query."}
    forbidden = (
        "insert ", "update ", "delete ", "drop ", "alter ", "create ",
        "attach ", "detach ", "pragma ", "vacuum",
    )
    if any(f in low for f in forbidden):
        return {"error": "Forbidden keyword detected. Use SELECT/WITH only."}
    limit = int(args.get("limit", 200))
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        cur = con.execute(sql)
        rows = cur.fetchmany(limit)
        return {"columns": [d[0] for d in cur.description or []], "rows": [dict(r) for r in rows]}
    except Exception as exc:
        return {"error": str(exc)}
    finally:
        con.close()


@register(
    "export_excel_report",
    "Build the standard 8-sheet Excel report (Summary, Experiments, Notebooks, Runs, Submissions, "
    "GPU Usage, Budgets, Outputs & Logs) and return the local path. Also downloadable at /export.xlsx.",
    _obj({}),
)
def t_export_excel_report(args):
    path = excel_export.export_to_file()
    return {"path": str(path), "filename": path.name}


@register(
    "build_custom_excel",
    "Build a custom Excel from one or more SELECT queries. Each query becomes a sheet.",
    _obj(
        {
            "filename": {"type": "string", "description": "Output filename (without dir)."},
            "sheets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "sheet_name": {"type": "string"},
                        "sql": {"type": "string"},
                    },
                    "required": ["sheet_name", "sql"],
                },
            },
        },
        required=["sheets"],
    ),
)
def t_build_custom_excel(args):
    from openpyxl import Workbook

    wb = Workbook()
    default = wb.active
    wb.remove(default)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        for sh in args["sheets"]:
            sql = sh["sql"].strip().rstrip(";")
            low = sql.lstrip().lower()
            if not (low.startswith("select") or low.startswith("with")):
                return {"error": f"Sheet {sh['sheet_name']}: only SELECT/WITH allowed."}
            cur = con.execute(sql)
            cols = [d[0] for d in cur.description or []]
            rows = cur.fetchall()
            ws = wb.create_sheet(sh["sheet_name"][:31] or "Sheet")
            ws.append(cols)
            for r in rows:
                ws.append([r[c] for c in cols])
    finally:
        con.close()
    fname = args.get("filename") or "kaglaw_custom.xlsx"
    if not fname.endswith(".xlsx"):
        fname += ".xlsx"
    path = EXPORTS_DIR / fname
    wb.save(path)
    return {"path": str(path), "filename": fname}


@register(
    "summarize_competition_progress",
    "Build a Vietnamese-friendly summary of progress on a competition across nicks: "
    "best score per nick, # submissions per nick, days remaining info if available.",
    _obj({"competition": {"type": "string"}}, required=["competition"]),
)
def t_summarize_competition_progress(args):
    comp = args["competition"]
    with db.connect() as con:
        per_nick = con.execute(
            """SELECT account_nick, COUNT(*) AS n_subs,
                      MAX(CAST(public_score AS REAL)) AS best_public,
                      MAX(submitted_at) AS last_sub_at
               FROM submissions WHERE competition=? GROUP BY account_nick
               ORDER BY best_public DESC""",
            (comp,),
        ).fetchall()
        lb_size = con.execute(
            "SELECT MAX(leaderboard_size) FROM submissions WHERE competition=?",
            (comp,),
        ).fetchone()[0]
    return {
        "competition": comp,
        "per_nick": [dict(r) for r in per_nick],
        "leaderboard_size": lb_size,
    }


# ----- Read kernel source / competition details / local files / web -----

@register(
    "kaggle_get_kernel_source",
    "Download the source of a public Kaggle kernel (notebook). Returns the code as text. "
    "Useful to study top approaches in a competition.",
    _obj(
        {
            "kernel_ref": {"type": "string", "description": "Format <user>/<slug>"},
            "as_nick": {"type": "string"},
            "max_chars": {"type": "integer", "default": 12000},
        },
        required=["kernel_ref"],
    ),
)
def t_kaggle_get_kernel_source(args):
    account = _pick_account(args.get("as_nick"))
    if not account:
        return {"error": "no accounts configured"}
    import tempfile
    from .. import kaggle_client as kc
    with tempfile.TemporaryDirectory(prefix="kaglaw_pull_") as td:
        try:
            with kc.as_account(account) as api:
                api.kernels_pull(args["kernel_ref"], td, metadata=True, quiet=True)
        except Exception as exc:
            return {"error": str(exc)}
        src = None
        for ext in (".ipynb", ".py", ".r", ".R", ".Rmd"):
            files = list(Path(td).glob(f"*{ext}"))
            if files:
                src = files[0]
                break
        if not src:
            return {"error": "no source file pulled"}
        text = src.read_text(encoding="utf-8", errors="replace")
        mx = int(args.get("max_chars", 12000))
        return {
            "kernel_ref": args["kernel_ref"],
            "file_name": src.name,
            "size_chars": len(text),
            "truncated": len(text) > mx,
            "source": text[:mx],
        }


@register(
    "kaggle_competition_info",
    "Get description/evaluation/deadline of a Kaggle competition by slug.",
    _obj(
        {"competition": {"type": "string"}, "as_nick": {"type": "string"}},
        required=["competition"],
    ),
)
def t_kaggle_competition_info(args):
    account = _pick_account(args.get("as_nick"))
    if not account:
        return {"error": "no accounts configured"}
    items = kaggle_client.competitions_list(account, search=args["competition"])
    for it in items:
        ref = it.get("ref") or it.get("Ref") or it.get("id")
        if ref and ref.lower().endswith(args["competition"].lower()):
            return it
    return items[0] if items else {"error": "not found"}


@register(
    "kaggle_competition_data_files",
    "List the data files of a Kaggle competition (so you know what's in train/test).",
    _obj(
        {"competition": {"type": "string"}, "as_nick": {"type": "string"}},
        required=["competition"],
    ),
)
def t_kaggle_competition_data_files(args):
    account = _pick_account(args.get("as_nick"))
    if not account:
        return {"error": "no accounts configured"}
    from .. import kaggle_client as kc
    with kc.as_account(account) as api:
        try:
            resp = api.competitions_data_list_files(args["competition"])
        except Exception as exc:
            return {"error": str(exc)}
        return [kc._resp_to_dict(f) for f in resp]


@register(
    "read_local_file",
    "Read a text file from the local machine. Useful for inspecting submission CSV, "
    "notebook source, etc. Limit ~5000 lines / 200KB.",
    _obj(
        {
            "path": {"type": "string"},
            "max_bytes": {"type": "integer", "default": 200000},
            "head_lines": {"type": "integer", "description": "If set, only return first N lines"},
        },
        required=["path"],
    ),
)
def t_read_local_file(args):
    p = Path(args["path"])
    if not p.exists():
        return {"error": f"not found: {p}"}
    if not p.is_file():
        return {"error": f"not a file: {p}"}
    size = p.stat().st_size
    mx = int(args.get("max_bytes", 200000))
    if args.get("head_lines"):
        n = int(args["head_lines"])
        with p.open("r", encoding="utf-8", errors="replace") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= n:
                    break
                lines.append(line)
        return {"path": str(p), "size_bytes": size, "lines_read": len(lines), "content": "".join(lines)}
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return {"error": str(exc)}
    return {
        "path": str(p),
        "size_bytes": size,
        "truncated": size > mx,
        "content": text[:mx],
    }


@register(
    "list_local_dir",
    "List entries in a local directory.",
    _obj({"path": {"type": "string"}}, required=["path"]),
)
def t_list_local_dir(args):
    p = Path(args["path"])
    if not p.exists():
        return {"error": f"not found: {p}"}
    if not p.is_dir():
        return {"error": f"not a directory: {p}"}
    out = []
    for child in sorted(p.iterdir()):
        try:
            stat = child.stat()
        except Exception:
            continue
        out.append({
            "name": child.name,
            "is_dir": child.is_dir(),
            "size": stat.st_size if child.is_file() else None,
        })
    return {"path": str(p), "entries": out}


@register(
    "web_search",
    "Search the public web via DuckDuckGo HTML. Returns list of {title, href, snippet}. "
    "Use sparingly for general research (e.g. find a paper, blog post, SOTA result).",
    _obj(
        {"query": {"type": "string"}, "max_results": {"type": "integer", "default": 8}},
        required=["query"],
    ),
)
def t_web_search(args):
    import httpx
    from html.parser import HTMLParser

    q = args["query"]
    mx = int(args.get("max_results", 8))
    try:
        r = httpx.get(
            "https://duckduckgo.com/html/",
            params={"q": q},
            headers={"User-Agent": "Mozilla/5.0 (kaglaw)"},
            timeout=15.0,
            follow_redirects=True,
        )
        r.raise_for_status()
    except Exception as exc:
        return {"error": str(exc)}

    class P(HTMLParser):
        def __init__(self):
            super().__init__()
            self.results = []
            self._cur: dict = {}
            self._field: str | None = None
            self._buf: list[str] = []

        def handle_starttag(self, tag, attrs):
            a = dict(attrs)
            if tag == "a" and "result__a" in (a.get("class") or ""):
                if self._cur:
                    self.results.append(self._cur)
                self._cur = {"href": a.get("href", "").strip()}
                self._field = "title"
                self._buf = []
            elif tag == "a" and "result__snippet" in (a.get("class") or ""):
                self._field = "snippet"
                self._buf = []

        def handle_endtag(self, tag):
            if tag == "a" and self._field:
                self._cur[self._field] = "".join(self._buf).strip()
                self._field = None
                self._buf = []

        def handle_data(self, data):
            if self._field:
                self._buf.append(data)

    parser = P()
    parser.feed(r.text)
    if parser._cur:
        parser.results.append(parser._cur)
    cleaned = []
    for it in parser.results[:mx]:
        href = it.get("href", "")
        # DuckDuckGo wraps results: //duckduckgo.com/l/?uddg=<encoded_url>
        if "uddg=" in href:
            from urllib.parse import unquote, parse_qs, urlparse
            qs = parse_qs(urlparse(href).query)
            href = unquote(qs.get("uddg", [href])[0])
        cleaned.append({"title": it.get("title", ""), "url": href, "snippet": it.get("snippet", "")})
    return cleaned


# ============================================================
# NOTEBOOK AUTHORING + RESULTS (local-only, auto-execute)
# ============================================================

@register(
    "create_notebook",
    "Author a NEW Kaggle code file from source you write yourself, then register it so it "
    "can be pushed later. This is how you create notebooks from the chat. `code` is plain "
    "Python source; separate logical steps with a line `# %%` to make notebook cells, and "
    "`# %% [markdown]` for a markdown cell (its lines may start with '# '). Returns notebook_id, "
    "path and cell count. Local-only (nothing hits Kaggle until you push).",
    _obj(
        {
            "title": {"type": "string", "description": "Human title; also used to slug the file/folder."},
            "code": {"type": "string", "description": "The full source. Use '# %%' to split cells."},
            "language": {"type": "string", "enum": ["python", "r"], "default": "python"},
            "kernel_type": {"type": "string", "enum": ["notebook", "script"], "default": "notebook"},
            "enable_gpu": {"type": "boolean", "default": False},
            "enable_tpu": {"type": "boolean", "default": False},
            "enable_internet": {"type": "boolean", "default": True},
            "dataset_sources": {"type": "array", "items": {"type": "string"}, "description": "user/dataset-slug refs"},
            "competition_sources": {"type": "array", "items": {"type": "string"}, "description": "competition slugs"},
            "kernel_sources": {"type": "array", "items": {"type": "string"}},
        },
        required=["title", "code"],
    ),
)
def t_create_notebook(args):
    title = args["title"]
    kernel_type = args.get("kernel_type", "notebook")
    language = args.get("language", "python")
    try:
        path = notebook_builder.write_source(
            title, args["code"], language=language, kernel_type=kernel_type
        )
    except Exception as exc:
        return {"error": f"write failed: {exc}"}
    spec = actions.NotebookSpec(
        title=title,
        local_path=str(path),
        language=language,
        kernel_type=kernel_type,
        enable_gpu=bool(args.get("enable_gpu")),
        enable_tpu=bool(args.get("enable_tpu")),
        enable_internet=bool(args.get("enable_internet", True)),
        dataset_sources=args.get("dataset_sources") or [],
        competition_sources=args.get("competition_sources") or [],
        kernel_sources=args.get("kernel_sources") or [],
    )
    nb_id = actions.register_notebook(spec)
    n_cells = len(notebook_builder.split_into_cells(args["code"]))
    return {
        "notebook_id": nb_id,
        "title": title,
        "path": str(path),
        "kernel_type": kernel_type,
        "cells": n_cells,
        "next_step": "Push it with push_notebook_to_accounts(notebook_id, nicks=[...]) after the user confirms.",
    }


@register(
    "get_notebook_code",
    "Read back the current source of a registered notebook as '# %%'-delimited text, so you "
    "can review or edit it. Pass notebook_id.",
    _obj(
        {"notebook_id": {"type": "integer"}, "max_chars": {"type": "integer", "default": 20000}},
        required=["notebook_id"],
    ),
)
def t_get_notebook_code(args):
    nb = actions.get_notebook(int(args["notebook_id"]))
    if not nb:
        return {"error": f"notebook {args['notebook_id']} not found"}
    try:
        src = notebook_builder.read_source(nb["local_path"])
    except Exception as exc:
        return {"error": str(exc)}
    mx = int(args.get("max_chars", 20000))
    return {
        "notebook_id": nb["id"],
        "title": nb["title"],
        "language": nb["language"],
        "kernel_type": nb["kernel_type"],
        "size_chars": len(src),
        "truncated": len(src) > mx,
        "code": src[:mx],
    }


@register(
    "update_notebook_code",
    "Edit a registered notebook's source. Either pass `new_code` to replace the whole file, "
    "or `replacements` (a map of literal old->new strings) to patch specific lines. A .bak of "
    "the previous version is kept. Local-only.",
    _obj(
        {
            "notebook_id": {"type": "integer"},
            "new_code": {"type": "string", "description": "Full replacement source ('# %%' to split cells)."},
            "replacements": {
                "type": "object",
                "description": "Literal old-string -> new-string edits applied to the source.",
                "additionalProperties": {"type": "string"},
            },
        },
        required=["notebook_id"],
    ),
)
def t_update_notebook_code(args):
    nb = actions.get_notebook(int(args["notebook_id"]))
    if not nb:
        return {"error": f"notebook {args['notebook_id']} not found"}
    if not args.get("new_code") and not args.get("replacements"):
        return {"error": "provide either new_code or replacements"}
    try:
        current = notebook_builder.read_source(nb["local_path"])
    except Exception as exc:
        return {"error": str(exc)}

    counts: dict[str, int] = {}
    if args.get("new_code"):
        new_source = args["new_code"]
    else:
        new_source = current
        for old, new in (args.get("replacements") or {}).items():
            c = new_source.count(old)
            counts[old] = c
            if c:
                new_source = new_source.replace(old, new)
        missing = [k for k, v in counts.items() if v == 0]
        if missing:
            return {"error": f"replacement pattern(s) not found in source: {missing}", "counts": counts}

    # keep a backup of the actual file
    try:
        p = Path(nb["local_path"])
        if p.is_file():
            shutil.copy2(p, p.with_suffix(p.suffix + ".bak"))
        new_path = notebook_builder.overwrite_source(nb["local_path"], new_source)
    except Exception as exc:
        return {"error": f"write failed: {exc}"}
    return {
        "notebook_id": nb["id"],
        "path": str(new_path),
        "mode": "new_code" if args.get("new_code") else "replacements",
        "replacements_applied": counts or None,
        "new_size_chars": len(new_source),
    }


@register(
    "get_run_output",
    "Fetch the result of a run: its status/error plus the files kaglaw pulled when it "
    "completed (kernel log, submission.csv, etc.). Pass a `file` name to read that file's "
    "head. If output isn't downloaded yet, call sync_run_status first.",
    _obj(
        {
            "run_id": {"type": "integer"},
            "file": {"type": "string", "description": "Optional: name of a pulled output file to read."},
            "max_bytes": {"type": "integer", "default": 8000},
        },
        required=["run_id"],
    ),
)
def t_get_run_output(args):
    run_id = int(args["run_id"])
    with db.connect() as con:
        r = con.execute(
            "SELECT r.*, n.title AS notebook_title FROM runs r "
            "LEFT JOIN notebooks n ON n.id=r.notebook_id WHERE r.id=?",
            (run_id,),
        ).fetchone()
    if not r:
        return {"error": f"run {run_id} not found"}
    out: dict[str, Any] = {
        "run_id": run_id,
        "notebook_title": r["notebook_title"],
        "nick": r["account_nick"],
        "slug": r["slug"],
        "status": r["status"],
        "error": r["error"],
        "runtime_seconds": r["runtime_seconds"],
        "output_path": r["output_path"],
        "log_summary": r["log_summary"],
    }
    odir = r["output_path"]
    if not odir or not Path(odir).exists():
        out["note"] = "No output downloaded yet. Run sync_run_status(run_id) once the kernel completes."
        return out
    files = [p for p in sorted(Path(odir).iterdir()) if p.is_file()]
    out["files"] = [{"name": p.name, "size": p.stat().st_size} for p in files]
    if args.get("file"):
        target = Path(odir) / args["file"]
        if not target.exists():
            return {**out, "error": f"file not found in output: {args['file']}"}
        mx = int(args.get("max_bytes", 8000))
        text = target.read_text(encoding="utf-8", errors="replace")
        out["file_read"] = args["file"]
        out["content"] = text[:mx]
        out["content_truncated"] = len(text) > mx
    return out


@register(
    "test_notebook_local",
    "Smoke-test a notebook by RUNNING IT LOCALLY (subprocess + timeout) before pushing to "
    "Kaggle — catches syntax/import/logic errors cheaply. Runs on this machine, so deps must "
    "be installed and /kaggle/input paths may not exist. Returns rc, stdout/stderr tail, any "
    "parsed metric, and files produced. Do this before push to save GPU quota.",
    _obj(
        {
            "notebook_id": {"type": "integer"},
            "timeout": {"type": "integer", "description": "Seconds before it's killed (default 300)."},
            "keep_workdir": {"type": "boolean", "default": False},
        },
        required=["notebook_id"],
    ),
)
def t_test_notebook_local(args):
    return runners.run_notebook_local(
        int(args["notebook_id"]),
        timeout=int(args["timeout"]) if args.get("timeout") is not None else None,
        keep_workdir=bool(args.get("keep_workdir")),
    )


@register(
    "import_kernel",
    "Pull an EXISTING Kaggle kernel (yours or any public one) into kaglaw as a registered "
    "notebook — source + settings — so you can edit it and push a new version. kernel_ref = "
    "'<user>/<slug>'. To push back as a new version of the SAME kernel, push under the owner "
    "nick and keep the title so the slug matches.",
    _obj(
        {
            "kernel_ref": {"type": "string", "description": "Format <user>/<slug>"},
            "as_nick": {"type": "string", "description": "Whose credentials to pull with (default first)."},
            "title": {"type": "string", "description": "Override the registered title."},
        },
        required=["kernel_ref"],
    ),
)
def t_import_kernel(args):
    return actions.import_kernel(args["kernel_ref"], as_nick=args.get("as_nick"), title=args.get("title"))


# ============================================================
# EXPERIMENTS (runs enriched with params + parsed metric)
# ============================================================

@register(
    "list_experiments",
    "List runs as experiment records: params, parsed metric (CV/AUC/score from log), status, "
    "lb score, batch. Filter by competition / batch_id / notebook_id / nick. Use has_metric=true "
    "to only show runs whose metric was parsed.",
    _obj(
        {
            "competition": {"type": "string"},
            "batch_id": {"type": "string"},
            "notebook_id": {"type": "integer"},
            "nick": {"type": "string"},
            "has_metric": {"type": "boolean", "default": False},
            "limit": {"type": "integer", "default": 100},
        }
    ),
)
def t_list_experiments(args):
    exps = actions.list_experiments(
        competition=args.get("competition"),
        batch_id=args.get("batch_id"),
        notebook_id=args.get("notebook_id"),
        nick=args.get("nick"),
        has_metric=bool(args.get("has_metric")),
        limit=int(args.get("limit", 100)),
    )
    # trim noisy fields for the model
    keep = ("id", "notebook_title", "account_nick", "competition", "params", "status",
            "metric_name", "metric_value", "lb_public", "runtime_seconds", "batch_id",
            "tags", "version_number", "code_snapshot_path")
    return [{k: e.get(k) for k in keep} for e in exps]


@register(
    "compare_experiments",
    "Build a comparison matrix of experiments (one column per param + the metric), sorted by "
    "metric. Filter by competition / batch_id / notebook_id. Returns best + rows. This is the "
    "main research view to see which params produced the best score.",
    _obj(
        {
            "competition": {"type": "string"},
            "batch_id": {"type": "string"},
            "notebook_id": {"type": "integer"},
            "descending": {"type": "boolean", "default": True, "description": "True = higher metric is better (AUC/acc); False for loss/RMSE."},
            "limit": {"type": "integer", "default": 100},
        }
    ),
)
def t_compare_experiments(args):
    return actions.compare_experiments(
        competition=args.get("competition"),
        batch_id=args.get("batch_id"),
        notebook_id=args.get("notebook_id"),
        descending=bool(args.get("descending", True)),
        limit=int(args.get("limit", 100)),
    )


@register(
    "set_run_metric",
    "Manually set/override the research metric of a run (e.g. after reading the log yourself). ",
    _obj(
        {
            "run_id": {"type": "integer"},
            "name": {"type": "string", "description": "Metric name e.g. 'cv', 'auc'."},
            "value": {"type": "number"},
        },
        required=["run_id", "name", "value"],
    ),
)
def t_set_run_metric(args):
    return actions.set_run_metric(int(args["run_id"]), str(args["name"]), float(args["value"]))


@register(
    "autolink_run_scores",
    "Attach the leaderboard score (lb_public/lb_private) to runs by matching them to the "
    "submission they produced — by competition+nick and closest time. Best-effort/heuristic. "
    "Runs automatically during submission sync; call manually to refresh.",
    _obj(
        {
            "competition": {"type": "string"},
            "window_hours": {"type": "number", "default": 24, "description": "Max gap run↔submission to link."},
        }
    ),
)
def t_autolink_run_scores(args):
    return actions.autolink_runs_to_submissions(
        competition=args.get("competition"),
        window_hours=float(args.get("window_hours", 24)),
    )


@register(
    "set_run_lb",
    "Manually set a run's leaderboard score (when autolink can't, or to correct it).",
    _obj(
        {
            "run_id": {"type": "integer"},
            "public": {"type": "string", "description": "Public LB score."},
            "private": {"type": "string", "description": "Private LB score."},
        },
        required=["run_id"],
    ),
)
def t_set_run_lb(args):
    return actions.set_run_lb(int(args["run_id"]), args.get("public"), args.get("private"))


@register(
    "reextract_metrics",
    "Re-scan already-pulled run logs and (re)fill the metric column. Useful after changing the "
    "metric regex patterns or to backfill old runs. only_missing=false re-parses everything.",
    _obj(
        {
            "competition": {"type": "string"},
            "only_missing": {"type": "boolean", "default": True},
        }
    ),
)
def t_reextract_metrics(args):
    return actions.reextract_metrics(
        competition=args.get("competition"), only_missing=bool(args.get("only_missing", True))
    )


# ============================================================
# ORCHESTRATION: batches, queue, budgets (read/light actions)
# ============================================================

@register(
    "batch_status",
    "Progress of a sweep/push batch: per-status job counts, best metric so far, and the "
    "experiment matrix for that batch. Pass batch_id (returned by launch_sweep / push tools).",
    _obj({"batch_id": {"type": "string"}}, required=["batch_id"]),
)
def t_batch_status(args):
    return orchestrator.batch_status(args["batch_id"])


@register(
    "list_batches",
    "List recent batches (sweeps and multi-nick pushes) with job/run counts + best metric.",
    _obj({"limit": {"type": "integer", "default": 30}}),
)
def t_list_batches(args):
    return orchestrator.list_batches(limit=int(args.get("limit", 30)))


@register(
    "cancel_batch",
    "Cancel the still-queued jobs of a batch (already-launched runs keep going on Kaggle).",
    _obj({"batch_id": {"type": "string"}}, required=["batch_id"]),
)
def t_cancel_batch(args):
    return orchestrator.cancel_batch(args["batch_id"])


@register(
    "dispatch_jobs_now",
    "Manually nudge the queue: launch as many queued jobs as free per-nick slots allow "
    "(the scheduler also does this every ~30s). Returns how many were dispatched.",
    _obj({"max_dispatch": {"type": "integer", "description": "Cap launches this call."}}),
)
def t_dispatch_jobs_now(args):
    md = args.get("max_dispatch")
    return orchestrator.dispatch_jobs(max_dispatch=int(md) if md is not None else None)


@register(
    "nick_budget",
    "Estimated quota for one nick: running kernels, GPU hours used in last 7d, remaining GPU "
    "budget, submissions today. (Kaggle has no quota API — these are estimates.)",
    _obj({"nick": {"type": "string"}}, required=["nick"]),
)
def t_nick_budget(args):
    return budgets.nick_usage(args["nick"])


@register(
    "list_budgets",
    "Estimated quota across all nicks (or a given subset): free slots + remaining GPU hours. "
    "Use this to decide where there's capacity to run.",
    _obj({"nicks": {"type": "array", "items": {"type": "string"}}}),
)
def t_list_budgets(args):
    nicks = args.get("nicks") or [a.nick for a in account_mod.list_accounts()]
    return budgets.all_usages(nicks)


@register(
    "nick_status",
    "Live status per nick: quota estimate (free slots, GPU hours remaining, submits today) PLUS "
    "exactly WHICH runs are active right now (notebook, slug, status, competition, batch) and how "
    "many queued jobs are waiting for that nick. Best view for 'who is running what'.",
    _obj({"nicks": {"type": "array", "items": {"type": "string"}}}),
)
def t_nick_status(args):
    return orchestrator.nick_status(args.get("nicks"))


@register(
    "list_notifications",
    "Recent in-app notifications (e.g. a sweep batch finished). unseen_only=true for new ones.",
    _obj({"unseen_only": {"type": "boolean", "default": False}, "limit": {"type": "integer", "default": 20}}),
)
def t_list_notifications(args):
    return orchestrator.list_notifications(
        unseen_only=bool(args.get("unseen_only")), limit=int(args.get("limit", 20))
    )


# ============================================================
# LONG-TERM MEMORY (persists across ALL chats)
# ============================================================

@register(
    "remember",
    "Save a durable fact/preference into long-term memory so it's recalled in EVERY future "
    "chat (auto-injected into your prompt). Use when the user states a lasting preference or "
    "convention, e.g. 'luôn dùng nick main', 'metric là AUC', '5-fold CV', 'GPU mặc định bật'. "
    "Keep each memory short and atomic.",
    _obj(
        {
            "text": {"type": "string"},
            "kind": {"type": "string", "enum": ["preference", "fact", "note"], "default": "preference"},
            "tags": {"type": "string"},
        },
        required=["text"],
    ),
)
def t_remember(args):
    try:
        mid = memory_store.add_memory(args["text"], kind=args.get("kind", "preference"), tags=args.get("tags"))
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    return {"ok": True, "memory_id": mid, "text": args["text"]}


@register(
    "recall",
    "List or search long-term memories (the facts/preferences you've saved). Optional `query` "
    "filters by substring. These are already injected into your prompt — use this to show the "
    "user what's stored or before editing.",
    _obj({"query": {"type": "string"}, "kind": {"type": "string"}, "limit": {"type": "integer", "default": 50}}),
)
def t_recall(args):
    if args.get("query"):
        return memory_store.search_memories(args["query"], limit=int(args.get("limit", 50)))
    return memory_store.list_memories(kind=args.get("kind"), limit=int(args.get("limit", 50)))


@register(
    "forget",
    "Delete a long-term memory by id. Ask the user before forgetting unless they clearly asked.",
    _obj({"memory_id": {"type": "integer"}}, required=["memory_id"]),
)
def t_forget(args):
    return memory_store.delete_memory(int(args["memory_id"]))


@register(
    "update_memory",
    "Edit an existing long-term memory (text/kind/pinned).",
    _obj(
        {
            "memory_id": {"type": "integer"},
            "text": {"type": "string"},
            "kind": {"type": "string", "enum": ["preference", "fact", "note"]},
            "pinned": {"type": "boolean"},
        },
        required=["memory_id"],
    ),
)
def t_update_memory(args):
    return memory_store.update_memory(
        int(args["memory_id"]), text=args.get("text"), kind=args.get("kind"),
        pinned=args.get("pinned"),
    )


# ============================================================
# DESTRUCTIVE TOOLS (need confirm=True)
# ============================================================

@register(
    "register_notebook",
    "Register a notebook from a local file so it can be pushed to nicks later. "
    "Returns the notebook id.",
    _obj(
        {
            "title": {"type": "string"},
            "local_path": {"type": "string", "description": "Absolute path to .ipynb / .py on this machine"},
            "language": {"type": "string", "default": "python"},
            "kernel_type": {"type": "string", "enum": ["notebook", "script"], "default": "notebook"},
            "enable_gpu": {"type": "boolean", "default": False},
            "enable_tpu": {"type": "boolean", "default": False},
            "enable_internet": {"type": "boolean", "default": True},
            "dataset_sources": {"type": "array", "items": {"type": "string"}},
            "competition_sources": {"type": "array", "items": {"type": "string"}},
            "kernel_sources": {"type": "array", "items": {"type": "string"}},
            "confirm": {"type": "boolean", "default": False},
        },
        required=["title", "local_path"],
    ),
    destructive=True,
)
def t_register_notebook(args):
    preview = {
        "action": "register_notebook",
        "title": args["title"],
        "local_path": args["local_path"],
        "gpu": bool(args.get("enable_gpu")),
        "tpu": bool(args.get("enable_tpu")),
    }
    if (gate := _confirm_gate(args, preview)) is not None:
        return gate
    p = Path(args["local_path"])
    if not p.exists():
        return {"error": f"path not found: {p}"}
    # Stage the file into data/notebooks/<slug>/
    safe_dir = NOTEBOOKS_DIR / actions.slugify(args["title"])
    safe_dir.mkdir(parents=True, exist_ok=True)
    target = safe_dir / p.name
    shutil.copy2(p, target)
    spec = actions.NotebookSpec(
        title=args["title"],
        local_path=str(target),
        language=args.get("language", "python"),
        kernel_type=args.get("kernel_type", "notebook"),
        enable_gpu=bool(args.get("enable_gpu")),
        enable_tpu=bool(args.get("enable_tpu")),
        enable_internet=bool(args.get("enable_internet", True)),
        dataset_sources=args.get("dataset_sources") or [],
        competition_sources=args.get("competition_sources") or [],
        kernel_sources=args.get("kernel_sources") or [],
    )
    return {"notebook_id": actions.register_notebook(spec)}


@register(
    "push_notebook_to_accounts",
    "Push a registered notebook to one or more Kaggle accounts. Destructive: triggers actual "
    "pushes that consume GPU quota when the kernel runs. ALWAYS ask the user before calling "
    "with confirm=true.",
    _obj(
        {
            "notebook_id": {"type": "integer"},
            "nicks": {"type": "array", "items": {"type": "string"}},
            "version_notes": {"type": "string"},
            "competition": {"type": "string", "description": "Tag these runs as experiments on this comp (defaults to the notebook's first competition_source)."},
            "tags": {"type": "string", "description": "Free-text tags for the experiment, e.g. 'baseline,lgbm'."},
            "confirm": {"type": "boolean", "default": False},
        },
        required=["notebook_id", "nicks"],
    ),
    destructive=True,
)
def t_push_notebook(args):
    preview = {
        "action": "push_notebook_to_accounts",
        "notebook_id": args["notebook_id"],
        "nicks": args["nicks"],
        "version_notes": args.get("version_notes", ""),
        "competition": args.get("competition"),
    }
    if (gate := _confirm_gate(args, preview)) is not None:
        return gate
    return actions.push_notebook_to_accounts(
        int(args["notebook_id"]),
        list(args["nicks"]),
        args.get("version_notes", ""),
        competition=args.get("competition"),
        tags=args.get("tags"),
    )


@register(
    "variant_push",
    "Push the SAME notebook to multiple nicks with per-nick literal-string replacements "
    "(parameter sweep). Each variant: {nick, replacements: {old_str: new_str}, slug_suffix?}. "
    "Each replacement pattern must occur in the source or the variant is skipped. "
    "Destructive — confirm required.",
    _obj(
        {
            "notebook_id": {"type": "integer"},
            "variants": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "nick": {"type": "string"},
                        "replacements": {
                            "type": "object",
                            "description": "Map of literal old string → new string",
                            "additionalProperties": {"type": "string"},
                        },
                        "slug_suffix": {"type": "string", "description": "e.g. '-s1' so kernels are distinct"},
                    },
                    "required": ["nick", "replacements"],
                },
            },
            "version_notes": {"type": "string"},
            "competition": {"type": "string", "description": "Comp these experiments target (defaults to the notebook's first competition_source)."},
            "tags": {"type": "string", "description": "Free-text tags applied to every variant."},
            "confirm": {"type": "boolean", "default": False},
        },
        required=["notebook_id", "variants"],
    ),
    destructive=True,
)
def t_variant_push(args):
    preview = {
        "action": "variant_push",
        "notebook_id": args["notebook_id"],
        "n_variants": len(args["variants"]),
        "competition": args.get("competition"),
        "variants_preview": [
            {"nick": v["nick"], "replacements": v.get("replacements"), "slug_suffix": v.get("slug_suffix")}
            for v in args["variants"][:5]
        ],
    }
    if (gate := _confirm_gate(args, preview)) is not None:
        return gate
    return actions.push_variants(
        int(args["notebook_id"]), list(args["variants"]), args.get("version_notes", ""),
        competition=args.get("competition"), tags=args.get("tags"),
    )


@register(
    "launch_sweep",
    "Launch a PARAMETER SWEEP: expand a grid into many experiments and queue them across nicks. "
    "The notebook source must contain literal {{name}} placeholders for each grid key "
    "(e.g. `lr = {{lr}}`, `SEED = {{seed}}`). Jobs are dispatched by the scheduler respecting "
    "per-nick concurrency + GPU budget. Destructive (will consume GPU quota). Confirm required.",
    _obj(
        {
            "notebook_id": {"type": "integer"},
            "grid": {
                "type": "object",
                "description": "Map param -> list of values, e.g. {\"lr\":[0.1,0.05], \"seed\":[1,2]}.",
                "additionalProperties": {"type": "array"},
            },
            "competition": {"type": "string"},
            "nicks": {"type": "array", "items": {"type": "string"}, "description": "Allowed accounts; default = all."},
            "search": {"type": "string", "enum": ["grid", "random"], "default": "grid"},
            "n": {"type": "integer", "description": "For search=random: how many combos."},
            "seed": {"type": "integer", "default": 0},
            "tags": {"type": "string"},
            "version_notes": {"type": "string"},
            "confirm": {"type": "boolean", "default": False},
        },
        required=["notebook_id", "grid"],
    ),
    destructive=True,
)
def t_launch_sweep(args):
    grid = args["grid"] or {}
    combos = orchestrator.expand_grid(
        grid, search=args.get("search", "grid"),
        n=int(args["n"]) if args.get("n") is not None else None,
        seed=int(args.get("seed", 0)),
    )
    preview = {
        "action": "launch_sweep",
        "notebook_id": args["notebook_id"],
        "n_experiments": len(combos),
        "grid": grid,
        "search": args.get("search", "grid"),
        "nicks": args.get("nicks") or "(all accounts)",
        "sample_combos": combos[:5],
    }
    if (gate := _confirm_gate(args, preview)) is not None:
        return gate
    return orchestrator.launch_sweep(
        int(args["notebook_id"]), grid,
        competition=args.get("competition"),
        nicks=args.get("nicks"),
        search=args.get("search", "grid"),
        n=int(args["n"]) if args.get("n") is not None else None,
        seed=int(args.get("seed", 0)),
        tags=args.get("tags"),
        version_notes=args.get("version_notes", ""),
    )


@register(
    "submit_to_competition",
    "Submit a local file to a Kaggle competition under a specific nick. Destructive (counts "
    "toward daily submission limit). Ask for confirmation before confirm=true.",
    _obj(
        {
            "nick": {"type": "string"},
            "competition": {"type": "string"},
            "file_path": {"type": "string"},
            "message": {"type": "string"},
            "confirm": {"type": "boolean", "default": False},
        },
        required=["nick", "competition", "file_path"],
    ),
    destructive=True,
)
def t_submit(args):
    preview = {
        "action": "submit_to_competition",
        "nick": args["nick"],
        "competition": args["competition"],
        "file_path": args["file_path"],
        "message": args.get("message", ""),
    }
    if (gate := _confirm_gate(args, preview)) is not None:
        return gate
    return actions.submit_file_to_competition(
        args["nick"], args["competition"], args["file_path"], args.get("message", "")
    )


@register(
    "sync_run_status",
    "Refresh status (and pull output if complete) for a specific run id.",
    _obj({"run_id": {"type": "integer"}}, required=["run_id"]),
)
def t_sync_run_status(args):
    return actions.sync_run_status(int(args["run_id"]))


@register(
    "sync_all_active_runs",
    "Refresh status for all runs that are queued/running. Network-heavy; safe to run.",
    _obj({}),
)
def t_sync_active(args):
    return actions.sync_all_active_runs()


@register(
    "sync_submissions_for_competition",
    "Pull every nick's submissions + leaderboard for a competition into kaglaw db. Slow.",
    _obj({"competition": {"type": "string"}}, required=["competition"]),
)
def t_sync_subs(args):
    return actions.sync_submissions_for_competition(args["competition"])


@register(
    "remove_account",
    "Remove an account from kaglaw. Destructive: confirm required. Optionally delete files.",
    _obj(
        {
            "nick": {"type": "string"},
            "delete_files": {"type": "boolean", "default": False},
            "confirm": {"type": "boolean", "default": False},
        },
        required=["nick"],
    ),
    destructive=True,
)
def t_remove_account(args):
    preview = {"action": "remove_account", "nick": args["nick"], "delete_files": bool(args.get("delete_files"))}
    if (gate := _confirm_gate(args, preview)) is not None:
        return gate
    account_mod.remove_account(args["nick"], delete_files=bool(args.get("delete_files")))
    return {"ok": True}


@register(
    "delete_notebook",
    "Delete a registered notebook from kaglaw db (does NOT delete it on Kaggle).",
    _obj(
        {"notebook_id": {"type": "integer"}, "confirm": {"type": "boolean", "default": False}},
        required=["notebook_id"],
    ),
    destructive=True,
)
def t_delete_notebook(args):
    preview = {"action": "delete_notebook", "notebook_id": args["notebook_id"]}
    if (gate := _confirm_gate(args, preview)) is not None:
        return gate
    actions.delete_notebook(int(args["notebook_id"]))
    return {"ok": True}


# ============================================================
# Helpers
# ============================================================

def _pick_account(nick: str | None):
    if nick:
        return account_mod.get_account(nick)
    lst = account_mod.list_accounts()
    return lst[0] if lst else None


def get_specs() -> list[ToolSpec]:
    return [t.spec for t in REGISTRY.values()]


def call_tool(name: str, args: dict[str, Any]) -> tuple[Any, bool]:
    """Execute a tool. Returns (result, is_error)."""
    if name not in REGISTRY:
        return {"error": f"Unknown tool: {name}"}, True
    tool = REGISTRY[name]
    try:
        result = tool.handler(args or {})
        # If the handler returned a dict with 'error', treat as soft error.
        if isinstance(result, dict) and "error" in result and len(result) <= 2:
            return result, True
        return result, False
    except Exception as exc:
        import traceback

        return {"error": str(exc), "traceback": traceback.format_exc(limit=4)}, True


def result_to_text(result: Any) -> str:
    """Stringify a tool result for inclusion in the conversation."""
    try:
        return json.dumps(result, indent=2, ensure_ascii=False, default=str)
    except Exception:
        return str(result)
