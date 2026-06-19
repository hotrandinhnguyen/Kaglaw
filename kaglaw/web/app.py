from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    PlainTextResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .. import accounts as account_mod
from .. import actions, budgets, db, excel_export, memory_store, notebook_builder, orchestrator, scheduler, settings_store
from ..agent import loop as agent_loop
from ..config import NOTEBOOKS_DIR

log = logging.getLogger("kaglaw.web")

BASE_DIR = Path(__file__).parent

app = FastAPI(title="kaglaw", version="0.1.0")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def _jinja_fromjson(value):
    if value in (None, ""):
        return []
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return []


_md = None


def _jinja_md(text: str | None) -> str:
    """Render assistant text as sanitized HTML markdown."""
    global _md
    if not text:
        return ""
    if _md is None:
        from markdown_it import MarkdownIt
        _md = MarkdownIt("commonmark", {"breaks": True, "linkify": True}).enable(
            ["table", "strikethrough"]
        )
    import bleach
    html = _md.render(text)
    return bleach.clean(
        html,
        tags={
            "p", "br", "strong", "em", "code", "pre", "ul", "ol", "li",
            "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "a",
            "table", "thead", "tbody", "tr", "th", "td", "hr", "del", "img",
        },
        attributes={"a": ["href", "title"], "img": ["src", "alt", "title"]},
        protocols={"http", "https", "mailto"},
        strip=True,
    )


templates.env.filters["fromjson"] = _jinja_fromjson
templates.env.filters["md"] = _jinja_md
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.on_event("startup")
def _on_startup() -> None:
    db.init_db()
    account_mod.sync_accounts_to_db()
    scheduler.start()


@app.on_event("shutdown")
def _on_shutdown() -> None:
    scheduler.stop()


def _render(request: Request, name: str, **extra):
    return templates.TemplateResponse(request, name, extra)


# ---------------------------- Dashboard ----------------------------

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    accs = account_mod.list_accounts()
    runs = actions.list_runs(limit=20)
    active = actions.list_runs(active_only=True, limit=100)
    subs = actions.list_submissions()[:20]
    return _render(
        request,
        "index.html",
        accounts=accs,
        recent_runs=runs,
        active_count=len(active),
        recent_subs=subs,
    )


# ---------------------------- Accounts ----------------------------

@app.get("/accounts", response_class=HTMLResponse)
def accounts_page(request: Request):
    return _render(request, "accounts.html", accounts=account_mod.list_accounts())


@app.post("/accounts/add")
async def accounts_add(
    nick: str = Form(...),
    notes: str = Form(""),
    kaggle_json: UploadFile = File(...),
):
    data = await kaggle_json.read()
    try:
        account_mod.add_account_from_json(nick, data, notes or None)
    except Exception as exc:
        raise HTTPException(400, f"Add failed: {exc}")
    return RedirectResponse("/accounts", status_code=303)


@app.post("/accounts/{nick}/delete")
def accounts_delete(nick: str, delete_files: str = Form("")):
    account_mod.remove_account(nick, delete_files=bool(delete_files))
    return RedirectResponse("/accounts", status_code=303)


@app.post("/accounts/rescan")
def accounts_rescan():
    account_mod.sync_accounts_to_db()
    return RedirectResponse("/accounts", status_code=303)


# ---------------------------- Notebooks ----------------------------

@app.get("/notebooks", response_class=HTMLResponse)
def notebooks_page(request: Request):
    return _render(
        request,
        "notebooks.html",
        notebooks=actions.list_notebooks(),
        accounts=account_mod.list_accounts(),
    )


@app.post("/notebooks/upload")
async def notebooks_upload(
    title: str = Form(...),
    language: str = Form("python"),
    kernel_type: str = Form("notebook"),
    enable_gpu: str = Form(""),
    enable_tpu: str = Form(""),
    enable_internet: str = Form("on"),
    dataset_sources: str = Form(""),
    competition_sources: str = Form(""),
    kernel_sources: str = Form(""),
    notebook_file: UploadFile = File(...),
):
    safe_dir = NOTEBOOKS_DIR / actions.slugify(title)
    safe_dir.mkdir(parents=True, exist_ok=True)
    target = safe_dir / (notebook_file.filename or "notebook.ipynb")
    with target.open("wb") as f:
        shutil.copyfileobj(notebook_file.file, f)

    spec = actions.NotebookSpec(
        title=title,
        local_path=str(target),
        language=language,
        kernel_type=kernel_type,
        enable_gpu=bool(enable_gpu),
        enable_tpu=bool(enable_tpu),
        enable_internet=bool(enable_internet),
        dataset_sources=_split_csv(dataset_sources),
        competition_sources=_split_csv(competition_sources),
        kernel_sources=_split_csv(kernel_sources),
    )
    actions.register_notebook(spec)
    return RedirectResponse("/notebooks", status_code=303)


@app.get("/notebooks/{nb_id}/code", response_class=PlainTextResponse)
def notebooks_code(nb_id: int):
    nb = actions.get_notebook(nb_id)
    if not nb:
        raise HTTPException(404, "Notebook not found")
    try:
        return notebook_builder.read_source(nb["local_path"])
    except Exception as exc:  # noqa: BLE001
        return f"[could not read code: {exc}]"


@app.get("/notebooks/{nb_id}/editor", response_class=HTMLResponse)
def notebooks_editor(nb_id: int):
    """In-browser editor: textarea + 'Save (local)' + 'Save & push' (new Kaggle version)."""
    import html as _h
    nb = actions.get_notebook(nb_id)
    if not nb:
        raise HTTPException(404, "Notebook not found")
    try:
        code = notebook_builder.read_source(nb["local_path"])
    except Exception as exc:  # noqa: BLE001
        code = f"[could not read: {exc}]"
    esc = _h.escape(code)
    nick_boxes = "".join(
        f'<label class="inline"><input type="checkbox" name="nicks" value="{_h.escape(a.nick)}"> '
        f'{_h.escape(a.nick)}</label>'
        for a in account_mod.list_accounts()
    ) or '<span class="muted">Chưa có nick nào.</span>'
    confirm = ("Push = tạo version mới trên Kaggle + chạy (tốn GPU). Tiếp tục?")
    return (
        f'<form id="ed-{nb_id}">'
        f'<textarea name="code" rows="22" class="code-edit" spellcheck="false">{esc}</textarea>'
        f'<div class="row" style="margin-top:6px;gap:8px;align-items:center;">'
        f'<input name="version_notes" placeholder="version note (cho Save & push)" style="flex:1;min-width:160px;">'
        f'</div>'
        f'<div class="row" style="margin-top:4px;gap:10px;flex-wrap:wrap;">{nick_boxes}</div>'
        f'<div class="row" style="margin-top:8px;gap:10px;align-items:center;">'
        f'<button class="btn-primary" hx-post="/notebooks/{nb_id}/code" '
        f'hx-target="#sp-{nb_id}" hx-swap="innerHTML">💾 Lưu (local)</button>'
        f'<button hx-post="/notebooks/{nb_id}/save-push" hx-target="#sp-{nb_id}" hx-swap="innerHTML" '
        f'hx-confirm="{confirm}">⬆ Save &amp; push (version mới)</button>'
        f'<span id="sp-{nb_id}" class="muted"></span></div></form>'
        '<p class="muted" style="margin-top:4px;">Tách cell bằng <code>#&nbsp;%%</code>. '
        '<b>Lưu (local)</b> chỉ ghi file trên máy. <b>Save &amp; push</b> = lưu + tạo '
        '<b>version mới trên Kaggle</b> dưới các nick đã chọn (giống nút Save Version của Kaggle).</p>'
    )


@app.post("/notebooks/{nb_id}/code", response_class=HTMLResponse)
def notebooks_code_save(nb_id: int, code: str = Form(...)):
    nb = actions.get_notebook(nb_id)
    if not nb:
        raise HTTPException(404, "Notebook not found")
    try:
        notebook_builder.overwrite_source(nb["local_path"], code)
    except Exception as exc:  # noqa: BLE001
        return f"<span style='color:var(--danger)'>❌ {exc}</span>"
    return f"<span style='color:var(--ok)'>✅ Đã lưu {len(code)} ký tự (local)</span>"


@app.post("/notebooks/{nb_id}/save-push", response_class=HTMLResponse)
def notebooks_save_push(
    nb_id: int,
    code: str = Form(...),
    version_notes: str = Form(""),
    nicks: list[str] = Form(default=[]),
):
    nb = actions.get_notebook(nb_id)
    if not nb:
        raise HTTPException(404, "Notebook not found")
    if not nicks:
        return "<span style='color:var(--danger)'>❌ Chọn ít nhất 1 nick để push.</span>"
    try:
        notebook_builder.overwrite_source(nb["local_path"], code)
    except Exception as exc:  # noqa: BLE001
        return f"<span style='color:var(--danger)'>❌ Lưu lỗi: {exc}</span>"
    results = actions.push_notebook_to_accounts(nb_id, nicks, version_notes)
    oks = [r for r in results if r.get("ok")]
    parts = []
    for r in results:
        if r.get("ok"):
            parts.append(f"{r['nick']}: ✅ v{r.get('version') or '?'} (run #{r.get('run_id')})")
        else:
            parts.append(f"{r['nick']}: ❌ {r.get('error')}")
    color = "var(--ok)" if oks else "var(--danger)"
    return (f"<span style='color:{color}'>Đã lưu + push {len(oks)}/{len(results)} nick.</span><br>"
            + "<br>".join(parts))


@app.post("/notebooks/import")
def notebooks_import(kernel_ref: str = Form(...), as_nick: str = Form("")):
    actions.import_kernel(kernel_ref, as_nick=as_nick or None)
    return RedirectResponse("/notebooks", status_code=303)


@app.post("/notebooks/{nb_id}/test-local", response_class=HTMLResponse)
def notebooks_test_local(nb_id: int):
    from .. import runners
    res = runners.run_notebook_local(nb_id)
    if res.get("ok"):
        head = f"<b style='color:var(--ok)'>✅ PASS</b> · {res['duration_seconds']}s"
    elif res.get("timed_out"):
        head = f"<b style='color:var(--warn)'>⏱ TIMEOUT</b> · {res['duration_seconds']}s"
    else:
        head = f"<b style='color:var(--danger)'>❌ FAIL</b> rc={res.get('returncode')}"
    metric = res.get("metric")
    metric_txt = f" · metric {metric['name']}={metric['value']}" if metric else ""
    files = ", ".join(res.get("output_files") or []) or "(none)"
    tail = (res.get("stderr_tail") or "")[-1500:] or (res.get("stdout_tail") or "")[-1500:]
    import html as _html
    return (f"{head}{metric_txt}<br>output files: {_html.escape(files)}"
            f"<pre style='max-height:240px;overflow:auto'>{_html.escape(tail)}</pre>")


@app.post("/notebooks/{nb_id}/delete")
def notebooks_delete(nb_id: int):
    actions.delete_notebook(nb_id)
    return RedirectResponse("/notebooks", status_code=303)


@app.post("/notebooks/{nb_id}/push")
def notebooks_push(
    nb_id: int,
    nicks: list[str] = Form(...),
    version_notes: str = Form(""),
):
    results = actions.push_notebook_to_accounts(nb_id, nicks, version_notes)
    return {"results": results}


# ---------------------------- Runs ----------------------------

@app.get("/runs", response_class=HTMLResponse)
def runs_page(request: Request, active: int = 0):
    runs = actions.list_runs(active_only=bool(active), limit=300)
    return _render(request, "runs.html", runs=runs, active=bool(active))


@app.post("/runs/sync")
def runs_sync_all():
    res = actions.sync_all_active_runs()
    return res


@app.post("/runs/{run_id}/sync")
def runs_sync_one(run_id: int):
    return actions.sync_run_status(run_id)


# ---------------------------- Experiments ----------------------------

def _svg_metric_chart(rows: list[dict], best_id=None, width=720, height=180) -> str:
    """Dependency-free SVG bar chart of metric_value per run (best highlighted)."""
    pts = [r for r in rows if r.get("metric_value") is not None]
    if not pts:
        return ""
    pad_l, pad_b, pad_t = 44, 22, 12
    vals = [float(r["metric_value"]) for r in pts]
    vmin, vmax = min(vals), max(vals)
    span = (vmax - vmin) or (abs(vmax) or 1.0)
    lo = vmin - span * 0.1
    hi = vmax + span * 0.1
    n = len(pts)
    plot_w = width - pad_l - 8
    plot_h = height - pad_b - pad_t
    bw = plot_w / n
    bars = []
    for i, r in enumerate(pts):
        v = float(r["metric_value"])
        h = (v - lo) / (hi - lo) * plot_h if hi > lo else plot_h
        x = pad_l + i * bw + bw * 0.15
        y = pad_t + (plot_h - h)
        is_best = best_id is not None and r.get("run_id") == best_id
        fill = "#3ec27a" if is_best else "#4f9dff"
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw*0.7:.1f}" height="{h:.1f}" '
            f'fill="{fill}" rx="2"><title>run #{r.get("run_id")}: {v}</title></rect>'
        )
    axis = (
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+plot_h}" stroke="#2a3140"/>'
        f'<line x1="{pad_l}" y1="{pad_t+plot_h}" x2="{width-8}" y2="{pad_t+plot_h}" stroke="#2a3140"/>'
        f'<text x="4" y="{pad_t+8}" fill="#8a93a6" font-size="10">{hi:.4g}</text>'
        f'<text x="4" y="{pad_t+plot_h}" fill="#8a93a6" font-size="10">{lo:.4g}</text>'
    )
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
            f'style="background:#0a0c10;border-radius:8px;">{axis}{"".join(bars)}</svg>')


@app.get("/experiments", response_class=HTMLResponse)
def experiments_page(request: Request, competition: str = "", batch_id: str = ""):
    comp = competition or None
    batch = batch_id or None
    matrix = actions.compare_experiments(competition=comp, batch_id=batch, limit=300)
    with db.connect() as con:
        comps = [
            r[0] for r in con.execute(
                "SELECT DISTINCT competition FROM runs "
                "WHERE competition IS NOT NULL AND competition<>'' ORDER BY competition"
            ).fetchall()
        ]
    best_id = matrix["best"]["run_id"] if matrix.get("best") else None
    chart = _svg_metric_chart(matrix.get("rows", []), best_id=best_id)
    return _render(
        request, "experiments.html",
        matrix=matrix, competitions=comps, competition=comp or "", batch_id=batch or "",
        chart=chart,
    )


@app.post("/experiments/reextract")
def experiments_reextract(competition: str = Form(""), only_missing: str = Form("on")):
    return actions.reextract_metrics(
        competition=competition or None, only_missing=bool(only_missing)
    )


# ---------------------------- Batches / queue ----------------------------

@app.get("/batches", response_class=HTMLResponse)
def batches_page(request: Request):
    accs = account_mod.list_accounts()
    return _render(
        request, "batches.html",
        batches=orchestrator.list_batches(limit=50),
        budgets=budgets.all_usages([a.nick for a in accs]),
        notifications=orchestrator.list_notifications(limit=10),
    )


@app.get("/status", response_class=HTMLResponse)
def status_page(request: Request):
    return _render(request, "status.html", status=orchestrator.nick_status())


@app.post("/batches/{batch_id}/cancel")
def batches_cancel(batch_id: str):
    return orchestrator.cancel_batch(batch_id)


@app.post("/jobs/dispatch")
def jobs_dispatch():
    return orchestrator.dispatch_jobs()


@app.post("/notifications/seen")
def notifications_seen():
    return orchestrator.mark_notifications_seen()


# ---------------------------- Competitions / Submissions ----------------------------

@app.get("/competitions", response_class=HTMLResponse)
def competitions_page(request: Request):
    tracked = actions.list_tracked_competitions()
    return _render(
        request,
        "competitions.html",
        tracked=tracked,
        accounts=account_mod.list_accounts(),
        recent=actions.list_submissions()[:50],
    )


@app.post("/competitions/sync")
def competitions_sync(competition: str = Form(...)):
    res = actions.sync_submissions_for_competition(competition)
    return res


@app.post("/competitions/submit")
async def competitions_submit(
    nick: str = Form(...),
    competition: str = Form(...),
    message: str = Form(""),
    submission_file: UploadFile = File(...),
):
    tmp = NOTEBOOKS_DIR / "_submissions"
    tmp.mkdir(parents=True, exist_ok=True)
    fpath = tmp / (submission_file.filename or "submission.csv")
    with fpath.open("wb") as f:
        shutil.copyfileobj(submission_file.file, f)
    res = actions.submit_file_to_competition(nick, competition, str(fpath), message)
    return res


@app.get("/competitions/{competition}", response_class=HTMLResponse)
def competition_detail(request: Request, competition: str):
    subs = actions.list_submissions(competition)
    return _render(
        request, "competition_detail.html", competition=competition, submissions=subs
    )


# ---------------------------- Chat / Agent ----------------------------

@app.get("/chat", response_class=HTMLResponse)
def chat_index(request: Request):
    convs = agent_loop.list_conversations(limit=40)
    return _render(request, "chat_list.html", conversations=convs,
                   settings=settings_store.get_active_config())


@app.post("/chat/new")
def chat_new():
    cfg = settings_store.get_active_config()
    conv_id = agent_loop.create_conversation(title=None, provider=cfg["provider"], model=cfg["model"])
    return RedirectResponse(f"/chat/{conv_id}", status_code=303)


@app.get("/chat/{conv_id}", response_class=HTMLResponse)
def chat_view(request: Request, conv_id: int):
    conv = agent_loop.get_conversation(conv_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")
    msgs = agent_loop.messages_for_display(conv_id)
    return _render(request, "chat.html", conv=conv, messages=msgs,
                   settings=settings_store.get_active_config())


def _sse_format(event: str, data: str) -> bytes:
    payload = "".join(f"data: {line}\n" for line in data.splitlines() or [""])
    return f"event: {event}\n{payload}\n".encode("utf-8")


@app.post("/chat/{conv_id}/send")
def chat_send_stream(conv_id: int, prompt: str = Form(...)):
    conv = agent_loop.get_conversation(conv_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")
    cfg = settings_store.get_active_config()
    if not cfg.get("api_key"):
        raise HTTPException(400, f"Missing API key for {cfg['provider']}. Set it in /settings.")

    # Auto-title from the first user message if still untitled.
    if not conv.get("title"):
        title = (prompt[:60] + "…") if len(prompt) > 60 else prompt
        with db.connect() as con:
            con.execute("UPDATE conversations SET title=? WHERE id=?", (title, conv_id))

    msg_tmpl = templates.env.get_template("_msg.html")

    def gen():
        try:
            for ev in agent_loop.run_turn_streaming(
                conv_id, prompt,
                provider=cfg["provider"], model=cfg["model"], api_key=cfg["api_key"],
            ):
                if ev["event"] == "msg":
                    html = msg_tmpl.render(m=ev["msg"])
                    yield _sse_format("append", html)
                elif ev["event"] == "status":
                    yield _sse_format("status", ev["text"])
                elif ev["event"] == "error":
                    yield _sse_format("error", ev["message"])
                elif ev["event"] == "done":
                    yield _sse_format("done", str(ev.get("iterations", 0)))
        except Exception as exc:
            log.exception("stream gen failed")
            yield _sse_format("error", str(exc))

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/chat/{conv_id}/delete")
def chat_delete(conv_id: int):
    agent_loop.delete_conversation(conv_id)
    return RedirectResponse("/chat", status_code=303)


# ---------------------------- Memory ----------------------------

@app.get("/memory", response_class=HTMLResponse)
def memory_page(request: Request):
    return _render(request, "memory.html", memories=memory_store.list_memories(limit=500))


@app.post("/memory/add")
def memory_add(text: str = Form(...), kind: str = Form("preference")):
    if text.strip():
        memory_store.add_memory(text, kind=kind)
    return RedirectResponse("/memory", status_code=303)


@app.post("/memory/{mem_id}/delete")
def memory_delete(mem_id: int):
    memory_store.delete_memory(mem_id)
    return RedirectResponse("/memory", status_code=303)


# ---------------------------- Settings ----------------------------

@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request):
    return _render(
        request, "settings.html",
        provider=settings_store.get("llm.provider") or "anthropic",
        anthropic_model=settings_store.get("llm.model.anthropic"),
        openai_model=settings_store.get("llm.model.openai"),
        anthropic_key_set=bool(settings_store.get("llm.api_key.anthropic")),
        openai_key_set=bool(settings_store.get("llm.api_key.openai")),
    )


@app.post("/settings/save")
def settings_save(
    provider: str = Form(...),
    anthropic_model: str = Form(""),
    openai_model: str = Form(""),
    anthropic_key: str = Form(""),
    openai_key: str = Form(""),
):
    settings_store.set("llm.provider", provider)
    if anthropic_model:
        settings_store.set("llm.model.anthropic", anthropic_model)
    if openai_model:
        settings_store.set("llm.model.openai", openai_model)
    if anthropic_key:
        settings_store.set("llm.api_key.anthropic", anthropic_key)
    if openai_key:
        settings_store.set("llm.api_key.openai", openai_key)
    return RedirectResponse("/settings", status_code=303)


# ---------------------------- Export ----------------------------

@app.get("/export.xlsx")
def export_xlsx():
    path = excel_export.export_to_file()
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=path.name,
    )


# ---------------------------- helpers ----------------------------

def _split_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]
