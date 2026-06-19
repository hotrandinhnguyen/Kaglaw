"""Context compaction: keep very long chats inside the model's window.

When a conversation's full history exceeds CONTEXT_BUDGET_CHARS, we fold the
OLD messages into a rolling text summary (one LLM call, cached on the
conversation row) and only send [recent verbatim tail] to the model, with the
summary injected into the system prompt. Short chats are sent unchanged.

build_context(conv_id, client, model) -> (system_suffix, send_history)
  system_suffix : "" or a "## Tóm tắt phần trước…" block to append to the system prompt
  send_history  : canonical history list to actually send this turn
"""

from __future__ import annotations

import logging
from typing import Any

from . import db
from .config import CONTEXT_BUDGET_CHARS, CONTEXT_RECENT_CHARS

log = logging.getLogger("kaglaw.compaction")

_SUMMARY_SYS = (
    "Bạn là bộ NÉN HỘI THOẠI cho một agent Kaggle. Tóm tắt đoạn hội thoại dưới đây thành "
    "bản ghi nhớ ngắn gọn (tiếng Việt, gạch đầu dòng), GIỮ LẠI: quyết định đã chốt, ID quan "
    "trọng (notebook_id, run_id, batch_id, competition slug), score/metric, sở thích người dùng, "
    "và việc còn dang dở. Bỏ chi tiết rườm rà. Không bịa."
)


def _load_with_ids(conv_id: int) -> list[tuple[int, dict[str, Any]]]:
    with db.connect() as con:
        rows = con.execute(
            "SELECT * FROM messages WHERE conversation_id=? ORDER BY id", (conv_id,)
        ).fetchall()
    out: list[tuple[int, dict[str, Any]]] = []
    import json
    for r in rows:
        if r["role"] == "user":
            c = {"role": "user", "text": r["text"] or ""}
        elif r["role"] == "assistant":
            c = {"role": "assistant", "text": r["text"] or ""}
            if r["tool_calls"]:
                c["tool_calls"] = json.loads(r["tool_calls"])
        elif r["role"] == "tool":
            c = {"role": "tool", "tool_call_id": r["tool_call_id"], "name": r["tool_name"],
                 "content": r["text"] or "", "is_error": bool(r["is_error"])}
        else:
            continue
        out.append((r["id"], c))
    return out


def _chars(c: dict[str, Any]) -> int:
    n = len(c.get("text") or "") + len(c.get("content") or "")
    for tc in c.get("tool_calls") or []:
        n += len(str(tc.get("args"))) + len(tc.get("name", ""))
    return n


def _render(pairs: list[tuple[int, dict[str, Any]]]) -> str:
    lines: list[str] = []
    for _id, c in pairs:
        role = c["role"]
        if role == "user":
            lines.append(f"User: {c.get('text', '')}")
        elif role == "assistant":
            if c.get("text"):
                lines.append(f"Assistant: {c['text']}")
            for tc in c.get("tool_calls") or []:
                lines.append(f"Assistant→tool {tc.get('name')}({str(tc.get('args'))[:300]})")
        elif role == "tool":
            lines.append(f"Tool {c.get('name')} -> {(c.get('content') or '')[:400]}")
    return "\n".join(lines)


def _summarize(conv_id: int, old: list[tuple[int, dict[str, Any]]], client, model) -> str:
    """Return a summary covering `old`, reusing/extending the stored one. Caches it."""
    with db.connect() as con:
        row = con.execute(
            "SELECT summary, summary_upto FROM conversations WHERE id=?", (conv_id,)
        ).fetchone()
    prev_summary = (row["summary"] if row else "") or ""
    prev_upto = (row["summary_upto"] if row else 0) or 0
    last_id = old[-1][0]
    if prev_summary and prev_upto >= last_id:
        return prev_summary  # cached summary already covers all old messages

    new_msgs = [(i, c) for (i, c) in old if i > prev_upto]
    body = (f"[Tóm tắt đã có:]\n{prev_summary}\n\n" if prev_summary else "") \
        + "[Hội thoại cần gộp thêm:]\n" + _render(new_msgs)
    try:
        resp = client.chat([{"role": "user", "text": body}], _SUMMARY_SYS, [], model=model, max_tokens=1024)
        summary = (resp.text or "").strip() or prev_summary
    except Exception:  # noqa: BLE001
        log.exception("summarize failed; reusing previous summary")
        summary = prev_summary or "(không tóm tắt được phần cũ)"
    with db.connect() as con:
        con.execute("UPDATE conversations SET summary=?, summary_upto=? WHERE id=?",
                    (summary, last_id, conv_id))
    return summary


def build_context(conv_id: int, client, model: str | None) -> tuple[str, list[dict[str, Any]]]:
    pairs = _load_with_ids(conv_id)
    canon = [c for _i, c in pairs]
    total = sum(_chars(c) for c in canon)
    if total <= CONTEXT_BUDGET_CHARS or len(pairs) <= 2:
        return "", canon

    # Find cutoff: recent tail starts at a user message and fits in RECENT budget.
    acc = 0
    cutoff = 0
    for i in range(len(canon) - 1, -1, -1):
        acc += _chars(canon[i])
        if canon[i]["role"] == "user":
            if acc <= CONTEXT_RECENT_CHARS:
                cutoff = i
            else:
                break
    if cutoff <= 0:
        # fallback: keep from the last user message so the tail is valid
        for i in range(len(canon) - 1, -1, -1):
            if canon[i]["role"] == "user":
                cutoff = i
                break
    if cutoff <= 0:
        return "", canon  # can't compact safely; send as-is

    old = pairs[:cutoff]
    recent = canon[cutoff:]
    summary = _summarize(conv_id, old, client, model)
    block = ("\n\n## Tóm tắt phần TRƯỚC của cuộc trò chuyện này (đã nén để tiết kiệm ngữ cảnh, "
             "coi như đã xảy ra):\n" + summary)
    return block, recent
