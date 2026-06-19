"""Agent loop with persistent SQLite-backed conversation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Iterator

from .. import compaction, db, memory_store
from . import tools as tool_mod
from .llm import LLMClient, ToolUse, get_client
from .prompts import SYSTEM_PROMPT

log = logging.getLogger("kaglaw.agent")

MAX_ITERATIONS = 12  # hard ceiling per user turn


def _build_turn_context(conv_id: int, client: LLMClient, model: str | None):
    """Compose the effective system prompt (base + long-term memory + rolling
    summary) and the history to actually send (compacted if very long)."""
    summary_block, history = compaction.build_context(conv_id, client, model)
    system = SYSTEM_PROMPT + memory_store.memories_block() + summary_block
    return system, history


# ----------------------------- Conversation persistence -----------------------------

def create_conversation(title: str | None, provider: str, model: str | None) -> int:
    with db.connect() as con:
        cur = con.execute(
            "INSERT INTO conversations(title, provider, model) VALUES(?,?,?)",
            (title, provider, model),
        )
        return int(cur.lastrowid)


def list_conversations(limit: int = 50) -> list[dict[str, Any]]:
    with db.connect() as con:
        rows = con.execute(
            "SELECT id, title, provider, model, created_at, updated_at "
            "FROM conversations ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_conversation(conv_id: int) -> dict[str, Any] | None:
    with db.connect() as con:
        r = con.execute("SELECT * FROM conversations WHERE id=?", (conv_id,)).fetchone()
    return dict(r) if r else None


def delete_conversation(conv_id: int) -> None:
    with db.connect() as con:
        con.execute("DELETE FROM messages WHERE conversation_id=?", (conv_id,))
        con.execute("DELETE FROM conversations WHERE id=?", (conv_id,))


def _save_message(conv_id: int, role: str, *, text: str | None = None,
                  tool_calls: list[dict] | None = None,
                  tool_call_id: str | None = None,
                  tool_name: str | None = None,
                  is_error: bool = False) -> int:
    with db.connect() as con:
        cur = con.execute(
            """INSERT INTO messages(conversation_id, role, text, tool_calls,
                                    tool_call_id, tool_name, is_error)
               VALUES(?,?,?,?,?,?,?)""",
            (conv_id, role, text,
             json.dumps(tool_calls) if tool_calls else None,
             tool_call_id, tool_name, int(is_error)),
        )
        con.execute("UPDATE conversations SET updated_at=datetime('now') WHERE id=?", (conv_id,))
        return int(cur.lastrowid)


def load_history(conv_id: int) -> list[dict[str, Any]]:
    with db.connect() as con:
        rows = con.execute(
            "SELECT * FROM messages WHERE conversation_id=? ORDER BY id",
            (conv_id,),
        ).fetchall()
    history: list[dict[str, Any]] = []
    for r in rows:
        if r["role"] == "user":
            history.append({"role": "user", "text": r["text"] or ""})
        elif r["role"] == "assistant":
            entry = {"role": "assistant", "text": r["text"] or ""}
            if r["tool_calls"]:
                entry["tool_calls"] = json.loads(r["tool_calls"])
            history.append(entry)
        elif r["role"] == "tool":
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": r["tool_call_id"],
                    "name": r["tool_name"],
                    "content": r["text"] or "",
                    "is_error": bool(r["is_error"]),
                }
            )
    return history


def messages_for_display(conv_id: int) -> list[dict[str, Any]]:
    """Display-friendly transcript (text rows for user/assistant + tool-call/result pairs)."""
    with db.connect() as con:
        rows = con.execute(
            "SELECT * FROM messages WHERE conversation_id=? ORDER BY id",
            (conv_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ----------------------------- Main entry point -----------------------------

@dataclass
class TurnResult:
    iterations: int
    final_text: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read: int = 0


def run_turn(
    conv_id: int,
    user_text: str,
    *,
    provider: str = "anthropic",
    model: str | None = None,
    api_key: str | None = None,
) -> TurnResult:
    """Append the user message, run the agent loop, persist all turns. Returns the final reply."""
    client: LLMClient = get_client(provider, api_key)

    _save_message(conv_id, "user", text=user_text)
    system, history = _build_turn_context(conv_id, client, model)

    specs = tool_mod.get_specs()
    final_text = ""
    in_tok = out_tok = cache_read = 0

    for i in range(MAX_ITERATIONS):
        resp = client.chat(history, system, specs, model=model)
        in_tok += resp.usage.get("input_tokens", 0)
        out_tok += resp.usage.get("output_tokens", 0)
        cache_read += resp.usage.get("cache_read_input_tokens", 0)

        # Persist assistant turn (text + any tool_use blocks)
        tc_for_save = [
            {"id": tu.id, "name": tu.name, "args": tu.args} for tu in resp.tool_uses
        ] or None
        _save_message(conv_id, "assistant", text=resp.text or None, tool_calls=tc_for_save)
        history.append({
            "role": "assistant",
            "text": resp.text or "",
            "tool_calls": [
                {"id": tu.id, "name": tu.name, "args": tu.args} for tu in resp.tool_uses
            ],
        })

        if not resp.tool_uses:
            final_text = resp.text or ""
            break

        # Execute each tool_use and append tool_result back into history.
        for tu in resp.tool_uses:
            result, is_err = tool_mod.call_tool(tu.name, tu.args)
            content = tool_mod.result_to_text(result)
            # Truncate very large outputs to keep context manageable.
            if len(content) > 20000:
                content = content[:20000] + "\n...[truncated]"
            _save_message(
                conv_id, "tool", text=content,
                tool_call_id=tu.id, tool_name=tu.name, is_error=is_err,
            )
            history.append({
                "role": "tool",
                "tool_call_id": tu.id,
                "name": tu.name,
                "content": content,
                "is_error": is_err,
            })
        # loop: call the model again with the tool results
    else:
        final_text = "[Agent đã chạm trần số bước (%d). Dừng để tránh loop.]" % MAX_ITERATIONS
        _save_message(conv_id, "assistant", text=final_text)

    return TurnResult(
        iterations=i + 1,
        final_text=final_text,
        total_input_tokens=in_tok,
        total_output_tokens=out_tok,
        total_cache_read=cache_read,
    )


# ----------------------------- Streaming variant -----------------------------

def run_turn_streaming(
    conv_id: int,
    user_text: str,
    *,
    provider: str = "anthropic",
    model: str | None = None,
    api_key: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Same logic as run_turn but yields step-by-step events:

      {"event": "msg",    "msg": <message dict for _msg.html>}
      {"event": "status", "text": "..."}
      {"event": "done",   "iterations": int}
      {"event": "error",  "message": str}
    """
    try:
        client: LLMClient = get_client(provider, api_key)
    except Exception as exc:
        yield {"event": "error", "message": f"LLM init failed: {exc}"}
        return

    user_msg_id = _save_message(conv_id, "user", text=user_text)
    yield {
        "event": "msg",
        "msg": {"id": user_msg_id, "role": "user", "text": user_text,
                "tool_calls": None, "tool_call_id": None,
                "tool_name": None, "is_error": 0},
    }

    system, history = _build_turn_context(conv_id, client, model)
    specs = tool_mod.get_specs()
    iterations = 0

    for i in range(MAX_ITERATIONS):
        iterations = i + 1
        yield {"event": "status", "text": f"Thinking… (step {iterations})"}
        try:
            resp = client.chat(history, system, specs, model=model)
        except Exception as exc:
            log.exception("LLM call failed")
            yield {"event": "error", "message": f"LLM call failed: {exc}"}
            return

        tc_for_save = [
            {"id": tu.id, "name": tu.name, "args": tu.args} for tu in resp.tool_uses
        ] or None
        assistant_id = _save_message(
            conv_id, "assistant", text=resp.text or None, tool_calls=tc_for_save
        )
        yield {
            "event": "msg",
            "msg": {
                "id": assistant_id, "role": "assistant",
                "text": resp.text or "",
                "tool_calls": json.dumps(tc_for_save) if tc_for_save else None,
                "tool_call_id": None, "tool_name": None, "is_error": 0,
            },
        }
        history.append({
            "role": "assistant",
            "text": resp.text or "",
            "tool_calls": [
                {"id": tu.id, "name": tu.name, "args": tu.args} for tu in resp.tool_uses
            ],
        })

        if not resp.tool_uses:
            yield {"event": "done", "iterations": iterations}
            return

        for tu in resp.tool_uses:
            yield {"event": "status", "text": f"Running tool {tu.name}…"}
            result, is_err = tool_mod.call_tool(tu.name, tu.args)
            content = tool_mod.result_to_text(result)
            if len(content) > 20000:
                content = content[:20000] + "\n...[truncated]"
            tool_id = _save_message(
                conv_id, "tool", text=content,
                tool_call_id=tu.id, tool_name=tu.name, is_error=is_err,
            )
            yield {
                "event": "msg",
                "msg": {
                    "id": tool_id, "role": "tool", "text": content,
                    "tool_calls": None, "tool_call_id": tu.id,
                    "tool_name": tu.name, "is_error": int(is_err),
                },
            }
            history.append({
                "role": "tool", "tool_call_id": tu.id, "name": tu.name,
                "content": content, "is_error": is_err,
            })

    final_text = f"[Agent đã chạm trần {MAX_ITERATIONS} bước.]"
    cap_id = _save_message(conv_id, "assistant", text=final_text)
    yield {
        "event": "msg",
        "msg": {"id": cap_id, "role": "assistant", "text": final_text,
                "tool_calls": None, "tool_call_id": None, "tool_name": None,
                "is_error": 0},
    }
    yield {"event": "done", "iterations": iterations}
