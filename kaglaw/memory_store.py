"""Long-term agent memory: facts/preferences recalled across ALL chats.

Unlike per-conversation history (which lives in `messages`), memories persist
globally and are injected into the system prompt every turn — so a preference
stated in one chat ("luôn dùng nick main, metric AUC, 5-fold") is respected in
every later chat. The agent writes them with the `remember` tool; the user can
also manage them at /memory.
"""

from __future__ import annotations

from typing import Any

from . import db
from .config import MEMORY_INJECT_MAX_CHARS

VALID_KINDS = ("preference", "fact", "note")


def add_memory(text: str, kind: str = "fact", tags: str | None = None, pinned: bool = True) -> int:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty memory text")
    kind = kind if kind in VALID_KINDS else "fact"
    with db.connect() as con:
        cur = con.execute(
            "INSERT INTO memories(kind, text, tags, pinned) VALUES(?,?,?,?)",
            (kind, text, tags, int(pinned)),
        )
        return int(cur.lastrowid)


def list_memories(kind: str | None = None, pinned_only: bool = False, limit: int = 200) -> list[dict[str, Any]]:
    q = "SELECT * FROM memories WHERE 1=1"
    params: list[Any] = []
    if kind:
        q += " AND kind=?"; params.append(kind)
    if pinned_only:
        q += " AND pinned=1"
    q += " ORDER BY pinned DESC, id DESC LIMIT ?"; params.append(limit)
    with db.connect() as con:
        return [dict(r) for r in con.execute(q, tuple(params)).fetchall()]


def search_memories(query: str, limit: int = 50) -> list[dict[str, Any]]:
    like = f"%{query}%"
    with db.connect() as con:
        rows = con.execute(
            "SELECT * FROM memories WHERE text LIKE ? OR tags LIKE ? ORDER BY id DESC LIMIT ?",
            (like, like, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def update_memory(mem_id: int, text: str | None = None, kind: str | None = None,
                  pinned: bool | None = None) -> dict[str, Any]:
    sets, params = [], []
    if text is not None:
        sets.append("text=?"); params.append(text.strip())
    if kind is not None:
        sets.append("kind=?"); params.append(kind if kind in VALID_KINDS else "fact")
    if pinned is not None:
        sets.append("pinned=?"); params.append(int(pinned))
    if not sets:
        return {"ok": False, "error": "nothing to update"}
    sets.append("updated_at=datetime('now')")
    params.append(mem_id)
    with db.connect() as con:
        cur = con.execute(f"UPDATE memories SET {', '.join(sets)} WHERE id=?", tuple(params))
    if cur.rowcount == 0:
        return {"ok": False, "error": f"memory {mem_id} not found"}
    return {"ok": True, "id": mem_id}


def delete_memory(mem_id: int) -> dict[str, Any]:
    with db.connect() as con:
        cur = con.execute("DELETE FROM memories WHERE id=?", (mem_id,))
    return {"ok": cur.rowcount > 0, "id": mem_id}


def memories_block() -> str:
    """Render pinned memories as a system-prompt section (capped). '' if none."""
    mems = list_memories(pinned_only=True, limit=200)
    if not mems:
        return ""
    lines = ["## Bộ nhớ dài hạn (áp dụng cho MỌI cuộc chat — hãy tôn trọng):"]
    for m in mems:
        lines.append(f"- [{m['kind']}] {m['text']}")
    block = "\n".join(lines)
    if len(block) > MEMORY_INJECT_MAX_CHARS:
        block = block[:MEMORY_INJECT_MAX_CHARS] + "\n- …(bộ nhớ bị cắt bớt)"
    return "\n\n" + block
