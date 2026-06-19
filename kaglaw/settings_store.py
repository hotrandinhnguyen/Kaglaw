"""Tiny wrapper over the `settings` SQLite table for LLM provider/model/key."""

from __future__ import annotations

import os

from . import db

_DEFAULTS = {
    "llm.provider": os.environ.get("KAGLAW_LLM_PROVIDER", "anthropic"),
    "llm.model.anthropic": os.environ.get("KAGLAW_LLM_MODEL_ANTHROPIC", "claude-sonnet-4-6"),
    "llm.model.openai": os.environ.get("KAGLAW_LLM_MODEL_OPENAI", "gpt-4o"),
    "llm.api_key.anthropic": os.environ.get("ANTHROPIC_API_KEY", ""),
    "llm.api_key.openai": os.environ.get("OPENAI_API_KEY", ""),
}


def get(key: str) -> str:
    with db.connect() as con:
        r = con.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    if r:
        return r["value"] or ""
    return _DEFAULTS.get(key, "")


def set(key: str, value: str) -> None:
    with db.connect() as con:
        con.execute(
            "INSERT INTO settings(key, value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )


def get_active_config() -> dict[str, str]:
    provider = get("llm.provider") or "anthropic"
    model = get(f"llm.model.{provider}")
    api_key = get(f"llm.api_key.{provider}")
    return {"provider": provider, "model": model, "api_key": api_key}
