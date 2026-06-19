"""LLM abstraction supporting Anthropic Claude + OpenAI with the same tool-use loop.

Canonical message format (internal):
  {"role": "user",      "text": "..."}
  {"role": "assistant", "text": "...", "tool_calls": [{"id", "name", "args"}]}
  {"role": "tool",      "tool_call_id": "...", "name": "...", "content": "...", "is_error": bool}

Canonical response (one model turn):
  LLMResponse(text, tool_uses, stop_reason)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ToolUse:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class LLMResponse:
    text: str
    tool_uses: list[ToolUse] = field(default_factory=list)
    stop_reason: str = "end_turn"  # 'end_turn' | 'tool_use' | 'max_tokens'
    usage: dict[str, int] = field(default_factory=dict)


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]  # JSONSchema


class LLMClient:
    """Provider-agnostic chat client. Subclasses implement `chat()`."""

    provider: str = "abstract"

    def chat(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[ToolSpec],
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        raise NotImplementedError


# ----------------------------- Anthropic -----------------------------

class AnthropicClient(LLMClient):
    provider = "anthropic"
    default_model = "claude-sonnet-4-6"

    def __init__(self, api_key: str | None = None):
        from anthropic import Anthropic

        self._sdk = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def chat(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[ToolSpec],
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        anth_messages = _to_anthropic_messages(messages)
        anth_tools = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools
        ]
        # Prompt-cache the system block for cheaper repeated calls in the loop.
        system_block = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        resp = self._sdk.messages.create(
            model=model or self.default_model,
            system=system_block,
            tools=anth_tools or None,
            messages=anth_messages,
            max_tokens=max_tokens,
        )
        text_parts: list[str] = []
        tool_uses: list[ToolUse] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_uses.append(ToolUse(id=block.id, name=block.name, args=block.input or {}))
        usage = {
            "input_tokens": getattr(resp.usage, "input_tokens", 0),
            "output_tokens": getattr(resp.usage, "output_tokens", 0),
            "cache_read_input_tokens": getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
            "cache_creation_input_tokens": getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
        }
        return LLMResponse(
            text="\n".join(text_parts).strip(),
            tool_uses=tool_uses,
            stop_reason=resp.stop_reason or "end_turn",
            usage=usage,
        )


def _to_anthropic_messages(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert canonical to Anthropic format.

    Anthropic wants alternating user/assistant; tool results live inside a
    'user' message as a list of tool_result blocks.
    """
    out: list[dict[str, Any]] = []
    pending_tool_results: list[dict[str, Any]] = []

    def _flush_tool_results():
        if pending_tool_results:
            out.append({"role": "user", "content": list(pending_tool_results)})
            pending_tool_results.clear()

    for m in history:
        role = m["role"]
        if role == "user":
            _flush_tool_results()
            out.append({"role": "user", "content": m["text"]})
        elif role == "assistant":
            _flush_tool_results()
            blocks: list[dict[str, Any]] = []
            if m.get("text"):
                blocks.append({"type": "text", "text": m["text"]})
            for tc in m.get("tool_calls") or []:
                blocks.append(
                    {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["args"]}
                )
            out.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            pending_tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": m["tool_call_id"],
                    "content": m["content"],
                    "is_error": bool(m.get("is_error")),
                }
            )
    _flush_tool_results()
    return out


# ----------------------------- OpenAI -----------------------------

class OpenAIClient(LLMClient):
    provider = "openai"
    default_model = "gpt-4o"

    def __init__(self, api_key: str | None = None):
        from openai import OpenAI

        self._sdk = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def chat(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[ToolSpec],
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        oai_messages = [{"role": "system", "content": system}] + _to_openai_messages(messages)
        oai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in tools
        ]
        resp = self._sdk.chat.completions.create(
            model=model or self.default_model,
            messages=oai_messages,
            tools=oai_tools or None,
            max_tokens=max_tokens,
        )
        choice = resp.choices[0]
        msg = choice.message
        tool_uses: list[ToolUse] = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            tool_uses.append(ToolUse(id=tc.id, name=tc.function.name, args=args))
        stop_reason = "tool_use" if tool_uses else "end_turn"
        usage = {
            "input_tokens": resp.usage.prompt_tokens if resp.usage else 0,
            "output_tokens": resp.usage.completion_tokens if resp.usage else 0,
        }
        return LLMResponse(
            text=(msg.content or "").strip(),
            tool_uses=tool_uses,
            stop_reason=stop_reason,
            usage=usage,
        )


def _to_openai_messages(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in history:
        role = m["role"]
        if role == "user":
            out.append({"role": "user", "content": m["text"]})
        elif role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": m.get("text") or ""}
            if m.get("tool_calls"):
                entry["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"]),
                        },
                    }
                    for tc in m["tool_calls"]
                ]
            out.append(entry)
        elif role == "tool":
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": m["tool_call_id"],
                    "content": m["content"],
                }
            )
    return out


# ----------------------------- Factory -----------------------------

def get_client(provider: Literal["anthropic", "openai"], api_key: str | None = None) -> LLMClient:
    if provider == "anthropic":
        return AnthropicClient(api_key)
    if provider == "openai":
        return OpenAIClient(api_key)
    raise ValueError(f"Unknown provider: {provider}")
