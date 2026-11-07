"""Normalize LangChain / Gemini message content to plain text."""

from __future__ import annotations

from typing import Any


def message_text(content: Any) -> str:
    """Flatten Gemini content-block lists into a single string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # Prefer final answer text; skip pure reasoning blobs when both exist.
                text = block.get("text") or block.get("content")
                btype = str(block.get("type") or "").lower()
                if text and btype in {"", "text", "output_text", "answer"}:
                    parts.append(str(text))
                elif text and btype not in {"reasoning", "thinking"}:
                    parts.append(str(text))
                elif text and not parts:
                    parts.append(str(text))
            else:
                text = getattr(block, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    text = getattr(content, "text", None)
    if text:
        return str(text)
    return str(content)
