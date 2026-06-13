"""
KiteFishAI SDK — Response Types
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, List, Optional


# ─── Chat Types ────────────────────────────────────────────────────────────────

@dataclass
class Message:
    role: str
    content: str


@dataclass
class Choice:
    index: int
    message: Message
    finish_reason: Optional[str] = None


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ChatCompletion:
    id: str
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None
    created: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict) -> "ChatCompletion":
        choices = []
        for c in data.get("choices", []):
            msg = c.get("message", {})
            choices.append(Choice(
                index=c.get("index", 0),
                message=Message(
                    role=msg.get("role", "assistant"),
                    content=msg.get("content", ""),
                ),
                finish_reason=c.get("finish_reason"),
            ))

        usage_data = data.get("usage")
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        ) if usage_data else None

        return cls(
            id=data.get("id", ""),
            model=data.get("model", ""),
            choices=choices,
            usage=usage,
            created=data.get("created"),
        )


# ─── Stream Types ───────────────────────────────────────────────────────────────

@dataclass
class StreamChunk:
    """A single SSE chunk from a streaming chat response."""
    id: str
    model: str
    delta: str                        # the text delta for this chunk
    finish_reason: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "StreamChunk":
        choices = data.get("choices", [{}])
        delta = choices[0].get("delta", {}).get("content", "") if choices else ""
        finish_reason = choices[0].get("finish_reason") if choices else None
        return cls(
            id=data.get("id", ""),
            model=data.get("model", ""),
            delta=delta or "",
            finish_reason=finish_reason,
        )


class ChatStream:
    """
    Iterator over streaming chat chunks.

    Usage::

        with client.chat.stream(...) as stream:
            for chunk in stream:
                print(chunk.delta, end="", flush=True)

        full_text = stream.get_final_text()
    """

    def __init__(self, response_ctx) -> None:
        self._ctx = response_ctx
        self._response = None
        self._chunks: List[StreamChunk] = []

    def __enter__(self) -> "ChatStream":
        self._response = self._ctx.__enter__()
        return self

    def __exit__(self, *args) -> None:
        self._ctx.__exit__(*args)

    def __iter__(self) -> Iterator[StreamChunk]:
        import json

        for line in self._response.iter_lines():
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                data = json.loads(payload)
                chunk = StreamChunk.from_dict(data)
                self._chunks.append(chunk)
                yield chunk
            except (json.JSONDecodeError, KeyError):
                continue

    def get_final_text(self) -> str:
        """Returns the full concatenated text after iteration is complete."""
        return "".join(c.delta for c in self._chunks)


# ─── Embedding Types ────────────────────────────────────────────────────────────

@dataclass
class Embedding:
    index: int
    embedding: List[float]
    object: str = "embedding"


@dataclass
class EmbeddingResponse:
    model: str
    data: List[Embedding]
    usage: Optional[Usage] = None

    @classmethod
    def from_dict(cls, data: dict) -> "EmbeddingResponse":
        embeddings = [
            Embedding(
                index=e.get("index", i),
                embedding=e.get("embedding", []),
                object=e.get("object", "embedding"),
            )
            for i, e in enumerate(data.get("data", []))
        ]

        usage_data = data.get("usage")
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=0,
            total_tokens=usage_data.get("total_tokens", 0),
        ) if usage_data else None

        return cls(
            model=data.get("model", ""),
            data=embeddings,
            usage=usage,
        )
