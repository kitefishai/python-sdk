"""
Client SDK — Chat Resource
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Union

from ..types import ChatCompletion, ChatStream

if TYPE_CHECKING:
    from .._client import Client


class Chat:
    """
    Access Client chat/reasoning models.

    Non-streaming::

        response = client.chat.complete(
            model="kf-reasoning-10b",
            messages=[{"role": "user", "content": "Summarise this policy."}],
            max_tokens=512,
        )
        print(response.choices[0].message.content)

    Streaming::

        with client.chat.stream(
            model="kf-reasoning-10b",
            messages=[{"role": "user", "content": "Explain DPDP Act."}],
        ) as stream:
            for chunk in stream:
                print(chunk.delta, end="", flush=True)
    """

    def __init__(self, client: "Client") -> None:
        self._client = client

    def complete(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 1.0,
        system: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> ChatCompletion:
        """
        Send a chat completion request and return the full response.

        Args:
            model:       Model ID, e.g. ``"kf-reasoning-10b"``.
            messages:    List of ``{"role": ..., "content": ...}`` dicts.
            max_tokens:  Maximum tokens to generate.
            temperature: Sampling temperature (0.0 – 2.0).
            top_p:       Nucleus sampling probability.
            system:      Optional system prompt (prepended automatically).
            extra:       Any additional parameters passed through to the API.

        Returns:
            :class:`~kitefishai.types.ChatCompletion`
        """
        payload = self._build_payload(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            system=system,
            stream=False,
            extra=extra,
        )
        response = self._client._request("POST", "/chat", json=payload)
        return ChatCompletion.from_dict(response.json())

    def stream(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 1.0,
        system: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> ChatStream:
        """
        Send a streaming chat request.

        Returns a context-manager :class:`~kitefishai.types.ChatStream`.
        Iterate over it to receive :class:`~kitefishai.types.StreamChunk` objects.

        Example::

            with client.chat.stream(model="kf-reasoning-10b", messages=[...]) as s:
                for chunk in s:
                    print(chunk.delta, end="")
            print()  # newline after stream

        After the ``with`` block, call ``stream.get_final_text()`` for the full string.
        """
        payload = self._build_payload(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            system=system,
            stream=True,
            extra=extra,
        )
        ctx = self._client._request("POST", "/chat", json=payload, stream=True)
        return ChatStream(ctx)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _build_payload(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        system: Optional[str],
        stream: bool,
        extra: Optional[dict],
    ) -> dict:
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        payload = {
            "model": model,
            "messages": all_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }
        if extra:
            payload.update(extra)
        return payload
