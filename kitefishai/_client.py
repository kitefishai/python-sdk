"""
Client Python SDK — Client
"""

from __future__ import annotations

import os
import httpx
from typing import Optional

from kitefishai.resources import Chat
from ._exceptions import BaseError, AuthenticationError


DEFAULT_BASE_URL = "https://api.kitefishai.com/v1"
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 2


class Client:
    """
    KiteFishAI Python SDK client.

    Usage::

        import kitefishai

        client = kitefishai.Client(api_key="kf-...")

        # Chat (non-streaming)
        response = client.chat.complete(
            model="kf-reasoning-10b",
            messages=[{"role": "user", "content": "Hello"}],
        )
        print(response.choices[0].message.content)

        # Chat (streaming)
        with client.chat.stream(
            model="kf-reasoning-10b",
            messages=[{"role": "user", "content": "Hello"}],
        ) as stream:
            for chunk in stream:
                print(chunk.delta, end="", flush=True)

        # Embeddings
        result = client.embeddings.create(
            model="minnow-em-v1",
            input=["query: what is DPDP?"],
        )
        print(result.data[0].embedding)
    """

    chat: Chat

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("KITEFISH_API_KEY")
        if not resolved_key:
            raise AuthenticationError(
                "No API key provided. Pass api_key=... or set the "
                "KITEFISH_API_KEY environment variable."
            )

        self.api_key = resolved_key
        self.base_url = (base_url or os.environ.get("KITEFISH_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

        self._http_client = http_client or httpx.Client(
            timeout=httpx.Timeout(timeout),
            headers=self._default_headers(),
        )

        # Resources
        self.chat = Chat(self)


    def _default_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"kitefishai-python/0.1.0",
            "X-KiteFish-SDK": "python",
        }

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[dict] = None,
        stream: bool = False,
    ) -> httpx.Response:
        url = f"{self.base_url}{path}"
        headers = self._default_headers()

        for attempt in range(self.max_retries + 1):
            try:
                if stream:
                    return self._http_client.stream(
                        method, url, headers=headers, json=json
                    )
                response = self._http_client.request(
                    method, url, headers=headers, json=json
                )
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as e:
                _raise_api_error(e)
            except httpx.TimeoutException:
                if attempt == self.max_retries:
                    raise BaseError(
                        f"Request timed out after {self.timeout}s. "
                        "Try increasing the timeout parameter."
                    )
            except httpx.RequestError as e:
                if attempt == self.max_retries:
                    raise BaseError(f"Request failed: {e}") from e

    def close(self) -> None:
        self._http_client.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *args) -> None:
        self.close()


def _raise_api_error(e: httpx.HTTPStatusError) -> None:
    from ._exceptions import (
        AuthenticationError,
        RateLimitError,
        APIError,
        NotFoundError,
    )

    status = e.response.status_code
    try:
        body = e.response.json()
        message = body.get("error", {}).get("message") or str(body)
    except Exception:
        message = e.response.text or str(e)

    if status == 401:
        raise AuthenticationError(message) from e
    elif status == 429:
        raise RateLimitError(message) from e
    elif status == 404:
        raise NotFoundError(message) from e
    else:
        raise APIError(message, status_code=status) from e
