"""
Client Python SDK
~~~~~~~~~~~~~~~~~~~~~

Sovereign AI for regulated Indian enterprises.

Basic usage::

    import kitefishai

    client = kitefishai.Client(api_key="kf-...")

    # Chat
    response = client.chat.complete(
        model="kf-reasoning-10b",
        messages=[{"role": "user", "content": "Hello"}],
    )
    print(response.choices[0].message.content)

    # Streaming
    with client.chat.stream(model="kf-reasoning-10b", messages=[...]) as stream:
        for chunk in stream:
            print(chunk.delta, end="", flush=True)

    # Embeddings
    result = client.embeddings.create(
        model="minnow-em-v1",
        input=["query: DPDP compliance requirements"],
    )

Full docs: https://docs.kitefishai.com
"""

from ._client import Client
from ._exceptions import (
    BaseError,
    AuthenticationError,
    RateLimitError,
    NotFoundError,
    APIError,
)
from .types import (
    ChatCompletion,
    ChatStream,
    StreamChunk,
    EmbeddingResponse,
    Embedding,
    Message,
    Choice,
    Usage,
)

__version__ = "0.1.0"

__all__ = [
    "Client",
    "BaseError",
    "AuthenticationError",
    "RateLimitError",
    "NotFoundError",
    "APIError",
    "ChatCompletion",
    "ChatStream",
    "StreamChunk",
    "EmbeddingResponse",
    "Embedding",
    "Message",
    "Choice",
    "Usage",
]
