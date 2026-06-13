"""
Client SDK — Embeddings Resource
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Union

from ..types import EmbeddingResponse

if TYPE_CHECKING:
    from .._client import Client


class Embeddings:
    """
    Access Client embedding models (Minnow-Em series).

    Example::

        result = client.embeddings.create(
            model="minnow-em-v1",
            input=["query: What is DPDP Act?", "passage: The DPDP Act 2023..."],
        )
        for item in result.data:
            print(item.index, len(item.embedding))

    MRL dimensions::

        # Request a smaller embedding dimension (512, 256, 128, 64)
        result = client.embeddings.create(
            model="minnow-em-v1",
            input=["query: insurance claim process"],
            dimensions=256,
        )
    """

    def __init__(self, client: "Client") -> None:
        self._client = client

    def create(
        self,
        *,
        model: str,
        input: Union[str, List[str]],
        dimensions: Optional[int] = None,
        encoding_format: str = "float",
        extra: Optional[dict] = None,
    ) -> EmbeddingResponse:
        """
        Generate embeddings for the given input text(s).

        Args:
            model:           Embedding model ID, e.g. ``"minnow-em-v1"``.
            input:           A single string or list of strings.
            dimensions:      Optional MRL dimension (896/512/256/128/64).
                             Defaults to the model's native dimension (896).
            encoding_format: ``"float"`` (default) or ``"base64"``.
            extra:           Any additional parameters passed through to the API.

        Returns:
            :class:`~kitefishai.types.EmbeddingResponse`
        """
        inputs = [input] if isinstance(input, str) else input

        payload: dict = {
            "model": model,
            "input": inputs,
            "encoding_format": encoding_format,
        }
        if dimensions is not None:
            payload["dimensions"] = dimensions
        if extra:
            payload.update(extra)

        response = self._client._request("POST", "/embeddings", json=payload)
        return EmbeddingResponse.from_dict(response.json())
