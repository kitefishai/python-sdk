"""
Tests for client.embeddings — create()
"""

import pytest
import respx
import httpx

from tests.conftest import EMBEDDING_RESPONSE
import kitefishai
from kitefishai.types import EmbeddingResponse, Embedding


BASE = "https://api.kitefishai.com/v1"


class TestEmbeddings:

    @respx.mock
    def test_basic_create(self, client):
        respx.post(f"{BASE}/embeddings").mock(
            return_value=httpx.Response(200, json=EMBEDDING_RESPONSE)
        )
        result = client.embeddings.create(
            model="minnow-em-v1",
            input=["query: hello", "passage: world"],
        )
        assert isinstance(result, EmbeddingResponse)
        assert len(result.data) == 2
        assert isinstance(result.data[0], Embedding)

    @respx.mock
    def test_embedding_values(self, client):
        respx.post(f"{BASE}/embeddings").mock(
            return_value=httpx.Response(200, json=EMBEDDING_RESPONSE)
        )
        result = client.embeddings.create(
            model="minnow-em-v1",
            input=["query: test"],
        )
        assert result.data[0].embedding == [0.1, 0.2, 0.3]
        assert result.data[1].embedding == [0.4, 0.5, 0.6]

    @respx.mock
    def test_single_string_input_wrapped(self, client):
        captured = {}

        def capture(request):
            import json
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=EMBEDDING_RESPONSE)

        respx.post(f"{BASE}/embeddings").mock(side_effect=capture)
        client.embeddings.create(
            model="minnow-em-v1",
            input="query: single string",
        )
        assert isinstance(captured["body"]["input"], list)
        assert captured["body"]["input"] == ["query: single string"]

    @respx.mock
    def test_dimensions_forwarded(self, client):
        captured = {}

        def capture(request):
            import json
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=EMBEDDING_RESPONSE)

        respx.post(f"{BASE}/embeddings").mock(side_effect=capture)
        client.embeddings.create(
            model="minnow-em-v1",
            input=["query: test"],
            dimensions=256,
        )
        assert captured["body"]["dimensions"] == 256

    @respx.mock
    def test_dimensions_omitted_by_default(self, client):
        captured = {}

        def capture(request):
            import json
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=EMBEDDING_RESPONSE)

        respx.post(f"{BASE}/embeddings").mock(side_effect=capture)
        client.embeddings.create(
            model="minnow-em-v1",
            input=["query: test"],
        )
        assert "dimensions" not in captured["body"]

    @respx.mock
    def test_model_sent_in_payload(self, client):
        captured = {}

        def capture(request):
            import json
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=EMBEDDING_RESPONSE)

        respx.post(f"{BASE}/embeddings").mock(side_effect=capture)
        client.embeddings.create(model="minnow-em-v1", input=["test"])
        assert captured["body"]["model"] == "minnow-em-v1"

    @respx.mock
    def test_usage_parsed(self, client):
        respx.post(f"{BASE}/embeddings").mock(
            return_value=httpx.Response(200, json=EMBEDDING_RESPONSE)
        )
        result = client.embeddings.create(model="minnow-em-v1", input=["test"])
        assert result.usage.prompt_tokens == 8
        assert result.usage.total_tokens == 8

    @respx.mock
    def test_index_on_each_embedding(self, client):
        respx.post(f"{BASE}/embeddings").mock(
            return_value=httpx.Response(200, json=EMBEDDING_RESPONSE)
        )
        result = client.embeddings.create(model="minnow-em-v1", input=["a", "b"])
        assert result.data[0].index == 0
        assert result.data[1].index == 1

    @respx.mock
    def test_auth_error_on_401(self, client):
        respx.post(f"{BASE}/embeddings").mock(
            return_value=httpx.Response(401, json={"error": {"message": "Unauthorized"}})
        )
        with pytest.raises(kitefishai.AuthenticationError):
            client.embeddings.create(model="minnow-em-v1", input=["test"])

    @respx.mock
    def test_extra_params_forwarded(self, client):
        captured = {}

        def capture(request):
            import json
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=EMBEDDING_RESPONSE)

        respx.post(f"{BASE}/embeddings").mock(side_effect=capture)
        client.embeddings.create(
            model="minnow-em-v1",
            input=["test"],
            extra={"truncation": True},
        )
        assert captured["body"]["truncation"] is True
