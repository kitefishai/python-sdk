"""
Tests for client.chat — complete() and stream()
"""

import pytest
import respx
import httpx

from tests.conftest import CHAT_RESPONSE, STREAM_CHUNKS
import kitefishai
from kitefishai.types import ChatCompletion, StreamChunk


BASE = "https://api.kitefishai.com/v1"


class TestChatComplete:

    @respx.mock
    def test_basic_complete(self, client):
        respx.post(f"{BASE}/chat").mock(
            return_value=httpx.Response(200, json=CHAT_RESPONSE)
        )
        response = client.chat.complete(
            model="kf-reasoning-10b",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content == "Hello from KiteFish!"
        assert response.choices[0].message.role == "assistant"
        assert response.choices[0].finish_reason == "stop"

    @respx.mock
    def test_usage_parsed(self, client):
        respx.post(f"{BASE}/chat").mock(
            return_value=httpx.Response(200, json=CHAT_RESPONSE)
        )
        response = client.chat.complete(
            model="kf-reasoning-10b",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5
        assert response.usage.total_tokens == 15

    @respx.mock
    def test_system_prompt_prepended(self, client):
        captured = {}

        def capture(request):
            import json
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=CHAT_RESPONSE)

        respx.post(f"{BASE}/chat").mock(side_effect=capture)
        client.chat.complete(
            model="kf-reasoning-10b",
            system="You are a compliance assistant.",
            messages=[{"role": "user", "content": "Hello"}],
        )
        messages = captured["body"]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a compliance assistant."
        assert messages[1]["role"] == "user"

    @respx.mock
    def test_stream_false_by_default(self, client):
        captured = {}

        def capture(request):
            import json
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=CHAT_RESPONSE)

        respx.post(f"{BASE}/chat").mock(side_effect=capture)
        client.chat.complete(
            model="kf-reasoning-10b",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert captured["body"]["stream"] is False

    @respx.mock
    def test_model_sent_in_payload(self, client):
        captured = {}

        def capture(request):
            import json
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=CHAT_RESPONSE)

        respx.post(f"{BASE}/chat").mock(side_effect=capture)
        client.chat.complete(
            model="kf-reasoning-10b",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert captured["body"]["model"] == "kf-reasoning-10b"

    @respx.mock
    def test_extra_params_forwarded(self, client):
        captured = {}

        def capture(request):
            import json
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=CHAT_RESPONSE)

        respx.post(f"{BASE}/chat").mock(side_effect=capture)
        client.chat.complete(
            model="kf-reasoning-10b",
            messages=[{"role": "user", "content": "Hi"}],
            extra={"frequency_penalty": 0.5},
        )
        assert captured["body"]["frequency_penalty"] == 0.5

    @respx.mock
    def test_auth_error_on_401(self, client):
        respx.post(f"{BASE}/chat").mock(
            return_value=httpx.Response(401, json={"error": {"message": "Unauthorized"}})
        )
        with pytest.raises(kitefishai.AuthenticationError):
            client.chat.complete(
                model="kf-reasoning-10b",
                messages=[{"role": "user", "content": "Hi"}],
            )

    @respx.mock
    def test_rate_limit_error_on_429(self, client):
        respx.post(f"{BASE}/chat").mock(
            return_value=httpx.Response(429, json={"error": {"message": "Rate limited"}})
        )
        with pytest.raises(kitefishai.RateLimitError):
            client.chat.complete(
                model="kf-reasoning-10b",
                messages=[{"role": "user", "content": "Hi"}],
            )

    @respx.mock
    def test_api_error_on_500(self, client):
        respx.post(f"{BASE}/chat").mock(
            return_value=httpx.Response(500, json={"error": {"message": "Server error"}})
        )
        with pytest.raises(kitefishai.APIError) as exc_info:
            client.chat.complete(
                model="kf-reasoning-10b",
                messages=[{"role": "user", "content": "Hi"}],
            )
        assert exc_info.value.status_code == 500


class TestChatStream:

    def test_stream_returns_chat_stream(self, client):
        from kitefishai.types import ChatStream
        result = client.chat.stream(
            model="kf-reasoning-10b",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert isinstance(result, ChatStream)

    def test_stream_chunk_parsing(self):
        import json
        from kitefishai.types import StreamChunk

        raw = '{"id":"r1","model":"kf-reasoning-10b","choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}'
        chunk = StreamChunk.from_dict(json.loads(raw))
        assert chunk.delta == "Hello"
        assert chunk.finish_reason is None
        assert chunk.model == "kf-reasoning-10b"

    def test_stream_chunk_finish_reason(self):
        import json
        from kitefishai.types import StreamChunk

        raw = '{"id":"r1","model":"kf-reasoning-10b","choices":[{"delta":{"content":""},"finish_reason":"stop"}]}'
        chunk = StreamChunk.from_dict(json.loads(raw))
        assert chunk.finish_reason == "stop"

    def test_stream_empty_delta(self):
        import json
        from kitefishai.types import StreamChunk

        raw = '{"id":"r1","model":"kf-reasoning-10b","choices":[{"delta":{},"finish_reason":null}]}'
        chunk = StreamChunk.from_dict(json.loads(raw))
        assert chunk.delta == ""
