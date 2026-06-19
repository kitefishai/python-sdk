"""
Tests for exceptions and response type parsing.
"""

import pytest
from kitefishai._exceptions import (
    KiteFishAIError,
    AuthenticationError,
    RateLimitError,
    NotFoundError,
    APIError,
)
from kitefishai.types import (
    ChatCompletion,
    EmbeddingResponse,
    StreamChunk,
    Usage,
    Message,
    Choice,
    Embedding,
)


class TestExceptions:

    def test_hierarchy(self):
        assert issubclass(AuthenticationError, KiteFishAIError)
        assert issubclass(RateLimitError, KiteFishAIError)
        assert issubclass(NotFoundError, KiteFishAIError)
        assert issubclass(APIError, KiteFishAIError)

    def test_api_error_status_code(self):
        err = APIError("Something failed", status_code=503)
        assert err.status_code == 503
        assert "503" in repr(err)

    def test_message_accessible(self):
        err = KiteFishAIError("test message")
        assert err.message == "test message"
        assert str(err) == "test message"


class TestChatCompletionParsing:

    def test_full_response(self):
        data = {
            "id": "req_1",
            "model": "kf-reasoning-10b",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            "created": 1700000000,
        }
        cc = ChatCompletion.from_dict(data)
        assert cc.id == "req_1"
        assert cc.model == "kf-reasoning-10b"
        assert len(cc.choices) == 1
        assert cc.choices[0].message.content == "Hello!"
        assert cc.choices[0].message.role == "assistant"
        assert cc.choices[0].finish_reason == "stop"
        assert cc.usage.total_tokens == 8
        assert cc.created == 1700000000

    def test_missing_usage(self):
        data = {
            "id": "req_2",
            "model": "kf-reasoning-10b",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": None}
            ],
        }
        cc = ChatCompletion.from_dict(data)
        assert cc.usage is None

    def test_empty_choices(self):
        cc = ChatCompletion.from_dict({"id": "", "model": "", "choices": []})
        assert cc.choices == []


class TestStreamChunkParsing:

    def test_delta_extracted(self):
        data = {
            "id": "r1",
            "model": "kf-reasoning-10b",
            "choices": [{"delta": {"content": "Hello"}, "finish_reason": None}],
        }
        chunk = StreamChunk.from_dict(data)
        assert chunk.delta == "Hello"
        assert chunk.finish_reason is None

    def test_empty_delta_on_missing_content(self):
        data = {
            "id": "r1",
            "model": "kf-reasoning-10b",
            "choices": [{"delta": {}, "finish_reason": "stop"}],
        }
        chunk = StreamChunk.from_dict(data)
        assert chunk.delta == ""
        assert chunk.finish_reason == "stop"

    def test_no_choices(self):
        data = {"id": "r1", "model": "kf-reasoning-10b", "choices": []}
        chunk = StreamChunk.from_dict(data)
        assert chunk.delta == ""


class TestEmbeddingResponseParsing:

    def test_full_response(self):
        data = {
            "model": "minnow-em-v1",
            "data": [
                {"index": 0, "embedding": [0.1, 0.2], "object": "embedding"},
                {"index": 1, "embedding": [0.3, 0.4], "object": "embedding"},
            ],
            "usage": {"prompt_tokens": 4, "total_tokens": 4},
        }
        er = EmbeddingResponse.from_dict(data)
        assert er.model == "minnow-em-v1"
        assert len(er.data) == 2
        assert er.data[0].embedding == [0.1, 0.2]
        assert er.data[1].index == 1
        assert er.usage.prompt_tokens == 4

    def test_missing_usage(self):
        data = {
            "model": "minnow-em-v1",
            "data": [{"index": 0, "embedding": [0.1]}],
        }
        er = EmbeddingResponse.from_dict(data)
        assert er.usage is None

    def test_empty_data(self):
        er = EmbeddingResponse.from_dict({"model": "minnow-em-v1", "data": []})
        assert er.data == []
