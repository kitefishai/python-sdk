"""
Tests for KiteFishAI Client initialisation and configuration.
"""

import pytest
import os
import kitefishai
from kitefishai._exceptions import AuthenticationError


class TestClientInit:

    def test_requires_api_key(self):
        env = os.environ.pop("KITEFISH_API_KEY", None)
        try:
            with pytest.raises(AuthenticationError, match="No API key"):
                kitefishai.Client()
        finally:
            if env:
                os.environ["KITEFISH_API_KEY"] = env

    def test_accepts_api_key_arg(self):
        client = kitefishai.Client(api_key="kf-test")
        assert client.api_key == "kf-test"

    def test_reads_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("KITEFISH_API_KEY", "kf-from-env")
        client = kitefishai.Client()
        assert client.api_key == "kf-from-env"

    def test_default_base_url(self):
        client = kitefishai.Client(api_key="kf-test")
        assert client.base_url == "https://api.kitefishai.com/v1"

    def test_custom_base_url(self):
        client = kitefishai.Client(api_key="kf-test", base_url="https://internal/v1")
        assert client.base_url == "https://internal/v1"

    def test_base_url_strips_trailing_slash(self):
        client = kitefishai.Client(api_key="kf-test", base_url="https://internal/v1/")
        assert client.base_url == "https://internal/v1"

    def test_reads_base_url_from_env(self, monkeypatch):
        monkeypatch.setenv("KITEFISH_BASE_URL", "https://onprem/v1")
        client = kitefishai.Client(api_key="kf-test")
        assert client.base_url == "https://onprem/v1"

    def test_default_timeout(self):
        client = kitefishai.Client(api_key="kf-test")
        assert client.timeout == 60.0

    def test_custom_timeout(self):
        client = kitefishai.Client(api_key="kf-test", timeout=120.0)
        assert client.timeout == 120.0

    def test_default_max_retries(self):
        client = kitefishai.Client(api_key="kf-test")
        assert client.max_retries == 2

    def test_resources_attached(self):
        client = kitefishai.Client(api_key="kf-test")
        assert hasattr(client, "chat")
        assert hasattr(client, "embeddings")

    def test_context_manager(self):
        with kitefishai.Client(api_key="kf-test") as client:
            assert client.api_key == "kf-test"

    def test_version_exported(self):
        assert kitefishai.__version__ == "0.1.0"
