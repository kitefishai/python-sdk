"""
Client SDK Exceptions
"""

from __future__ import annotations
from typing import Optional


class BaseError(Exception):
    """Base exception for all Client SDK errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class AuthenticationError(BaseError):
    """Raised when the API key is missing or invalid (HTTP 401)."""


class RateLimitError(BaseError):
    """Raised when you've exceeded your rate limit (HTTP 429)."""


class NotFoundError(BaseError):
    """Raised when the requested resource is not found (HTTP 404)."""


class APIError(BaseError):
    """Raised for any other non-2xx API response."""

    def __init__(self, message: str, *, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code

    def __repr__(self) -> str:
        return f"APIError(status_code={self.status_code}, message={self.message!r})"
