"""Pydantic v2 request schemas for the web subsystem.

Defines a ``SafeUrl`` annotated type that validates URLs through
:func:`pkb_agent.security.url_safety.assert_safe_url`, plus request models
for fetch and search operations.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field
from pydantic.functional_validators import AfterValidator

from pkb_agent.security.url_safety import assert_safe_url


def _validate_safe_url(value: str) -> str:
    """Pydantic validator: raise ``UnsafeUrlError`` for unsafe URLs."""
    return assert_safe_url(value)


# A string URL that must pass SSRF safety checks.
SafeUrl = Annotated[str, AfterValidator(_validate_safe_url)]


class FetchRequest(BaseModel):
    """Request body for fetching a single URL."""

    url: SafeUrl
    max_chars: int = Field(default=8000, ge=100, le=20000)


class SearchRequest(BaseModel):
    """Request body for a web search query."""

    query: str = Field(min_length=1)
    max_results: int = Field(default=5, ge=1, le=10)
