"""Small allowlist for corpus source provenance shown to the model."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse


def source_provenance(metadata: Any) -> dict[str, str]:
    """Copy explicit upstream fields without inferring a version or source.

    Other document metadata can contain private fields or free-form text and
    must not be copied into tool observations or semantic-review prompts.
    """
    if not isinstance(metadata, Mapping):
        return {}
    result: dict[str, str] = {}
    url = metadata.get("upstream_url")
    if isinstance(url, str) and len(url) <= 4096 and not any(char.isspace() for char in url):
        try:
            parsed = urlparse(url)
            if (
                parsed.scheme in {"http", "https"}
                and parsed.hostname
                and parsed.username is None
                and parsed.password is None
            ):
                result["upstream_url"] = url
        except ValueError:
            pass
    revision = metadata.get("upstream_revision")
    if isinstance(revision, str) and re.fullmatch(r"[A-Za-z0-9._/+:\-]{1,256}", revision):
        result["upstream_revision"] = revision
    digest = metadata.get("upstream_sha256")
    if isinstance(digest, str) and re.fullmatch(r"[A-Fa-f0-9]{64}", digest):
        result["upstream_sha256"] = digest
    return result
