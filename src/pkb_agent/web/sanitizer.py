"""HTML sanitization and text extraction.

Converts raw HTML into readable plain text for the agent. Removes script/style
and navigation chrome, prefers ``<article>`` / ``<main>`` content, and collapses
whitespace.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup, FeatureNotFound

_UNWANTED_TAGS: tuple[str, ...] = (
    "script",
    "style",
    "nav",
    "footer",
    "header",
    "aside",
    "noscript",
)

_WS_RE = re.compile(r"\s+")


def _make_soup(html: str) -> BeautifulSoup:
    """Build a BeautifulSoup tree, preferring lxml and falling back to stdlib."""
    try:
        return BeautifulSoup(html, "lxml")
    except FeatureNotFound:
        return BeautifulSoup(html, "html.parser")


def sanitize_html(html: str, base_url: str | None = None) -> tuple[str, str | None]:
    """Return ``(main_text, title)`` extracted from ``html``.

    Removes unwanted tags, prefers ``<article>``/``<main>``/``<body>`` content,
    and collapses runs of whitespace into single spaces.
    """
    if not isinstance(html, str) or not html:
        return "", None

    soup = _make_soup(html)

    for tag_name in _UNWANTED_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    title: str | None = None
    if soup.title is not None and soup.title.string:
        title = soup.title.string.strip() or None

    main = soup.find("article") or soup.find("main") or soup.body or soup
    text = main.get_text(separator=" ", strip=True)
    text = _WS_RE.sub(" ", text).strip()

    return text, title


def strip_html(html: str) -> str:
    """Convenience wrapper returning only the main text."""
    return sanitize_html(html)[0]


def truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    """Truncate ``text`` to ``max_chars``; return ``(text, truncated)``."""
    if not isinstance(text, str) or len(text) <= max_chars:
        return text, False
    return text[: max(max_chars - 12, 0)].rstrip() + "…[truncated]", True
