from __future__ import annotations

import pytest

from pkb_agent.agent.errors import UnsafeUrlError
from pkb_agent.security.url_safety import (
    assert_safe_url,
    is_private_host,
    is_safe_url,
    normalize_url,
)

# ---- is_safe_url: happy path ----------------------------------------------


@pytest.mark.parametrize("url", ["https://example.com", "http://example.com/path"])
def test_safe_urls_pass(url):
    ok, reason = is_safe_url(url)
    assert ok is True
    assert reason == "ok"


def test_safe_url_with_port():
    ok, _ = is_safe_url("https://example.com:8080/x")
    assert ok is True


# ---- empty / malformed / scheme -------------------------------------------


@pytest.mark.parametrize("url", ["", "   ", None])
def test_empty_url_blocked(url):
    ok, reason = is_safe_url(url)  # type: ignore[arg-type]
    assert ok is False
    assert "empty" in reason


def test_blocked_scheme_file():
    ok, reason = is_safe_url("file:///etc/passwd")
    assert ok is False
    assert "scheme" in reason


def test_custom_allowed_schemes():
    ok, _ = is_safe_url("ftp://example.com", allowed_schemes={"ftp"})
    assert ok is True


# ---- private / reserved IPs -----------------------------------------------


@pytest.mark.parametrize(
    "ip",
    ["127.0.0.1", "10.0.0.1", "192.168.1.1", "172.16.0.1", "169.254.169.254", "0.0.0.0"],
)
def test_private_ip_blocked(ip):
    ok, _ = is_safe_url(f"http://{ip}/")
    assert ok is False


def test_ipv6_loopback_blocked():
    ok, _ = is_safe_url("http://[::1]/")
    assert ok is False


def test_localhost_blocked():
    ok, _ = is_safe_url("http://localhost/")
    assert ok is False


def test_block_private_disabled_allows_private_ip():
    ok, _ = is_safe_url("http://10.0.0.1/", block_private=False)
    assert ok is True


# ---- is_private_host -------------------------------------------------------


@pytest.mark.parametrize(
    "host,expected",
    [
        ("example.com", False),
        ("8.8.8.8", False),
        ("10.1.2.3", True),
        ("127.0.0.1", True),
        ("169.254.1.1", True),
        ("", False),
    ],
)
def test_is_private_host(host, expected):
    assert is_private_host(host) is expected


# ---- userinfo / suffixes --------------------------------------------------


def test_userinfo_in_netloc_blocked():
    ok, reason = is_safe_url("http://user:pass@example.com/")
    assert ok is False
    assert "userinfo" in reason


@pytest.mark.parametrize("host", ["host.internal", "host.local"])
def test_internal_suffix_blocked(host):
    ok, reason = is_safe_url(f"http://{host}/")
    assert ok is False
    assert "suffix" in reason


# ---- assert_safe_url -------------------------------------------------------


def test_assert_safe_url_returns_url():
    url = "https://example.com"
    assert assert_safe_url(url) == url


def test_assert_safe_url_raises():
    with pytest.raises(UnsafeUrlError):
        assert_safe_url("http://localhost/")


# ---- normalize_url --------------------------------------------------------


@pytest.mark.parametrize(
    "url,expected",
    [
        ("example.com", "https://example.com"),
        ("http://example.com", "http://example.com"),
        ("  https://example.com  ", "https://example.com"),
        ("", ""),
        ("localhost", "localhost"),
    ],
)
def test_normalize_url(url, expected):
    assert normalize_url(url) == expected
