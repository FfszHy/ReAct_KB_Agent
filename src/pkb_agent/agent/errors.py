"""Centralized exception hierarchy for PKB-Agent."""

from __future__ import annotations


class PKBError(Exception):
    """Base error for all pkb-agent failures."""


# ---- Configuration / runtime -------------------------------------------------


class ConfigError(PKBError):
    """Raised when required configuration is missing or invalid."""


class NotInitializedError(PKBError):
    """Raised when a dependency (client, repository, ...) is not initialized."""


# ---- LLM ---------------------------------------------------------------------


class LLMError(PKBError):
    """Raised when the DeepSeek API call fails."""


class LLMRateLimitError(LLMError):
    """Raised when the LLM provider returns a rate-limit / overloaded response."""


class LLMResponseError(LLMError):
    """Raised when the LLM response is malformed or unusable."""


# ---- Agent -------------------------------------------------------------------


class AgentError(PKBError):
    """Raised on agent-loop level failures."""


class MaxStepsError(AgentError):
    """Raised when the agent exceeds the configured step budget."""


class AgentAbortError(AgentError):
    """Raised when the agent run is aborted (e.g. user interrupt)."""


# ---- Tools -------------------------------------------------------------------


class ToolError(PKBError):
    """Base error for tool-layer failures."""


class ToolNotFoundError(ToolError):
    def __init__(self, name: str):
        super().__init__(f"tool not found: {name}")
        self.name = name


class ToolArgumentError(ToolError):
    def __init__(self, tool: str, param: str, reason: str):
        super().__init__(f"[{tool}] invalid argument '{param}': {reason}")
        self.tool = tool
        self.param = param
        self.reason = reason


class ToolPermissionDenied(ToolError):
    def __init__(self, tool: str, reason: str):
        super().__init__(f"[{tool}] permission denied: {reason}")
        self.tool = tool
        self.reason = reason


class ToolExecutionError(ToolError):
    def __init__(self, tool: str, reason: str):
        super().__init__(f"[{tool}] execution failed: {reason}")
        self.tool = tool
        self.reason = reason


# ---- Storage -----------------------------------------------------------------


class StorageError(PKBError):
    """Raised on Supabase / repository failures."""


# ---- RAG ---------------------------------------------------------------------


class EmbeddingError(PKBError):
    """Raised when the embedding API fails."""


# ---- Web / Security ----------------------------------------------------------


class WebError(PKBError):
    """Raised on web search/fetch failures."""


class WebFetchError(WebError):
    pass


class SecurityError(PKBError):
    """Raised when a security policy is violated."""


class UnsafeUrlError(SecurityError):
    def __init__(self, url: str, reason: str):
        super().__init__(f"unsafe URL {url!r}: {reason}")
        self.url = url
        self.reason = reason
