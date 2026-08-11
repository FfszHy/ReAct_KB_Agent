"""Builtin tool implementations."""

from __future__ import annotations

from pkb_agent.tools.base import BaseTool
from pkb_agent.tools.builtin.calculator import CalculatorTool
from pkb_agent.tools.builtin.memory_search import MemorySearchTool
from pkb_agent.tools.builtin.memory_write import MemoryWriteTool
from pkb_agent.tools.builtin.now import NowTool
from pkb_agent.tools.builtin.rag_list_documents import RagListDocumentsTool
from pkb_agent.tools.builtin.rag_read import RagReadTool
from pkb_agent.tools.builtin.rag_search import RagSearchTool
from pkb_agent.tools.builtin.web_fetch import WebFetchTool
from pkb_agent.tools.builtin.web_search import WebSearchTool

__all__ = [
    "CalculatorTool",
    "MemorySearchTool",
    "MemoryWriteTool",
    "NowTool",
    "RagListDocumentsTool",
    "RagReadTool",
    "RagSearchTool",
    "WebFetchTool",
    "WebSearchTool",
    "build_builtin_tools",
]


def build_builtin_tools() -> list[BaseTool]:
    """Instantiate the full default tool set."""
    return [
        RagListDocumentsTool(),
        RagSearchTool(),
        RagReadTool(),
        WebSearchTool(),
        WebFetchTool(),
        MemorySearchTool(),
        MemoryWriteTool(),
        CalculatorTool(),
        NowTool(),
    ]
