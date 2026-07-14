"""Calculator tool: safe arithmetic evaluation."""

from __future__ import annotations

import ast
import operator
from typing import Any

from pkb_agent.tools.base import BaseTool, ToolContext, ToolParam
from pkb_agent.tools.result import ToolResult

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: lambda x: +x, ast.USub: lambda x: -x}
_MAX_POW_EXPONENT = 1000


class CalculatorTool(BaseTool):
    name = "calculator"
    description = (
        "Evaluate a numeric arithmetic expression (supports + - * / // % ** and "
        "parentheses). No variables or function calls. Use for precise math."
    )
    params = [
        ToolParam("expression", "string", "The arithmetic expression to evaluate."),
    ]
    permission = "allow"

    async def execute(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        expr = str(args.get("expression", "")).strip()
        if not expr:
            return ToolResult.failure("expression must not be empty")
        try:
            value = _safe_eval(expr)
        except ZeroDivisionError:
            return ToolResult.failure("division by zero")
        except (ValueError, SyntaxError, TypeError, RecursionError) as e:
            return ToolResult.failure(f"invalid expression: {e}")
        except Exception as e:  # pragma: no cover - defensive
            return ToolResult.failure(f"calculator failed: {e}")

        return ToolResult.success(
            {"expression": expr, "result": value, "type": type(value).__name__}
        )


def _safe_eval(expr: str) -> int | float:
    tree = ast.parse(expr, mode="eval")
    return _eval_node(tree.body)


def _eval_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numeric constants are allowed")
        return node.value
    if isinstance(node, ast.BinOp):
        bin_op = _BIN_OPS.get(type(node.op))
        if bin_op is None:
            raise ValueError(f"operator not allowed: {type(node.op).__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if bin_op is operator.pow and abs(right) > _MAX_POW_EXPONENT:
            raise ValueError("exponent too large")
        return bin_op(left, right)
    if isinstance(node, ast.UnaryOp):
        unary_op = _UNARY_OPS.get(type(node.op))
        if unary_op is None:
            raise ValueError(f"unary operator not allowed: {type(node.op).__name__}")
        return unary_op(_eval_node(node.operand))
    raise ValueError(f"expression element not allowed: {type(node).__name__}")
