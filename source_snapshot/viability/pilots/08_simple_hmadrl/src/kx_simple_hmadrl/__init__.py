"""Simple, causally separated FRED/GC-trade/GC-L2 research pilot."""

from .contracts import ExecutionCommand, MacroMessage, TargetMessage
from .splits import Split, classify_time, require_development_access

__all__ = [
    "ExecutionCommand", "MacroMessage", "TargetMessage",
    "Split", "classify_time", "require_development_access",
]
