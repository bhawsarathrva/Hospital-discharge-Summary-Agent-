"""
tools/base.py
Base classes for all agent tools.
Every tool returns a ToolResult — never raises unhandled exceptions.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ToolStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"  # Some data returned but incomplete
    FAILED = "failed"  # No usable data returned
    TIMEOUT = "timeout"
    NOT_FOUND = "not_found"


@dataclass
class ToolResult:
    status: ToolStatus
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status in (ToolStatus.SUCCESS, ToolStatus.PARTIAL)

    @property
    def usable(self) -> bool:
        """True if the result contains any data worth acting on."""
        return self.ok and self.data is not None

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "data": self.data if not isinstance(self.data, bytes) else "<binary>",
            "error": self.error,
            "metadata": self.metadata,
            "duration_s": round(self.duration_s, 3),
        }


class BaseTool(ABC):
    """
    All tools must inherit from this.
    Tools are responsible for their own error handling — they return
    ToolResult(FAILED) rather than raising exceptions.
    """

    name: str = "base_tool"
    description: str = ""

    def run(self, **kwargs) -> ToolResult:
        """
        Public interface. Wraps _run with timing and exception catch-all.
        """
        t0 = time.time()
        try:
            result = self._run(**kwargs)
        except Exception as exc:  # noqa: BLE001
            result = ToolResult(
                status=ToolStatus.FAILED,
                error=f"{type(exc).__name__}: {exc}",
            )
        result.duration_s = time.time() - t0
        return result

    @abstractmethod
    def _run(self, **kwargs) -> ToolResult:
        """Implement the actual tool logic here."""
        ...

    def schema(self) -> Dict[str, Any]:
        """Return JSON-schema-compatible description for the LLM."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {},
        }
