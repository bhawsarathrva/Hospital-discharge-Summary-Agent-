"""
tools/escalation.py
Flag-for-clinician-review action tool.
The agent calls this when it finds something it cannot safely resolve autonomously:
  - High-severity drug interactions
  - Conflicting critical values
  - Missing mandatory fields
  - Medications changed without documented reason
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from config.settings import SETTINGS
from tools.base import BaseTool, ToolResult, ToolStatus


class EscalationSeverity(str, Enum):
    CRITICAL = "critical"        # Potential patient harm if ignored
    HIGH = "high"                # Needs clinician attention before finalisation
    INFORMATIONAL = "informational"  # FYI — clinician should be aware


@dataclass
class EscalationRecord:
    severity: EscalationSeverity
    field: str
    message: str
    source_evidence: Optional[str] = None
    auto_formatted: str = ""     # Pre-formatted flag string for summary

    def __post_init__(self):
        icon = {"critical": "🚨", "high": "⚠️", "informational": "ℹ️"}[self.severity.value]
        self.auto_formatted = (
            f"{icon} [{self.severity.value.upper()}] {self.field}: {self.message}"
            + (f" (Source: {self.source_evidence})" if self.source_evidence else "")
        )

    def to_dict(self) -> dict:
        return {
            "severity": self.severity.value,
            "field": self.field,
            "message": self.message,
            "source_evidence": self.source_evidence,
            "auto_formatted": self.auto_formatted,
        }


class EscalationTool(BaseTool):
    """
    Records clinician flags during the agent run.
    All flags are persisted in the summary's clinician_flags list.
    The agent calls this instead of guessing or silently ignoring issues.
    """
    name = "escalate"
    description = (
        "Flag an issue for clinician review. Call this when: "
        "(1) a required field cannot be sourced from documents, "
        "(2) a drug interaction is HIGH severity, "
        "(3) two notes conflict on a clinical fact, "
        "(4) a medication was changed without documented reason. "
        "Never skip calling this — always surface uncertainty."
    )

    def __init__(self):
        self._records: List[EscalationRecord] = []

    def _run(
        self,
        severity: str,
        field: str,
        message: str,
        source_evidence: Optional[str] = None,
    ) -> ToolResult:
        try:
            sev = EscalationSeverity(severity.lower())
        except ValueError:
            sev = EscalationSeverity.INFORMATIONAL

        record = EscalationRecord(
            severity=sev,
            field=field,
            message=message,
            source_evidence=source_evidence,
        )
        self._records.append(record)

        return ToolResult(
            status=ToolStatus.SUCCESS,
            data=record.to_dict(),
            metadata={"total_flags": len(self._records)},
        )

    def get_all_flags(self) -> List[str]:
        """Return formatted flag strings for embedding in summary."""
        return [r.auto_formatted for r in self._records]

    def get_critical_flags(self) -> List[EscalationRecord]:
        return [r for r in self._records if r.severity == EscalationSeverity.CRITICAL]

    def reset(self) -> None:
        self._records.clear()

    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "severity": {
                    "type": "string",
                    "enum": ["critical", "high", "informational"],
                    "description": "Severity of the flag",
                    "required": True,
                },
                "field": {
                    "type": "string",
                    "description": "Which summary field this flag relates to",
                    "required": True,
                },
                "message": {
                    "type": "string",
                    "description": "Clear description of the issue",
                    "required": True,
                },
                "source_evidence": {
                    "type": "string",
                    "description": "Which document/note triggered this flag",
                    "required": False,
                },
            },
        }
