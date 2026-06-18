"""
tools/escalation_tool.py
Formatting and utility functions for escalating clinical flags to clinicians.
"""
from __future__ import annotations
from typing import Optional

def format_clinician_flag(severity: str, field: str, message: str, source_evidence: Optional[str] = None) -> str:
    """Format a clinical concern into a standardized flag string with severity icons."""
    icon = {"critical": "🚨", "high": "⚠️", "informational": "ℹ️"}.get(severity.lower(), "ℹ️")
    formatted = f"{icon} [{severity.upper()}] {field}: {message}"
    if source_evidence:
        formatted += f" (Source: {source_evidence})"
    return formatted
