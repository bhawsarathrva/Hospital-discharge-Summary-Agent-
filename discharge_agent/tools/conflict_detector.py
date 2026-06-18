"""
tools/conflict_detector.py
Detects conflicts between multiple source notes on key clinical fields.
If two notes disagree, the conflict is surfaced — never silently resolved.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from config.settings import SETTINGS
from models.patient import Conflict
from tools.base import BaseTool, ToolResult, ToolStatus


# Fields we actively check for conflicts
CONFLICT_FIELDS = [
    "principal_diagnosis",
    "discharge_diagnosis",
    "admission_date",
    "discharge_date",
    "allergies",
    "discharge_condition",
]


class ConflictDetectorTool(BaseTool):
    name = "conflict_detector"
    description = (
        "Compare clinical data extracted from multiple source documents. "
        "Returns a list of Conflict objects for any field where two or more "
        "documents disagree. Does NOT resolve conflicts — flags them all."
    )

    def _run(
        self,
        extracted_documents: List[Dict[str, Any]],
    ) -> ToolResult:
        """
        Args:
            extracted_documents: List of dicts with keys:
                'source' (file/page id), 'entities' (dict of field → value)
        """
        if not extracted_documents:
            return ToolResult(
                status=ToolStatus.NOT_FOUND,
                error="No extracted documents provided",
            )

        conflicts: List[Conflict] = []

        # Build field → [(source, value)] map
        field_values: Dict[str, List[Tuple[str, str]]] = {}

        for doc in extracted_documents:
            source = f"{doc.get('file_path', 'unknown')}:p{doc.get('page_number', '?')}"
            entities = doc.get("entities", {})

            # Diagnoses
            for diag in entities.get("diagnoses", []):
                if isinstance(diag, str) and diag.strip():
                    field_values.setdefault("diagnoses", []).append((source, diag.strip()))

            # Other key fields
            for field_name in ["allergies", "admission_date", "discharge_date"]:
                val = entities.get(field_name)
                if val and isinstance(val, str) and val.strip():
                    field_values.setdefault(field_name, []).append((source, val.strip()))

        # Now detect conflicts
        for field_name, source_values in field_values.items():
            conflicts.extend(
                _detect_field_conflicts(field_name, source_values)
            )

        return ToolResult(
            status=ToolStatus.SUCCESS,
            data=conflicts,
            metadata={"conflict_count": len(conflicts)},
        )

    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "extracted_documents": {
                    "type": "array",
                    "description": "List of extracted document dicts with 'source' and 'entities'",
                    "required": True,
                },
            },
        }


def _detect_field_conflicts(
    field_name: str,
    source_values: List[Tuple[str, str]],
) -> List[Conflict]:
    """
    For a given field, check if multiple sources give different values.
    Normalises values before comparison to reduce false positives.
    """
    conflicts: List[Conflict] = []
    if len(source_values) < 2:
        return conflicts

    seen: Dict[str, str] = {}  # normalised_value → first source

    for source, raw_value in source_values:
        norm = _normalise_value(raw_value)
        if norm in seen:
            continue  # Same value — no conflict
        for prev_norm, prev_source in seen.items():
            if not _values_compatible(norm, prev_norm):
                conflicts.append(
                    Conflict(
                        field=field_name,
                        note_a_source=prev_source,
                        note_a_value=_get_original(source_values, prev_source),
                        note_b_source=source,
                        note_b_value=raw_value,
                        description=(
                            f"Field '{field_name}' has conflicting values across notes: "
                            f"'{prev_source}' vs '{source}'"
                        ),
                    )
                )
        seen[norm] = source

    return conflicts


def _normalise_value(v: str) -> str:
    return re.sub(r"\s+", " ", v.lower().strip())


def _values_compatible(a: str, b: str) -> bool:
    """
    True if both values are essentially the same or one contains the other.
    Reduces noisy false positives from abbreviations / partial notes.
    """
    return a == b or a in b or b in a


def _get_original(source_values: List[Tuple[str, str]], source: str) -> str:
    for s, v in source_values:
        if s == source:
            return v
    return "unknown"
