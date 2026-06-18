"""
tools/lab_extractor.py
Extract laboratory results from parsed document text.
Marks results as FINAL, PENDING, or UNKNOWN.
Flags abnormal values.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from models.patient import LabResult, LabStatus
from tools.base import BaseTool, ToolResult, ToolStatus


# Common lab test patterns — extend as needed
LAB_PATTERNS = [
    # Haematology
    ("haemoglobin", r"h(?:a?e)?moglobin\s*(?:\([^)]*\))?\s*[:\-]?\s*([\d.]+)\s*(g[m/]?d?[lL]?)?"),
    ("wbc", r"(?:wbc|total count|tlc)\s*[:\-]?\s*([\d,.]+)\s*(cells?/(?:cumm|mm3))?"),
    ("platelets", r"platelet\s+count\s*[:\-]?\s*([\d,.]+)\s*(lakhs?/cumm|×10\^9/L)?"),
    # Biochemistry
    ("serum_creatinine", r"(?:serum\s+)?creatinine\s*[:\-]?\s*([\d.]+)\s*(mg/d[lL])?"),
    ("sodium", r"(?:s\.?sodium|serum\s+sodium|s\.?na\+?)\s*[:\-]?\s*([\d.]+)\s*(mmol/[lL]|mEq/[lL])?"),
    ("potassium", r"(?:s\.?potassium|serum\s+potassium|s\.?k\+?)\s*[:\-]?\s*([\d.]+)\s*(mmol/[lL]|mEq/[lL])?"),
    ("blood_glucose_rbs", r"(?:rbs|random\s+blood\s+sugar|blood\s+glucose)\s*[:\-]?\s*([\d.]+)\s*(mg/d[lL])?"),
    ("hba1c", r"hba?1c?\s*[:\-]?\s*([\d.]+)\s*%?"),
    ("crp", r"c-?reactive\s+protein\s*[:\-]?\s*([\d.]+)\s*(mg/[lL])?"),
    # Serology
    ("widal_typhi_o", r"salmonella\s+typhi\s+.?o.?\s*[:\-]?\s*(positive|negative|[\d:]+)"),
    ("widal_typhi_h", r"salmonella\s+typhi\s+.?h.?\s*[:\-]?\s*(positive|negative|[\d:]+)"),
    # ABG
    ("ph", r"\bph\s*[:\-]?\s*([\d.]+)"),
    ("hco3", r"hco?3\s*[:\-]?\s*([\d.]+)\s*(mmol/[lL])?"),
]

# Reference ranges (approximate — for abnormal flagging only)
REFERENCE_RANGES = {
    "haemoglobin": (12.0, 17.5, "g/dL"),
    "serum_creatinine": (0.7, 1.4, "mg/dL"),
    "sodium": (135.0, 150.0, "mmol/L"),
    "potassium": (3.5, 5.0, "mmol/L"),
    "blood_glucose_rbs": (70.0, 140.0, "mg/dL"),
    "ph": (7.35, 7.45, ""),
    "hco3": (22.0, 28.0, "mmol/L"),
}

PENDING_INDICATORS = [
    "pending", "awaited", "report awaited", "sent to lab",
    "result due", "not received", "s/o pending",
]


class LabExtractorTool(BaseTool):
    name = "lab_extractor"
    description = (
        "Extract laboratory results from document text using regex patterns. "
        "Marks results as FINAL, PENDING, or UNKNOWN. Flags abnormal values. "
        "Does NOT invent values — only extracts what is explicitly in the text."
    )

    def _run(
        self,
        documents: List[Dict[str, Any]],  # Parsed document dicts from DocumentParserTool
    ) -> ToolResult:
        if not documents:
            return ToolResult(
                status=ToolStatus.NOT_FOUND,
                error="No documents provided",
            )

        all_labs: List[LabResult] = []
        pending: List[str] = []

        for doc in documents:
            text = doc.get("raw_text", "")
            source = f"{doc.get('file_path', 'unknown')}:p{doc.get('page_number', '?')}"
            date = doc.get("entities", {}).get("date_of_test", None)

            # Check for pending indicators
            for indicator in PENDING_INDICATORS:
                if indicator in text.lower():
                    pending_match = _extract_pending_test_name(text, indicator)
                    if pending_match:
                        pending.append(pending_match)

            # Extract lab values
            labs = _regex_extract_labs(text, source, date)
            all_labs.extend(labs)

            # Also pick up LLM-extracted labs if available
            llm_labs = doc.get("entities", {}).get("lab_results", [])
            for ll in llm_labs:
                if isinstance(ll, dict):
                    lab = LabResult(
                        test_name=ll.get("test", "unknown"),
                        value=ll.get("value"),
                        unit=ll.get("unit"),
                        reference_range=ll.get("reference_range"),
                        status=LabStatus.FINAL,
                        date=date,
                        source_note=source,
                    )
                    all_labs.append(lab)

        # Deduplicate by test_name (keep most recent / most complete)
        deduped = _deduplicate_labs(all_labs)

        # Mark abnormals
        for lab in deduped:
            _mark_abnormal(lab)

        return ToolResult(
            status=ToolStatus.SUCCESS,
            data={"labs": deduped, "pending": list(set(pending))},
            metadata={
                "total_labs": len(deduped),
                "pending_count": len(pending),
            },
        )

    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "documents": {
                    "type": "array",
                    "description": "List of parsed document dicts",
                    "required": True,
                }
            },
        }


def _regex_extract_labs(
    text: str,
    source: str,
    date: Optional[str],
) -> List[LabResult]:
    text_lower = text.lower()
    results: List[LabResult] = []

    for test_name, pattern in LAB_PATTERNS:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            value = match.group(1) if match.lastindex >= 1 else None
            unit = match.group(2) if match.lastindex >= 2 else None

            # Try to get reference range
            ref_min, ref_max, ref_unit = REFERENCE_RANGES.get(test_name, (None, None, None))
            ref_range = f"{ref_min}–{ref_max} {ref_unit}".strip() if ref_min else None

            lab = LabResult(
                test_name=test_name.replace("_", " ").title(),
                value=value,
                unit=unit or ref_unit,
                reference_range=ref_range,
                status=LabStatus.FINAL,
                date=date,
                source_note=source,
            )
            results.append(lab)

    return results


def _mark_abnormal(lab: LabResult) -> None:
    """Flag if value is outside reference range."""
    try:
        test_key = lab.test_name.lower().replace(" ", "_")
        ref_min, ref_max, _ = REFERENCE_RANGES.get(test_key, (None, None, None))
        if ref_min is not None and lab.value:
            val = float(re.sub(r"[^\d.]", "", lab.value))
            lab.is_abnormal = not (ref_min <= val <= ref_max)
    except (ValueError, TypeError):
        pass


def _deduplicate_labs(labs: List[LabResult]) -> List[LabResult]:
    seen: Dict[str, LabResult] = {}
    for lab in labs:
        key = lab.test_name.lower()
        if key not in seen or (lab.value and not seen[key].value):
            seen[key] = lab
    return list(seen.values())


def _extract_pending_test_name(text: str, indicator: str) -> Optional[str]:
    """Try to extract what test is pending from surrounding context."""
    idx = text.lower().find(indicator)
    if idx == -1:
        return None
    snippet = text[max(0, idx - 60):idx + 40]
    # Look for capitalised test names nearby
    match = re.search(r"([A-Z][A-Za-z\s/]+(?:culture|sensitivity|test|report|result))", snippet)
    if match:
        return f"{match.group(1).strip()} — {indicator}"
    return f"Unknown test — {indicator} (context: '{snippet.strip()[:60]}')"
