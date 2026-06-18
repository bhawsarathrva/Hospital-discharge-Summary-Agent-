"""
config/settings.py
All tuneable constants for the discharge summary agent.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class AgentSettings:
    # ── LLM ─────────────────────────────────────────────────────────────────
    model: str = "gemini-1.5-flash"
    max_tokens: int = 4096
    temperature: float = 0.0          # Deterministic — clinical safety first

    # ── Loop control ────────────────────────────────────────────────────────
    max_steps: int = 30               # Hard cap: agent cannot loop forever
    max_retries_per_tool: int = 3     # Per-tool retry attempts
    retry_base_delay_s: float = 1.0   # Exponential backoff base

    # ── Document processing ─────────────────────────────────────────────────
    max_pdf_pages: int = 100          # Safety cap on pages per document
    ocr_fallback: bool = True         # Try text layer first, then OCR hint

    # ── Output paths ────────────────────────────────────────────────────────
    output_dir: str = "outputs"
    traces_subdir: str = "traces"
    summaries_subdir: str = "summaries"

    # ── Fabrication guard ────────────────────────────────────────────────────
    missing_sentinel: str = "[MISSING — requires clinician review]"
    pending_sentinel: str = "[PENDING — result not yet available]"
    conflict_prefix: str = "[CONFLICT:"
    conflict_suffix: str = "— requires clinician review]"
    flag_prefix: str = "[FLAG:"

    # ── Required summary sections ────────────────────────────────────────────
    required_sections: List[str] = field(default_factory=lambda: [
        "patient_demographics",
        "admission_date",
        "discharge_date",
        "principal_diagnosis",
        "secondary_diagnoses",
        "hospital_course",
        "procedures",
        "admission_medications",
        "discharge_medications",
        "medication_changes",
        "allergies",
        "follow_up_instructions",
        "pending_results",
        "discharge_condition",
    ])

    # ── Note type keywords ────────────────────────────────────────────────────
    note_type_keywords: dict = field(default_factory=lambda: {
        "admission_note": [
            "admission", "chief complaint", "history of present illness",
            "past history", "admission record",
        ],
        "progress_note": [
            "progress", "nursing note", "nursing documentation",
            "consultation", "daily note",
        ],
        "lab_result": [
            "haematology", "biochemistry", "pathology", "clinical pathology",
            "investigation", "blood count", "serum", "urine routine",
        ],
        "medication_record": [
            "drug chart", "medication", "drug administration",
            "prescription", "discharge medication",
        ],
        "radiology_report": [
            "ct", "mri", "x-ray", "xray", "usg", "ultrasound",
            "echo", "echocardiogram", "radiology",
        ],
        "discharge_summary": [
            "discharge summary", "condition at discharge",
            "advice on discharge", "follow-up",
        ],
        "icu_chart": ["icu", "intensive care", "icu chart"],
        "nursing_assessment": ["nursing assessment", "assessment on admission"],
    })


# Singleton
SETTINGS = AgentSettings()
