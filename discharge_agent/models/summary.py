"""
models/summary.py
DischargeSummary — the structured output produced by the agent.
Every field is Optional; missing fields are filled with SETTINGS.missing_sentinel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from config.settings import SETTINGS
from models.patient import (
    Conflict,
    LabResult,
    Medication,
    MedicationStatus,
    Procedure,
    VitalSigns,
)


@dataclass
class MedicationChange:
    medication: Medication
    change_type: MedicationStatus
    reason_documented: bool
    flag_message: Optional[str] = None  # Set if change_reason is missing

    def to_dict(self) -> dict:
        return {
            "medication": self.medication.to_dict(),
            "change_type": self.change_type.value,
            "reason_documented": self.reason_documented,
            "flag_message": self.flag_message,
        }


@dataclass
class DischargeSummary:
    """
    Structured discharge summary draft.
    Never auto-finalized — always a draft for clinician review.
    """

    patient_id: str = SETTINGS.missing_sentinel

    # Demographics
    patient_demographics: str = SETTINGS.missing_sentinel  # name/age/sex/MRN

    # Dates
    admission_date: str = SETTINGS.missing_sentinel
    discharge_date: str = SETTINGS.missing_sentinel

    # Diagnoses
    principal_diagnosis: str = SETTINGS.missing_sentinel
    secondary_diagnoses: List[str] = field(default_factory=list)

    # Clinical narrative
    hospital_course: str = SETTINGS.missing_sentinel

    # Procedures
    procedures: List[Procedure] = field(default_factory=list)

    # Vitals at admission / discharge
    admission_vitals: Optional[VitalSigns] = None
    discharge_vitals: Optional[VitalSigns] = None

    # Medications
    admission_medications: List[Medication] = field(default_factory=list)
    discharge_medications: List[Medication] = field(default_factory=list)
    medication_changes: List[MedicationChange] = field(default_factory=list)

    # Safety
    allergies: str = SETTINGS.missing_sentinel
    drug_interaction_flags: List[str] = field(default_factory=list)

    # Follow-up
    follow_up_instructions: str = SETTINGS.missing_sentinel

    # Pending
    pending_results: List[str] = field(default_factory=list)

    # Labs (most recent values)
    key_lab_results: List[LabResult] = field(default_factory=list)

    # Discharge
    discharge_condition: str = SETTINGS.missing_sentinel
    discharge_destination: str = SETTINGS.missing_sentinel

    # Meta
    conflicts_detected: List[Conflict] = field(default_factory=list)
    clinician_flags: List[str] = field(default_factory=list)  # All escalations
    source_documents: List[str] = field(default_factory=list)
    unreadable_documents: List[str] = field(default_factory=list)

    # Guardrail
    fabrication_scan_passed: bool = False
    is_draft: bool = True  # Always True — never auto-finalized

    def validate_completeness(self) -> List[str]:
        """
        Check all required fields. Returns list of field names that are
        still set to the missing_sentinel (i.e. not sourced from documents).
        """
        issues: List[str] = []
        for section in SETTINGS.required_sections:
            val = getattr(self, section, None)
            if val is None:
                issues.append(section)
            elif isinstance(val, str) and val == SETTINGS.missing_sentinel:
                issues.append(section)
            elif isinstance(val, list) and len(val) == 0:
                # Empty lists are acceptable for some fields
                pass
        return issues

    def to_dict(self) -> dict:
        return {
            "is_draft": self.is_draft,
            "patient_id": self.patient_id,
            "patient_demographics": self.patient_demographics,
            "admission_date": self.admission_date,
            "discharge_date": self.discharge_date,
            "principal_diagnosis": self.principal_diagnosis,
            "secondary_diagnoses": self.secondary_diagnoses,
            "hospital_course": self.hospital_course,
            "procedures": [p.to_dict() for p in self.procedures],
            "admission_vitals": self.admission_vitals.to_dict()
            if self.admission_vitals
            else None,
            "discharge_vitals": self.discharge_vitals.to_dict()
            if self.discharge_vitals
            else None,
            "admission_medications": [m.to_dict() for m in self.admission_medications],
            "discharge_medications": [m.to_dict() for m in self.discharge_medications],
            "medication_changes": [mc.to_dict() for mc in self.medication_changes],
            "allergies": self.allergies,
            "drug_interaction_flags": self.drug_interaction_flags,
            "follow_up_instructions": self.follow_up_instructions,
            "pending_results": self.pending_results,
            "key_lab_results": [lr.to_dict() for lr in self.key_lab_results],
            "discharge_condition": self.discharge_condition,
            "discharge_destination": self.discharge_destination,
            "conflicts_detected": [c.to_dict() for c in self.conflicts_detected],
            "clinician_flags": self.clinician_flags,
            "source_documents": self.source_documents,
            "unreadable_documents": self.unreadable_documents,
            "fabrication_scan_passed": self.fabrication_scan_passed,
        }

    def to_markdown(self) -> str:
        """Render summary as human-readable markdown for clinician review."""
        lines = [
            "# DISCHARGE SUMMARY DRAFT",
            "> ⚠️  **THIS IS AN AI-GENERATED DRAFT — FOR CLINICIAN REVIEW ONLY**",
            "> All fields marked `[MISSING]`, `[PENDING]`, `[CONFLICT]` or `[FLAG]` **require clinician attention.**",
            "",
            f"**Patient:** {self.patient_demographics}",
            f"**Admission Date:** {self.admission_date}",
            f"**Discharge Date:** {self.discharge_date}",
            "",
            "---",
            "## Principal Diagnosis",
            self.principal_diagnosis,
            "",
            "## Secondary Diagnoses",
        ]
        if self.secondary_diagnoses:
            for d in self.secondary_diagnoses:
                lines.append(f"- {d}")
        else:
            lines.append(SETTINGS.missing_sentinel)

        lines += [
            "",
            "## Hospital Course",
            self.hospital_course,
            "",
            "## Procedures",
        ]
        if self.procedures:
            for p in self.procedures:
                lines.append(
                    f"- **{p.name}** | Date: {p.date or 'unknown'} | {p.notes or ''}"
                )
        else:
            lines.append("None documented.")

        lines += [
            "",
            "## Discharge Medications",
        ]
        if self.discharge_medications:
            lines.append("| Drug | Dose | Route | Frequency | Duration | Change |")
            lines.append("|------|------|-------|-----------|----------|--------|")
            for m in self.discharge_medications:
                change = m.status.value if m.status else "—"
                lines.append(
                    f"| {m.name} | {m.dose or '—'} | {m.route or '—'} "
                    f"| {m.frequency or '—'} | {m.duration or '—'} | {change} |"
                )
        else:
            lines.append(SETTINGS.missing_sentinel)

        lines += ["", "## Medication Changes & Flags"]
        if self.medication_changes:
            for mc in self.medication_changes:
                flag = f"  ⚠️ {mc.flag_message}" if mc.flag_message else ""
                lines.append(
                    f"- **{mc.medication.name}**: {mc.change_type.value}"
                    f"{' — reason: ' + mc.medication.change_reason if mc.medication.change_reason else ' — [REASON NOT DOCUMENTED]'}"
                    f"{flag}"
                )
        else:
            lines.append("No changes detected.")

        lines += [
            "",
            "## Allergies",
            self.allergies,
            "",
            "## Drug Interaction Flags",
        ]
        if self.drug_interaction_flags:
            for f_ in self.drug_interaction_flags:
                lines.append(f"- ⚠️ {f_}")
        else:
            lines.append("None flagged.")

        lines += [
            "",
            "## Discharge Condition",
            self.discharge_condition,
            "",
            "## Follow-Up Instructions",
            self.follow_up_instructions,
            "",
            "## Pending Results",
        ]
        if self.pending_results:
            for pr in self.pending_results:
                lines.append(f"- ⏳ {pr}")
        else:
            lines.append("None documented as pending.")

        lines += ["", "## Key Laboratory Results"]
        if self.key_lab_results:
            lines.append("| Test | Value | Unit | Ref Range | Status | Date |")
            lines.append("|------|-------|------|-----------|--------|------|")
            for lr in self.key_lab_results:
                abnormal = " ⚠️" if lr.is_abnormal else ""
                lines.append(
                    f"| {lr.test_name}{abnormal} | {lr.value or '—'} | {lr.unit or '—'} "
                    f"| {lr.reference_range or '—'} | {lr.status.value} | {lr.date or '—'} |"
                )
        else:
            lines.append("No lab results extracted.")

        if self.conflicts_detected:
            lines += ["", "## ⚠️ Conflicts Detected — Requires Clinician Review"]
            for c in self.conflicts_detected:
                lines.append(
                    f"- **{c.field}**: `{c.note_a_source}` → *{c.note_a_value}* vs "
                    f"`{c.note_b_source}` → *{c.note_b_value}*"
                )

        if self.clinician_flags:
            lines += ["", "## 🚩 Clinician Flags"]
            for cf in self.clinician_flags:
                lines.append(f"- {cf}")

        lines += [
            "",
            "---",
            "## Source Documents",
        ]
        for s in self.source_documents:
            lines.append(f"- {s}")
        if self.unreadable_documents:
            lines += ["", "### Unreadable Documents"]
            for u in self.unreadable_documents:
                lines.append(f"- ❌ {u}")

        lines += [
            "",
            "---",
            f"*Fabrication scan passed: {self.fabrication_scan_passed}*",
            "*This document is an AI-generated draft and must be reviewed and signed by a qualified clinician before use.*",
        ]
        return "\n".join(lines)
