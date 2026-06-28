"""
models/patient.py
Core clinical data models — all fields Optional to represent missing data.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class MedicationStatus(str, Enum):
    CONTINUED = "continued"
    ADDED = "added"
    STOPPED = "stopped"
    DOSE_CHANGED = "dose_changed"
    ROUTE_CHANGED = "route_changed"
    UNKNOWN_CHANGE = "unknown_change"


class LabStatus(str, Enum):
    FINAL = "final"
    PENDING = "pending"
    CORRECTED = "corrected"
    UNKNOWN = "unknown"


@dataclass
class Medication:
    name: str
    dose: Optional[str] = None
    route: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None
    indication: Optional[str] = None
    # Reconciliation fields
    status: Optional[MedicationStatus] = None
    change_reason: Optional[str] = None  # None if not documented
    source_note: Optional[str] = None  # Which document this came from

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dose": self.dose,
            "route": self.route,
            "frequency": self.frequency,
            "duration": self.duration,
            "indication": self.indication,
            "status": self.status.value if self.status else None,
            "change_reason": self.change_reason,
            "source_note": self.source_note,
        }


@dataclass
class LabResult:
    test_name: str
    value: Optional[str] = None
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    status: LabStatus = LabStatus.UNKNOWN
    date: Optional[str] = None
    is_abnormal: Optional[bool] = None
    source_note: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "test_name": self.test_name,
            "value": self.value,
            "unit": self.unit,
            "reference_range": self.reference_range,
            "status": self.status.value,
            "date": self.date,
            "is_abnormal": self.is_abnormal,
            "source_note": self.source_note,
        }


@dataclass
class Procedure:
    name: str
    date: Optional[str] = None
    notes: Optional[str] = None
    source_note: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "date": self.date,
            "notes": self.notes,
            "source_note": self.source_note,
        }


@dataclass
class VitalSigns:
    blood_pressure: Optional[str] = None
    pulse_rate: Optional[str] = None
    respiratory_rate: Optional[str] = None
    temperature: Optional[str] = None
    spo2: Optional[str] = None
    grbs: Optional[str] = None
    date: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "blood_pressure": self.blood_pressure,
            "pulse_rate": self.pulse_rate,
            "respiratory_rate": self.respiratory_rate,
            "temperature": self.temperature,
            "spo2": self.spo2,
            "grbs": self.grbs,
            "date": self.date,
        }


@dataclass
class PatientDocument:
    """Represents one source PDF page / section after extraction."""

    file_path: str
    page_number: int
    note_type: str  # Classified type (see settings.NOTE_TYPE_KEYWORDS)
    raw_text: str
    extraction_confidence: float = 1.0  # 0–1; low for handwritten notes
    read_error: Optional[str] = None  # If extraction failed

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "page_number": self.page_number,
            "note_type": self.note_type,
            "extraction_confidence": self.extraction_confidence,
            "read_error": self.read_error,
            # raw_text excluded from dict — large, stored separately
        }


@dataclass
class Conflict:
    """A detected conflict between two source notes."""

    field: str
    note_a_source: str
    note_a_value: str
    note_b_source: str
    note_b_value: str
    description: str

    def to_flagged_string(self, conflict_prefix: str, conflict_suffix: str) -> str:
        return (
            f"{conflict_prefix} '{self.field}': "
            f"'{self.note_a_source}' states '{self.note_a_value}'; "
            f"'{self.note_b_source}' states '{self.note_b_value}' "
            f"{conflict_suffix}"
        )

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "note_a_source": self.note_a_source,
            "note_a_value": self.note_a_value,
            "note_b_source": self.note_b_source,
            "note_b_value": self.note_b_value,
            "description": self.description,
        }
