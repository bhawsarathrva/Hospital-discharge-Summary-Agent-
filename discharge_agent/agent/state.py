"""
agent/state.py
Agent working memory.
Holds all extracted data as the agent progresses through its plan.
Immutable after construction — mutations return new State objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from models.patient import Conflict, LabResult, Medication, PatientDocument, Procedure
from models.summary import MedicationChange
from models.trace import AgentTrace


@dataclass
class AgentState:
    """
    Mutable working memory for one patient's agent run.
    Each tool call updates this state.
    """

    patient_id: str
    patient_dir: str

    # Document corpus
    raw_documents: List[PatientDocument] = field(default_factory=list)
    parsed_documents: List[Dict[str, Any]] = field(default_factory=list)
    unreadable_files: List[str] = field(default_factory=list)

    # Plan
    current_plan: List[Dict[str, Any]] = field(default_factory=list)
    completed_steps: List[int] = field(default_factory=list)
    pending_steps: List[int] = field(default_factory=list)
    failed_steps: List[int] = field(default_factory=list)

    # Extracted clinical data
    demographics: Dict[str, Any] = field(default_factory=dict)
    admission_date: Optional[str] = None
    discharge_date: Optional[str] = None
    diagnoses: List[str] = field(default_factory=list)
    discharge_diagnoses: List[str] = field(default_factory=list)
    hospital_course: Optional[str] = None
    procedures: List[Procedure] = field(default_factory=list)
    admission_medications: List[Medication] = field(default_factory=list)
    discharge_medications: List[Medication] = field(default_factory=list)
    medication_changes: List[MedicationChange] = field(default_factory=list)
    allergies: Optional[str] = None
    lab_results: List[LabResult] = field(default_factory=list)
    pending_results: List[str] = field(default_factory=list)
    discharge_condition: Optional[str] = None
    follow_up: Optional[str] = None

    # Safety
    conflicts: List[Conflict] = field(default_factory=list)
    clinician_flags: List[str] = field(default_factory=list)
    drug_interaction_flags: List[str] = field(default_factory=list)

    # Control
    step_count: int = 0
    is_complete: bool = False
    termination_reason: str = ""

    # Trace
    trace: Optional[AgentTrace] = None

    def add_flag(self, flag: str) -> None:
        if flag not in self.clinician_flags:
            self.clinician_flags.append(flag)

    def add_conflict(self, conflict: Conflict) -> None:
        self.conflicts.append(conflict)

    def increment_step(self) -> None:
        self.step_count += 1

    def to_summary_dict(self) -> Dict[str, Any]:
        """Flatten state into dict for summary assembly prompt."""
        return {
            "demographics": self.demographics,
            "admission_date": self.admission_date,
            "discharge_date": self.discharge_date,
            "diagnoses": self.diagnoses,
            "discharge_diagnoses": self.discharge_diagnoses,
            "hospital_course": self.hospital_course,
            "procedures": [p.to_dict() for p in self.procedures],
            "admission_medications": [m.to_dict() for m in self.admission_medications],
            "discharge_medications": [m.to_dict() for m in self.discharge_medications],
            "medication_changes": [mc.to_dict() for mc in self.medication_changes],
            "allergies": self.allergies,
            "lab_results": [lr.to_dict() for lr in self.lab_results],
            "pending_results": self.pending_results,
            "discharge_condition": self.discharge_condition,
            "follow_up": self.follow_up,
            "conflicts": [c.to_dict() for c in self.conflicts],
            "clinician_flags": self.clinician_flags,
            "drug_interaction_flags": self.drug_interaction_flags,
        }
