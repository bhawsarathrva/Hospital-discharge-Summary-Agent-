"""
agents/summary_agent.py
SummaryAgent synthesizes the clinical narrative, assembles the structured
discharge summary, and runs the final fabrication guard.
"""
from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional
from pathlib import Path

import importlib.util
from pathlib import Path

def _load_root_tool(name: str):
    root = Path(__file__).parent.parent
    spec = importlib.util.spec_from_file_location(f"root_tools_{name}", str(root / "tools" / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

logger_mod = _load_root_tool("logger")
get_logger = logger_mod.get_logger

logger = get_logger("summary_agent")
from config.settings import SETTINGS
from discharge_agent.models.patient import Conflict, LabResult, Medication, Procedure, VitalSigns
from discharge_agent.models.summary import DischargeSummary, MedicationChange
from prompts.extraction_prompt import HOSPITAL_COURSE_PROMPT_TEMPLATE

logger = get_logger("summary_agent")

class SummaryAgent:
    """
    SummaryAgent coordinates the synthesis and formatting of the final discharge summary draft.
    It builds the DischargeSummary structure, populates the sections, and runs fabrication guard validation.
    """
    def __init__(self, llm_client: Optional[Any] = None):
        self.llm = llm_client

    def run(self, state: Any) -> Any:
        """
        Runs the summary assembly and validation.
        Supports both AgentState objects and dictionary-based states.
        """
        # 1. Synthesize Hospital Course
        logger.info("Synthesizing hospital course narrative")
        hospital_course = self._synthesize_hospital_course(state)

        # 2. Assemble DischargeSummary
        logger.info("Assembling final discharge summary draft")
        summary = self._assemble_summary(state, hospital_course)

        # 3. Run Fabrication Guard Scan
        logger.info("Running fabrication guard checks")
        summary = self._run_fabrication_guard(summary, state)

        # Write final summary object/dict back to state
        if hasattr(state, "is_complete"):
            state.is_complete = True
            state.__dict__["_draft_summary"] = summary
        elif isinstance(state, dict):
            state["is_complete"] = True
            state["_draft_summary"] = summary.to_dict()

        return state

    def _synthesize_hospital_course(self, state: Any) -> str:
        # Get parsed documents
        parsed_docs = []
        if hasattr(state, "parsed_documents"):
            parsed_docs = state.parsed_documents
        elif isinstance(state, dict):
            parsed_docs = state.get("parsed_documents", [])

        if not parsed_docs:
            return SETTINGS.missing_sentinel

        # Gather relevant notes
        note_priority = [
            "admission_note", "progress_note", "nursing_documentation",
            "icu_chart", "discharge_summary",
        ]
        relevant_texts = []
        for priority_type in note_priority:
            for doc in parsed_docs:
                if doc.get("note_type") == priority_type:
                    text = doc.get("raw_text", "")[:1500]
                    if text.strip():
                        relevant_texts.append(
                            f"[{priority_type.upper()} | "
                            f"{doc.get('file_path','?')}:p{doc.get('page_number','?')}]\n{text}"
                        )

        if not relevant_texts:
            return SETTINGS.missing_sentinel

        combined = "\n\n---\n\n".join(relevant_texts[:8])

        if self.llm:
            prompt = HOSPITAL_COURSE_PROMPT_TEMPLATE.format(notes_text=combined)
            try:
                narrative = self.llm.complete(prompt, max_tokens=600)
                return narrative.strip() or SETTINGS.missing_sentinel
            except Exception as exc:
                logger.error(f"LLM course synthesis failed: {exc}")
                return f"[Hospital course synthesis failed: {exc}]"
        else:
            return (
                "Synthesised from source notes (LLM unavailable):\n"
                + "\n".join(t[:300] for t in relevant_texts[:3])
            )

    def _assemble_summary(self, state: Any, hospital_course: str) -> DischargeSummary:
        M = SETTINGS.missing_sentinel

        # Extract values from state
        patient_id = "unknown"
        demo_str = M
        admission_date = M
        discharge_date = M
        principal = M
        secondary = []
        procedures = []
        admission_meds = []
        discharge_meds = []
        med_changes = []
        allergies = M
        drug_interaction_flags = []
        follow_up = M
        pending_results = []
        lab_results = []
        discharge_condition = M
        conflicts = []
        clinician_flags = []
        parsed_docs = []
        unreadable_files = []

        if hasattr(state, "patient_id"):
            patient_id = state.patient_id
            demo = state.demographics
            if demo:
                demo_str = (
                    f"Name: {demo.get('name', M)} | "
                    f"Age: {demo.get('age', M)} | "
                    f"Sex: {demo.get('sex', M)} | "
                    f"MRN: {demo.get('mrn', M)}"
                )
            admission_date = state.admission_date or M
            discharge_date = state.discharge_date or M
            all_diag = state.diagnoses or state.discharge_diagnoses
            principal = all_diag[0] if all_diag else M
            secondary = all_diag[1:] if len(all_diag) > 1 else []
            procedures = state.procedures
            admission_meds = state.admission_medications
            discharge_meds = state.discharge_medications
            med_changes = state.medication_changes
            allergies = state.allergies or M
            drug_interaction_flags = state.drug_interaction_flags
            follow_up = state.follow_up or M
            pending_results = state.pending_results
            lab_results = state.lab_results
            discharge_condition = state.discharge_condition or M
            conflicts = state.conflicts
            clinician_flags = state.clinician_flags
            parsed_docs = state.parsed_documents
            unreadable_files = state.unreadable_files
        elif isinstance(state, dict):
            patient_id = state.get("patient_id", "unknown")
            demo = state.get("demographics", {})
            if demo:
                demo_str = (
                    f"Name: {demo.get('name', M)} | "
                    f"Age: {demo.get('age', M)} | "
                    f"Sex: {demo.get('sex', M)} | "
                    f"MRN: {demo.get('mrn', M)}"
                )
            admission_date = state.get("admission_date") or M
            discharge_date = state.get("discharge_date") or M
            all_diag = state.get("diagnoses", []) + state.get("discharge_diagnoses", [])
            principal = all_diag[0] if all_diag else M
            secondary = all_diag[1:] if len(all_diag) > 1 else []
            
            # Convert raw lists/dicts back to objects
            procedures = [Procedure(**p) if isinstance(p, dict) else p for p in state.get("procedures", [])]
            admission_meds = [Medication(**m) if isinstance(m, dict) else m for m in state.get("admission_medications", [])]
            discharge_meds = [Medication(**m) if isinstance(m, dict) else m for m in state.get("discharge_medications", [])]
            
            med_changes = []
            for mc in state.get("medication_changes", []):
                if isinstance(mc, dict):
                    med_data = mc.get("medication")
                    if isinstance(med_data, dict):
                        med_obj = Medication(**med_data)
                    else:
                        med_obj = med_data
                    med_changes.append(
                        MedicationChange(
                            medication=med_obj,
                            change_type=MedicationStatus(mc.get("change_type")),
                            reason_documented=mc.get("reason_documented", False),
                            flag_message=mc.get("flag_message"),
                        )
                    )
                else:
                    med_changes.append(mc)

            allergies = state.get("allergies") or M
            drug_interaction_flags = state.get("drug_interaction_flags", [])
            follow_up = state.get("follow_up") or M
            pending_results = state.get("pending_results", [])
            lab_results = [LabResult(**lr) if isinstance(lr, dict) else lr for lr in state.get("lab_results", [])]
            discharge_condition = state.get("discharge_condition") or M
            
            conflicts = [Conflict(**c) if isinstance(c, dict) else c for c in state.get("conflicts", [])]
            clinician_flags = state.get("clinician_flags", [])
            parsed_docs = state.get("parsed_documents", [])
            unreadable_files = state.get("unreadable_files", [])

        # Formatted conflicts overrides principal diagnosis if applicable
        conflict_strings = {
            c.field: c.to_flagged_string(
                SETTINGS.conflict_prefix, SETTINGS.conflict_suffix
            )
            for c in conflicts
        }

        summary = DischargeSummary(
            patient_id=patient_id,
            patient_demographics=demo_str,
            admission_date=admission_date,
            discharge_date=discharge_date,
            principal_diagnosis=conflict_strings.get("diagnoses", principal),
            secondary_diagnoses=secondary,
            hospital_course=hospital_course,
            procedures=procedures,
            admission_medications=admission_meds,
            discharge_medications=discharge_meds,
            medication_changes=med_changes,
            allergies=allergies,
            drug_interaction_flags=drug_interaction_flags,
            follow_up_instructions=follow_up,
            pending_results=pending_results,
            key_lab_results=lab_results,
            discharge_condition=discharge_condition,
            discharge_destination=M,
            conflicts_detected=conflicts,
            clinician_flags=clinician_flags,
            source_documents=[
                f"{d.get('file_path','?')}:p{d.get('page_number','?')}"
                for d in parsed_docs
            ],
            unreadable_documents=unreadable_files,
            is_draft=True,
        )
        return summary

    def _run_fabrication_guard(self, summary: DischargeSummary, state: Any) -> DischargeSummary:
        issues = []

        # Check required fields are populated
        for field_name in SETTINGS.required_sections:
            val = getattr(summary, field_name, None)
            if val is None:
                issues.append(f"Field '{field_name}' is None")
            elif isinstance(val, str) and val.strip() == "":
                issues.append(f"Field '{field_name}' is empty string")

        if issues:
            for issue in issues:
                flag = f"🚨 [CRITICAL] fabrication_guard: Guard issue: {issue}"
                if flag not in summary.clinician_flags:
                    summary.clinician_flags.append(flag)
            summary.fabrication_scan_passed = False
        else:
            summary.fabrication_scan_passed = True

        # Sync flags back to state if applicable
        if hasattr(state, "clinician_flags"):
            state.clinician_flags = summary.clinician_flags
        elif isinstance(state, dict):
            state["clinician_flags"] = summary.clinician_flags

        return summary
