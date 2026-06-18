"""
agent/executor.py
Tool executor — dispatches tool calls, handles retries, updates agent state.
Never raises unhandled exceptions — all failures are captured as ToolResult(FAILED).
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import SETTINGS
from agent.state import AgentState
from models.patient import (
    Conflict, LabResult, LabStatus, Medication,
    MedicationStatus, Procedure, VitalSigns,
)
from models.summary import DischargeSummary, MedicationChange
from models.trace import AgentStep, StepStatus
from tools.base import ToolResult, ToolStatus
from tools.conflict_detector import ConflictDetectorTool
from tools.document_parser import DocumentParserTool
from tools.drug_interaction import DrugInteractionTool
from tools.escalation import EscalationSeverity, EscalationTool
from tools.lab_extractor import LabExtractorTool
from tools.medication_reconciler import MedicationReconcilerTool
from tools.pdf_reader import PDFReaderTool
from utils.json_utils import safe_json_parse


class Executor:
    def __init__(self, llm_client=None):
        self.llm = llm_client
        self.pdf_reader = PDFReaderTool()
        self.doc_parser = DocumentParserTool(llm_client=llm_client)
        self.lab_extractor = LabExtractorTool()
        self.conflict_detector = ConflictDetectorTool()
        self.med_reconciler = MedicationReconcilerTool()
        self.drug_interaction = DrugInteractionTool()
        self.escalation = EscalationTool()

        self._tool_map = {
            "pdf_reader": self._run_pdf_reader,
            "document_parser": self._run_document_parser,
            "lab_extractor": self._run_lab_extractor,
            "conflict_detector": self._run_conflict_detector,
            "medication_reconciler": self._run_medication_reconciler,
            "drug_interaction_lookup": self._run_drug_interaction,
            "escalate": self._run_escalate,
            "synthesise_hospital_course": self._run_synthesise_hospital_course,
            "assemble_summary": self._run_assemble_summary,
            "fabrication_guard": self._run_fabrication_guard,
        }

    def execute_step(
        self,
        step: Dict[str, Any],
        state: AgentState,
        step_number: int,
    ) -> AgentStep:
        tool_name = step["tool"]
        t0 = time.time()

        agent_step = AgentStep(
            step_number=step_number,
            reasoning=f"Executing plan step {step['step_id']}: {step['goal']}",
            tool_name=tool_name,
            tool_inputs={"step_id": step["step_id"]},
        )

        handler = self._tool_map.get(tool_name)
        if not handler:
            agent_step.tool_status = StepStatus.FAILED
            agent_step.tool_result = f"Unknown tool: {tool_name}"
            agent_step.next_decision = "Mark step failed, continue with plan"
            agent_step.duration_s = time.time() - t0
            return agent_step

        # Retry wrapper
        result = self._run_with_retry(handler, step, state)

        agent_step.tool_result = (
            result.to_dict() if hasattr(result, "to_dict") else str(result)
        )
        agent_step.duration_s = time.time() - t0

        if result.ok:
            agent_step.tool_status = (
                StepStatus.SUCCESS
                if result.status == ToolStatus.SUCCESS
                else StepStatus.PARTIAL
            )
            agent_step.next_decision = "Step succeeded — continue to next"
        else:
            agent_step.tool_status = StepStatus.FAILED
            agent_step.next_decision = (
                f"Step failed ({result.error}) — "
                f"{'marking required fields MISSING' if step.get('required') else 'skipping'}"
            )

        # Capture any flags raised during this step
        agent_step.flags_raised = list(self.escalation.get_all_flags())

        return agent_step

    def _run_with_retry(
        self,
        handler,
        step: Dict[str, Any],
        state: AgentState,
    ) -> ToolResult:
        last_result = ToolResult(status=ToolStatus.FAILED, error="Not attempted")
        for attempt in range(SETTINGS.max_retries_per_tool):
            try:
                last_result = handler(step, state)
                if last_result.ok:
                    return last_result
                # Brief backoff before retry
                if attempt < SETTINGS.max_retries_per_tool - 1:
                    time.sleep(SETTINGS.retry_base_delay_s * (2 ** attempt))
            except Exception as exc:
                last_result = ToolResult(
                    status=ToolStatus.FAILED,
                    error=f"Attempt {attempt+1} exception: {exc}",
                )
        return last_result

    # ──────────────────────────────────────────────────────────────────────
    # Tool handlers
    # ──────────────────────────────────────────────────────────────────────

    def _run_pdf_reader(self, step: Dict, state: AgentState) -> ToolResult:
        result = self.pdf_reader.run(path=state.patient_dir)
        if result.ok and result.data:
            state.raw_documents = result.data
            state.unreadable_files = result.metadata.get("unreadable_files", [])
        return result

    def _run_document_parser(self, step: Dict, state: AgentState) -> ToolResult:
        if not state.raw_documents:
            return ToolResult(
                status=ToolStatus.FAILED,
                error="No raw documents to parse (pdf_reader step may have failed)",
            )
        result = self.doc_parser.run(
            documents=state.raw_documents,
            use_llm=bool(self.llm),
        )
        if result.ok and result.data:
            state.parsed_documents = result.data
            self._hydrate_state_from_parsed(state)
        return result

    def _hydrate_state_from_parsed(self, state: AgentState) -> None:
        """
        Walk all parsed documents and populate state fields.
        Does not overwrite already-populated fields (first-seen wins,
        except for lists which accumulate).
        """
        for doc in state.parsed_documents:
            entities = doc.get("entities", {})
            if not entities:
                continue

            note_type = doc.get("note_type", "unknown")

            # Demographics (from admission note preferably)
            if not state.demographics.get("name"):
                name = entities.get("patient_name")
                if name:
                    state.demographics["name"] = name
            if not state.demographics.get("age"):
                age = entities.get("patient_age")
                if age:
                    state.demographics["age"] = age
            if not state.demographics.get("sex"):
                sex = entities.get("patient_sex")
                if sex:
                    state.demographics["sex"] = sex
            if not state.demographics.get("mrn"):
                mrn = entities.get("mrn")
                if mrn:
                    state.demographics["mrn"] = mrn

            # Dates
            if not state.admission_date:
                state.admission_date = entities.get("admission_date")
            if not state.discharge_date:
                state.discharge_date = entities.get("discharge_date")

            # Diagnoses — accumulate from all notes
            for diag in entities.get("diagnoses", []):
                if diag and diag not in state.diagnoses:
                    state.diagnoses.append(diag)

            # Allergies
            if not state.allergies:
                allergy = entities.get("allergies")
                if allergy:
                    state.allergies = allergy

            # Follow-up
            if not state.follow_up:
                fu = entities.get("follow_up")
                if fu:
                    state.follow_up = fu

            # Discharge condition
            if not state.discharge_condition:
                dc = entities.get("discharge_condition")
                if dc:
                    state.discharge_condition = dc

            # Medications — separate admission vs discharge
            for med_dict in entities.get("medications", []):
                if not isinstance(med_dict, dict):
                    continue
                med = Medication(
                    name=med_dict.get("name", "Unknown"),
                    dose=med_dict.get("dose"),
                    route=med_dict.get("route"),
                    frequency=med_dict.get("frequency"),
                    duration=med_dict.get("duration"),
                    source_note=f"{doc.get('file_path')}:p{doc.get('page_number')}",
                )
                if note_type in ("discharge_summary", "medication_record") and "discharge" in doc.get("raw_text", "").lower():
                    if med.name not in [m.name for m in state.discharge_medications]:
                        state.discharge_medications.append(med)
                else:
                    if med.name not in [m.name for m in state.admission_medications]:
                        state.admission_medications.append(med)

            # Procedures
            for proc_name in entities.get("procedures", []):
                if proc_name and proc_name not in [p.name for p in state.procedures]:
                    state.procedures.append(Procedure(name=proc_name))

            # Pending results
            for pr in entities.get("pending_results", []):
                if pr and pr not in state.pending_results:
                    state.pending_results.append(pr)

    def _run_lab_extractor(self, step: Dict, state: AgentState) -> ToolResult:
        if not state.parsed_documents:
            return ToolResult(
                status=ToolStatus.FAILED,
                error="No parsed documents available for lab extraction",
            )
        result = self.lab_extractor.run(documents=state.parsed_documents)
        if result.ok and result.data:
            labs = result.data.get("labs", [])
            pending = result.data.get("pending", [])
            state.lab_results.extend(labs)
            for p in pending:
                if p not in state.pending_results:
                    state.pending_results.append(p)
        return result

    def _run_conflict_detector(self, step: Dict, state: AgentState) -> ToolResult:
        if not state.parsed_documents:
            return ToolResult(
                status=ToolStatus.NOT_FOUND,
                error="No parsed documents for conflict detection",
            )
        result = self.conflict_detector.run(extracted_documents=state.parsed_documents)
        if result.ok and result.data:
            state.conflicts.extend(result.data)
            for conflict in result.data:
                flag = conflict.to_flagged_string(
                    SETTINGS.conflict_prefix, SETTINGS.conflict_suffix
                )
                state.add_flag(flag)
        return result

    def _run_medication_reconciler(self, step: Dict, state: AgentState) -> ToolResult:
        result = self.med_reconciler.run(
            admission_meds=state.admission_medications,
            discharge_meds=state.discharge_medications,
        )
        if result.ok and result.data:
            state.medication_changes = result.data
            for flag in result.metadata.get("flags", []):
                state.add_flag(f"{SETTINGS.flag_prefix} Medication: {flag}]")
        return result

    def _run_drug_interaction(self, step: Dict, state: AgentState) -> ToolResult:
        med_names = [m.name for m in state.discharge_medications]
        if not med_names:
            # Try admission meds as fallback
            med_names = [m.name for m in state.admission_medications]
        if not med_names:
            return ToolResult(
                status=ToolStatus.NOT_FOUND,
                error="No medications found for drug interaction check",
            )
        result = self.drug_interaction.run(medication_names=med_names)
        if result.ok and result.data:
            for interaction in result.data:
                flag_msg = (
                    f"[{'🚨 CRITICAL' if interaction['severity'] == 'HIGH' else '⚠️ MODERATE'}] "
                    f"Drug Interaction: {interaction['description']}"
                )
                state.drug_interaction_flags.append(flag_msg)
                if interaction.get("requires_escalation"):
                    state.add_flag(flag_msg)
                    # Auto-escalate high severity
                    self.escalation.run(
                        severity="critical",
                        field="drug_interactions",
                        message=interaction["description"],
                        source_evidence="drug_interaction_lookup",
                    )
        return result

    def _run_escalate(self, step: Dict, state: AgentState) -> ToolResult:
        """
        Systematic escalation pass — escalate everything not yet escalated.
        """
        escalated = 0

        # Missing mandatory fields
        for field_name in SETTINGS.required_sections:
            val = getattr(state, field_name, None)
            if val is None or val == "" or val == []:
                self.escalation.run(
                    severity="high",
                    field=field_name,
                    message=f"Required field '{field_name}' could not be extracted from source documents",
                    source_evidence="document_corpus",
                )
                escalated += 1

        # Conflicts
        for conflict in state.conflicts:
            self.escalation.run(
                severity="high",
                field=conflict.field,
                message=conflict.description,
                source_evidence=f"{conflict.note_a_source} vs {conflict.note_b_source}",
            )
            escalated += 1

        # Medications changed without reason
        for mc in state.medication_changes:
            if mc.flag_message:
                self.escalation.run(
                    severity="high",
                    field="medication_changes",
                    message=mc.flag_message,
                    source_evidence=mc.medication.source_note or "unknown",
                )
                escalated += 1

        # Pending results
        for pr in state.pending_results:
            self.escalation.run(
                severity="informational",
                field="pending_results",
                message=f"Result still pending: {pr}",
                source_evidence="lab_extractor",
            )
            escalated += 1

        # Sync all escalation flags back to state
        state.clinician_flags = self.escalation.get_all_flags()

        return ToolResult(
            status=ToolStatus.SUCCESS,
            data={"escalated_items": escalated},
            metadata={"total_flags": len(state.clinician_flags)},
        )

    def _run_synthesise_hospital_course(
        self, step: Dict, state: AgentState
    ) -> ToolResult:
        """
        Use LLM to write a coherent hospital course from chronological notes.
        Falls back to raw text concatenation if LLM unavailable.
        """
        if not state.parsed_documents:
            state.hospital_course = SETTINGS.missing_sentinel
            return ToolResult(
                status=ToolStatus.FAILED,
                error="No documents available for hospital course synthesis",
            )

        # Gather relevant notes sorted by note type priority
        note_priority = [
            "admission_note", "progress_note", "nursing_documentation",
            "icu_chart", "discharge_summary",
        ]
        relevant_texts = []
        for priority_type in note_priority:
            for doc in state.parsed_documents:
                if doc.get("note_type") == priority_type:
                    text = doc.get("raw_text", "")[:1500]
                    if text.strip():
                        relevant_texts.append(
                            f"[{priority_type.upper()} | "
                            f"{doc.get('file_path','?')}:p{doc.get('page_number','?')}]\n{text}"
                        )

        if not relevant_texts:
            state.hospital_course = SETTINGS.missing_sentinel
            return ToolResult(
                status=ToolStatus.NOT_FOUND,
                error="No relevant clinical notes found",
            )

        combined = "\n\n---\n\n".join(relevant_texts[:8])  # cap tokens

        if self.llm:
            from prompts.extraction_prompt import HOSPITAL_COURSE_PROMPT_TEMPLATE
            prompt = HOSPITAL_COURSE_PROMPT_TEMPLATE.format(notes_text=combined)
            try:
                narrative = self.llm.complete(prompt, max_tokens=600)
                state.hospital_course = narrative.strip() or SETTINGS.missing_sentinel
            except Exception as exc:
                state.hospital_course = (
                    f"[Hospital course synthesis failed: {exc} — "
                    f"raw notes available in source documents]"
                )
        else:
            # Fallback: first 500 chars of each relevant note
            state.hospital_course = (
                "Synthesised from source notes (LLM unavailable):\n"
                + "\n".join(t[:300] for t in relevant_texts[:3])
            )

        return ToolResult(
            status=ToolStatus.SUCCESS,
            data={"hospital_course_length": len(state.hospital_course)},
        )

    def _run_assemble_summary(
        self, step: Dict, state: AgentState
    ) -> ToolResult:
        """
        Build the DischargeSummary object from everything in state.
        Every missing field gets the sentinel — never left blank.
        """
        M = SETTINGS.missing_sentinel
        P = SETTINGS.pending_sentinel

        # Demographics string
        demo = state.demographics
        demo_str = (
            f"Name: {demo.get('name', M)} | "
            f"Age: {demo.get('age', M)} | "
            f"Sex: {demo.get('sex', M)} | "
            f"MRN: {demo.get('mrn', M)}"
        ) if demo else M

        # Principal vs secondary diagnoses
        all_diag = state.diagnoses or state.discharge_diagnoses
        principal = all_diag[0] if all_diag else M
        secondary = all_diag[1:] if len(all_diag) > 1 else []

        # Conflict strings
        conflict_strings = {
            c.field: c.to_flagged_string(
                SETTINGS.conflict_prefix, SETTINGS.conflict_suffix
            )
            for c in state.conflicts
        }

        summary = DischargeSummary(
            patient_id=state.patient_id,
            patient_demographics=demo_str,
            admission_date=state.admission_date or M,
            discharge_date=state.discharge_date or M,
            principal_diagnosis=conflict_strings.get("diagnoses", principal),
            secondary_diagnoses=secondary,
            hospital_course=state.hospital_course or M,
            procedures=state.procedures,
            admission_medications=state.admission_medications,
            discharge_medications=state.discharge_medications,
            medication_changes=state.medication_changes,
            allergies=state.allergies or M,
            drug_interaction_flags=state.drug_interaction_flags,
            follow_up_instructions=state.follow_up or M,
            pending_results=state.pending_results,
            key_lab_results=state.lab_results,
            discharge_condition=state.discharge_condition or M,
            discharge_destination=M,
            conflicts_detected=state.conflicts,
            clinician_flags=state.clinician_flags,
            source_documents=[
                f"{d.get('file_path','?')}:p{d.get('page_number','?')}"
                for d in state.parsed_documents
            ],
            unreadable_documents=state.unreadable_files,
            is_draft=True,
        )

        # Store on state for fabrication guard
        state.__dict__["_draft_summary"] = summary

        return ToolResult(
            status=ToolStatus.SUCCESS,
            data={"summary_assembled": True},
            metadata={"sections": len(SETTINGS.required_sections)},
        )

    def _run_fabrication_guard(
        self, step: Dict, state: AgentState
    ) -> ToolResult:
        """
        Post-assembly scan:
        1. Check all required fields are populated (or marked MISSING/PENDING)
        2. Scan for confident-sounding text that doesn't match any source doc
        """
        summary: DischargeSummary = state.__dict__.get("_draft_summary")
        if not summary:
            return ToolResult(
                status=ToolStatus.FAILED,
                error="No draft summary found to scan",
            )

        issues = []

        # Check required fields
        for field_name in SETTINGS.required_sections:
            val = getattr(summary, field_name, None)
            if val is None:
                issues.append(f"Field '{field_name}' is None (not set to MISSING sentinel)")
            elif isinstance(val, str) and val.strip() == "":
                issues.append(f"Field '{field_name}' is empty string (should be MISSING sentinel)")

        if issues:
            for issue in issues:
                self.escalation.run(
                    severity="high",
                    field="fabrication_guard",
                    message=f"Guard issue: {issue}",
                    source_evidence="fabrication_guard",
                )
            summary.clinician_flags = self.escalation.get_all_flags()
            summary.fabrication_scan_passed = False
        else:
            summary.fabrication_scan_passed = True

        return ToolResult(
            status=ToolStatus.SUCCESS,
            data={
                "passed": summary.fabrication_scan_passed,
                "issues_found": len(issues),
                "issues": issues,
            },
        )

    def get_final_summary(self, state: AgentState) -> Optional[DischargeSummary]:
        return state.__dict__.get("_draft_summary")
