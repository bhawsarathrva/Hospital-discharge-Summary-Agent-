"""
agent/planner.py
Plans what the agent should do next given the current state.
Generates an ordered list of tool calls to satisfy all required sections.
Re-plans when tool calls fail or new information changes priorities.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from config.settings import SETTINGS
from agent.state import AgentState
from prompts.planning_prompt import PLANNING_PROMPT_TEMPLATE
from utils.json_utils import safe_json_parse


# Fixed plan steps — the agent always follows this sequence
# but can skip steps if data is already available or docs are missing
PLAN_TEMPLATE: List[Dict[str, Any]] = [
    {
        "step_id": 1,
        "goal": "Ingest all PDFs from patient directory",
        "tool": "pdf_reader",
        "required": True,
        "inputs_from_state": ["patient_dir"],
    },
    {
        "step_id": 2,
        "goal": "Classify and extract entities from all documents",
        "tool": "document_parser",
        "required": True,
        "inputs_from_state": ["raw_documents"],
        "depends_on": [1],
    },
    {
        "step_id": 3,
        "goal": "Extract laboratory results and flag pending tests",
        "tool": "lab_extractor",
        "required": True,
        "inputs_from_state": ["parsed_documents"],
        "depends_on": [2],
    },
    {
        "step_id": 4,
        "goal": "Detect conflicts between source notes",
        "tool": "conflict_detector",
        "required": True,
        "inputs_from_state": ["parsed_documents"],
        "depends_on": [2],
    },
    {
        "step_id": 5,
        "goal": "Reconcile admission vs discharge medications",
        "tool": "medication_reconciler",
        "required": True,
        "inputs_from_state": ["admission_medications", "discharge_medications"],
        "depends_on": [2],
    },
    {
        "step_id": 6,
        "goal": "Check discharge medications for drug interactions",
        "tool": "drug_interaction_lookup",
        "required": False,          # Non-blocking — no meds is OK
        "inputs_from_state": ["discharge_medications"],
        "depends_on": [],           # No hard dep — run regardless of med reconciler
    },
    {
        "step_id": 7,
        "goal": "Escalate any critical issues found so far",
        "tool": "escalate",
        "required": True,
        "inputs_from_state": ["conflicts", "drug_interaction_flags", "pending_results"],
        "depends_on": [3, 4],       # Only needs labs + conflicts — not meds
    },
    {
        "step_id": 8,
        "goal": "Synthesise hospital course narrative",
        "tool": "synthesise_hospital_course",
        "required": False,          # Degrades gracefully to MISSING
        "inputs_from_state": ["parsed_documents"],
        "depends_on": [2],
    },
    {
        "step_id": 9,
        "goal": "Assemble final discharge summary draft",
        "tool": "assemble_summary",
        "required": True,
        "inputs_from_state": ["all"],
        "depends_on": [7],          # Only hard dep is escalation step
    },
    {
        "step_id": 10,
        "goal": "Run fabrication guard scan on assembled summary",
        "tool": "fabrication_guard",
        "required": True,
        "inputs_from_state": ["draft_summary"],
        "depends_on": [9],
    },
]


class Planner:
    def __init__(self, llm_client=None):
        self.llm = llm_client

    def create_initial_plan(self, state: AgentState) -> List[Dict[str, Any]]:
        """
        Build the execution plan.
        Uses fixed template, optionally refined by LLM.
        """
        plan = [dict(step) for step in PLAN_TEMPLATE]  # deep copy

        # Mark all as pending
        for step in plan:
            step["status"] = "pending"
            step["retry_count"] = 0

        return plan

    def get_next_step(self, state: AgentState) -> Optional[Dict[str, Any]]:
        """
        Return the next executable step from the plan.
        A step is executable if:
          - status == "pending"
          - all dependencies are completed
        """
        for step in state.current_plan:
            if step.get("status") != "pending":
                continue

            deps = step.get("depends_on", [])
            deps_met = all(
                any(
                    s.get("step_id") == d and s.get("status") == "completed"
                    for s in state.current_plan
                )
                for d in deps
            )

            if deps_met or not deps:
                return step

        return None  # All steps done or blocked

    def mark_step_complete(
        self, state: AgentState, step_id: int
    ) -> None:
        for step in state.current_plan:
            if step["step_id"] == step_id:
                step["status"] = "completed"
                break

    def mark_step_failed(
        self, state: AgentState, step_id: int, reason: str
    ) -> None:
        for step in state.current_plan:
            if step["step_id"] == step_id:
                step["status"] = "failed"
                step["failure_reason"] = reason
                # Non-required failed steps are skipped; required ones are flagged
                if not step.get("required", True):
                    step["status"] = "skipped"
                break

    def should_terminate(self, state: AgentState) -> tuple[bool, str]:
        """
        Check if the agent should stop.
        Returns (should_stop, reason).
        """
        if state.step_count >= SETTINGS.max_steps:
            return True, f"step_cap_reached ({SETTINGS.max_steps} steps)"

        # Check if all steps are done or failed
        all_done = all(
            s.get("status") in ("completed", "failed", "skipped")
            for s in state.current_plan
        )
        if all_done:
            # Check if final assembly step completed
            for step in state.current_plan:
                if step["step_id"] == 9 and step.get("status") == "completed":
                    return True, "plan_complete"
            return True, "plan_exhausted_without_assembly"

        return False, ""

    def replan_on_failure(
        self, state: AgentState, failed_step_id: int
    ) -> None:
        """
        On failure: mark downstream steps as potentially affected.
        The next get_next_step call will find the next available step.
        """
        for step in state.current_plan:
            deps = step.get("depends_on", [])
            if failed_step_id in deps and step["status"] == "pending":
                # Mark as needing attention but don't block if other deps OK
                step.setdefault("warnings", []).append(
                    f"Dependency step {failed_step_id} failed"
                )
