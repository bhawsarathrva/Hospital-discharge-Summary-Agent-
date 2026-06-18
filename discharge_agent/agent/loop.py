"""
agent/loop.py
Core ReAct agent loop.

Flow per iteration:
  1. THINK  — LLM reasons about current state and what to do next
  2. ACT    — Executor dispatches the chosen tool
  3. OBSERVE — Result updates AgentState
  4. RE-PLAN — Planner checks if plan needs adjustment

Hard limits enforced:
  - MAX_STEPS from settings (iteration cap)
  - Per-tool retry handled in Executor
"""
from __future__ import annotations

import json
import time
from typing import Optional, Tuple

from config.settings import SETTINGS
from agent.executor import Executor
from agent.planner import Planner
from agent.state import AgentState
from models.summary import DischargeSummary
from models.trace import AgentStep, AgentTrace, StepStatus
from tools.base import ToolStatus
from utils.llm_client import LLMClient

# Rich console for live trace output (optional)
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    _console = Console()
    _RICH = True
except ImportError:
    _RICH = False


class AgentLoop:
    """
    The central agent loop for one patient.
    Instantiate once per patient run.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        verbose: bool = True,
    ):
        self.llm = llm_client
        self.verbose = verbose
        self.planner = Planner(llm_client=llm_client)
        self.executor = Executor(llm_client=llm_client)

    def run(
        self,
        patient_id: str,
        patient_dir: str,
    ) -> Tuple[Optional[DischargeSummary], AgentTrace]:
        """
        Run the full agent loop for one patient.

        Returns:
            (DischargeSummary | None, AgentTrace)
            Summary is None only if a catastrophic failure occurred at step 1.
        """
        state = AgentState(
            patient_id=patient_id,
            patient_dir=patient_dir,
        )
        trace = AgentTrace(patient_id=patient_id)
        state.trace = trace

        self._log(f"\n{'='*60}")
        self._log(f"  AGENT START: {patient_id}")
        self._log(f"  Source dir : {patient_dir}")
        self._log(f"  Max steps  : {SETTINGS.max_steps}")
        self._log(f"{'='*60}\n")

        # ── 1. Create initial plan ──────────────────────────────────────────
        state.current_plan = self.planner.create_initial_plan(state)
        self._log(
            f"[PLAN] {len(state.current_plan)} steps created"
        )

        # ── 2. Main loop ────────────────────────────────────────────────────
        while True:
            state.increment_step()

            # Hard cap check
            should_stop, reason = self.planner.should_terminate(state)
            if should_stop:
                trace.finish(reason)
                self._log(f"\n[TERMINATE] {reason} after {state.step_count} steps")
                break

            # Get next executable step
            next_step = self.planner.get_next_step(state)
            if next_step is None:
                trace.finish("no_more_steps")
                self._log("\n[TERMINATE] No more executable steps")
                break

            # ── THINK: Agent reasons about what it's about to do ──────────
            reasoning = self._think(state, next_step)

            # ── ACT: Execute the step ─────────────────────────────────────
            self._log(
                f"\n── Step {state.step_count:02d} ─ {next_step['tool']} ─────────"
            )
            self._log(f"  GOAL    : {next_step['goal']}")
            self._log(f"  REASON  : {reasoning[:120]}")

            agent_step = self.executor.execute_step(
                step=next_step,
                state=state,
                step_number=state.step_count,
            )
            agent_step.reasoning = reasoning

            # ── OBSERVE & update plan ─────────────────────────────────────
            if agent_step.tool_status in (StepStatus.SUCCESS, StepStatus.PARTIAL):
                self.planner.mark_step_complete(state, next_step["step_id"])
                self._log(
                    f"  STATUS  : ✅ {agent_step.tool_status.value}"
                    f" ({agent_step.duration_s:.2f}s)"
                )
            else:
                self.planner.mark_step_failed(
                    state,
                    next_step["step_id"],
                    str(agent_step.tool_result),
                )
                self.planner.replan_on_failure(state, next_step["step_id"])
                self._log(
                    f"  STATUS  : ❌ FAILED — {str(agent_step.tool_result)[:100]}"
                )

            # Log flags
            if agent_step.flags_raised:
                self._log(f"  FLAGS   : {len(agent_step.flags_raised)} raised")
                for flag in agent_step.flags_raised[-3:]:   # show last 3
                    self._log(f"    🚩 {flag[:100]}")

            self._log(f"  NEXT    : {agent_step.next_decision[:100]}")

            trace.add_step(agent_step)

        # ── 3. Retrieve final summary ────────────────────────────────────
        summary = self.executor.get_final_summary(state)

        # Patch remaining missing sentinel fields
        if summary is None:
            self._log("\n[WARNING] No summary assembled — creating minimal shell")
            summary = self._minimal_summary(state)
            trace.finish("assembly_failed")

        self._log(
            f"\n{'='*60}\n"
            f"  COMPLETE: {patient_id}\n"
            f"  Steps   : {state.step_count}\n"
            f"  Flags   : {len(summary.clinician_flags)}\n"
            f"  Conflicts: {len(summary.conflicts_detected)}\n"
            f"  Fab scan: {'PASSED' if summary.fabrication_scan_passed else 'ISSUES FOUND'}\n"
            f"{'='*60}\n"
        )

        return summary, trace

    def _think(self, state: AgentState, next_step: dict) -> str:
        """
        LLM reasoning step.
        If LLM unavailable, returns a deterministic reasoning string.
        """
        if not self.llm:
            return (
                f"Plan step {next_step['step_id']}: '{next_step['goal']}'. "
                f"Completed steps: {[s['step_id'] for s in state.current_plan if s.get('status')=='completed']}. "
                f"Proceeding with tool '{next_step['tool']}'."
            )

        prompt = f"""
You are running step {state.step_count} of a discharge summary agent.

## Current State
- Patient ID: {state.patient_id}
- Documents loaded: {len(state.raw_documents)} pages
- Diagnoses found so far: {state.diagnoses[:3]}
- Meds found (admission): {len(state.admission_medications)}
- Meds found (discharge): {len(state.discharge_medications)}
- Conflicts detected: {len(state.conflicts)}
- Flags raised: {len(state.clinician_flags)}

## Next Step
Goal: {next_step['goal']}
Tool: {next_step['tool']}

## Reasoning
In 1-2 sentences, explain WHY this step is needed now and what you expect to find.
Do NOT invent clinical facts. Do NOT skip this step without a good reason.
"""
        try:
            return self.llm.complete(prompt, max_tokens=150).strip()
        except Exception:
            return f"Step {next_step['step_id']}: {next_step['goal']} — proceeding."

    def _minimal_summary(self, state: AgentState) -> DischargeSummary:
        """Emergency fallback summary when assembly step failed."""
        from models.summary import DischargeSummary
        M = SETTINGS.missing_sentinel
        return DischargeSummary(
            patient_id=state.patient_id,
            patient_demographics=M,
            admission_date=state.admission_date or M,
            discharge_date=state.discharge_date or M,
            principal_diagnosis=M,
            hospital_course=M,
            allergies=state.allergies or M,
            follow_up_instructions=state.follow_up or M,
            discharge_condition=state.discharge_condition or M,
            clinician_flags=[
                "⚠️ [CRITICAL] Summary assembly failed — all fields require manual entry"
            ]
            + state.clinician_flags,
            source_documents=state.unreadable_files,
            unreadable_documents=state.unreadable_files,
            fabrication_scan_passed=False,
            is_draft=True,
        )

    def _log(self, msg: str) -> None:
        if not self.verbose:
            return
        if _RICH:
            _console.print(msg)
        else:
            print(msg)
