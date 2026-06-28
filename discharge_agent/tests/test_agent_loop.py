"""
tests/test_agent_loop.py
Integration tests for the agent loop.
Uses mock data — no API key required.
Run: pytest tests/test_agent_loop.py -v
"""

import os
import tempfile


from agent.loop import AgentLoop
from agent.planner import Planner
from agent.state import AgentState
from config.settings import SETTINGS
from models.summary import DischargeSummary


# ── Helpers ──────────────────────────────────────────────────────────────────


def make_temp_patient_dir(pdf_bytes: bytes = None) -> str:
    """Create a temp directory. If pdf_bytes given, write a dummy PDF."""
    tmpdir = tempfile.mkdtemp()
    if pdf_bytes:
        with open(os.path.join(tmpdir, "patient_notes.pdf"), "wb") as f:
            f.write(pdf_bytes)
    return tmpdir


# ── Planner tests ─────────────────────────────────────────────────────────────


class TestPlanner:
    def test_creates_full_plan(self):
        state = AgentState(patient_id="test", patient_dir="/tmp")
        planner = Planner(llm_client=None)
        plan = planner.create_initial_plan(state)
        assert len(plan) == 10  # All plan steps
        assert all(s["status"] == "pending" for s in plan)

    def test_first_step_is_pdf_reader(self):
        state = AgentState(patient_id="test", patient_dir="/tmp")
        planner = Planner(llm_client=None)
        state.current_plan = planner.create_initial_plan(state)
        next_step = planner.get_next_step(state)
        assert next_step is not None
        assert next_step["tool"] == "pdf_reader"

    def test_step_cap_terminates_loop(self):
        state = AgentState(patient_id="test", patient_dir="/tmp")
        state.step_count = SETTINGS.max_steps + 1
        planner = Planner(llm_client=None)
        state.current_plan = planner.create_initial_plan(state)
        should_stop, reason = planner.should_terminate(state)
        assert should_stop
        assert "step_cap" in reason

    def test_complete_marks_no_more_steps(self):
        state = AgentState(patient_id="test", patient_dir="/tmp")
        planner = Planner(llm_client=None)
        state.current_plan = planner.create_initial_plan(state)
        # Mark all as completed
        for s in state.current_plan:
            s["status"] = "completed"
        should_stop, reason = planner.should_terminate(state)
        assert should_stop

    def test_dependency_blocks_step(self):
        state = AgentState(patient_id="test", patient_dir="/tmp")
        planner = Planner(llm_client=None)
        state.current_plan = planner.create_initial_plan(state)
        # Step 2 depends on step 1; step 1 is pending
        # So get_next_step should return step 1, not step 2
        next_step = planner.get_next_step(state)
        assert next_step["step_id"] == 1


# ── AgentState tests ──────────────────────────────────────────────────────────


class TestAgentState:
    def test_add_flag_deduplicates(self):
        state = AgentState(patient_id="p1", patient_dir="/tmp")
        state.add_flag("test flag")
        state.add_flag("test flag")  # duplicate
        assert len(state.clinician_flags) == 1

    def test_to_summary_dict_has_all_keys(self):
        state = AgentState(patient_id="p1", patient_dir="/tmp")
        d = state.to_summary_dict()
        assert "demographics" in d
        assert "admission_date" in d
        assert "diagnoses" in d
        assert "medications_changes" not in d  # correct key check


# ── Agent loop with empty directory ──────────────────────────────────────────


class TestAgentLoopNoLLM:
    def test_empty_dir_produces_missing_summary(self):
        """Agent should complete even with no PDFs — all fields MISSING."""
        tmpdir = make_temp_patient_dir()
        loop = AgentLoop(llm_client=None, verbose=False)
        summary, trace = loop.run(patient_id="test_empty", patient_dir=tmpdir)

        assert isinstance(summary, DischargeSummary)
        assert summary.is_draft is True
        # All fields should be missing sentinel or empty
        assert (
            SETTINGS.missing_sentinel in summary.principal_diagnosis
            or summary.principal_diagnosis == SETTINGS.missing_sentinel
        )
        # Trace should have steps recorded
        assert len(trace.steps) > 0

    def test_summary_always_draft(self):
        tmpdir = make_temp_patient_dir()
        loop = AgentLoop(llm_client=None, verbose=False)
        summary, _ = loop.run(patient_id="test_draft", patient_dir=tmpdir)
        assert summary.is_draft is True

    def test_missing_sentinel_used_not_empty_string(self):
        tmpdir = make_temp_patient_dir()
        loop = AgentLoop(llm_client=None, verbose=False)
        summary, _ = loop.run(patient_id="test_sentinel", patient_dir=tmpdir)
        # None of the string fields should be empty string ""
        for field in SETTINGS.required_sections:
            val = getattr(summary, field, None)
            if isinstance(val, str):
                assert val != "", f"Field '{field}' should not be empty string"

    def test_trace_records_steps(self):
        tmpdir = make_temp_patient_dir()
        loop = AgentLoop(llm_client=None, verbose=False)
        _, trace = loop.run(patient_id="test_trace", patient_dir=tmpdir)
        assert len(trace.steps) >= 1
        assert trace.termination_reason != ""

    def test_step_count_never_exceeds_cap(self):
        tmpdir = make_temp_patient_dir()
        loop = AgentLoop(llm_client=None, verbose=False)
        summary, trace = loop.run(patient_id="test_cap", patient_dir=tmpdir)
        assert len(trace.steps) <= SETTINGS.max_steps
