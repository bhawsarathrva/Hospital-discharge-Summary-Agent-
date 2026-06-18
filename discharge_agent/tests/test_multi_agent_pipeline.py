"""
tests/test_multi_agent_pipeline.py
Unit tests for the modular multi-agent pipeline and graph workflow.
"""
import pytest
from agents.extractor import ExtractorAgent
from agents.medication_agent import MedicationAgent
from agents.safety_agent import SafetyAgent
from agents.summary_agent import SummaryAgent
from agents.planner import Planner
from workflows.graph import create_discharge_workflow
from discharge_agent.models.patient import Medication, Conflict

class DummyLLM:
    def __init__(self, response_text):
        self.response_text = response_text
        self.prompts_received = []

    def complete(self, prompt, max_tokens=None, temperature=None):
        self.prompts_received.append(prompt)
        return self.response_text

def test_medication_agent_reconciliation():
    agent = MedicationAgent(llm_client=None)
    state = {
        "admission_medications": [
            Medication(name="Metformin", dose="500mg")
        ],
        "discharge_medications": [
            Medication(name="Metformin", dose="1000mg")
        ]
    }
    updated_state = agent.run(state)
    changes = updated_state["medication_changes"]
    assert len(changes) == 1
    assert changes[0]["change_type"] == "dose_changed"
    assert any("Metformin" in f for f in updated_state["clinician_flags"])

def test_safety_agent_detection():
    agent = SafetyAgent(llm_client=None)
    state = {
        "discharge_medications": [
            Medication(name="Warfarin"),
            Medication(name="Aspirin")
        ],
        "pending_results": ["Sputum Culture"],
        "conflicts": [
            Conflict(
                field="principal_diagnosis",
                note_a_source="admission.pdf:p1",
                note_a_value="UTI",
                note_b_source="discharge.pdf:p1",
                note_b_value="Pneumonia",
                description="Conflicting diagnoses"
            )
        ],
        "clinician_flags": []
    }
    updated_state = agent.run(state)
    # High-severity drug interaction (Warfarin + Aspirin) + Conflict = 2 high flags
    # Pending result = 1 informational flag
    assert len(updated_state["clinician_flags"]) >= 3
    assert any("Warfarin + Aspirin" in f for f in updated_state["clinician_flags"])
    assert any("Conflicting diagnoses" in f for f in updated_state["clinician_flags"])

def test_summary_agent_assembly():
    agent = SummaryAgent(llm_client=None)
    state = {
        "patient_id": "PT-999",
        "demographics": {"name": "John Doe", "age": "45"},
        "admission_date": "2026-06-01",
        "discharge_date": "2026-06-05",
        "diagnoses": ["Pneumonia"],
        "discharge_medications": [],
        "medication_changes": [],
        "parsed_documents": [],
        "clinician_flags": []
    }
    updated_state = agent.run(state)
    summary = updated_state["_draft_summary"]
    assert summary["patient_id"] == "PT-999"
    assert "John Doe" in summary["patient_demographics"]
    assert summary["admission_date"] == "2026-06-01"

def test_planner_orchestrates_flow():
    planner = Planner(llm_client=None)
    state = {
        "patient_id": "PT-100",
        "patient_dir": "discharge_agent/data/patient_1",
        "parsed_documents": [],
        "raw_documents": [],
        "unreadable_files": [],
        "demographics": {},
        "admission_date": None,
        "discharge_date": None,
        "diagnoses": [],
        "discharge_diagnoses": [],
        "procedures": [],
        "admission_medications": [],
        "discharge_medications": [],
        "medication_changes": [],
        "allergies": None,
        "lab_results": [],
        "pending_results": [],
        "discharge_condition": None,
        "follow_up": None,
        "hospital_course": None,
        "conflicts": [],
        "clinician_flags": [],
        "drug_interaction_flags": [],
        "is_complete": False,
    }
    # Planner sequentially calls extractor -> conflict_detector -> medication_reconciler -> safety_auditor -> summary_assembler
    final_state = planner.run(state)
    assert final_state["is_complete"] is True
    assert "_draft_summary" in final_state

def test_state_graph_compiled_run():
    workflow = create_discharge_workflow(llm_client=None)
    state = {
        "patient_id": "PT-100",
        "patient_dir": "discharge_agent/data/patient_1",
        "parsed_documents": [],
        "raw_documents": [],
        "unreadable_files": [],
        "demographics": {},
        "admission_date": None,
        "discharge_date": None,
        "diagnoses": [],
        "discharge_diagnoses": [],
        "procedures": [],
        "admission_medications": [],
        "discharge_medications": [],
        "medication_changes": [],
        "allergies": None,
        "lab_results": [],
        "pending_results": [],
        "discharge_condition": None,
        "follow_up": None,
        "hospital_course": None,
        "conflicts": [],
        "clinician_flags": [],
        "drug_interaction_flags": [],
        "is_complete": False,
    }
    final_state = workflow.invoke(state)
    assert final_state["is_complete"] is True
    assert "_draft_summary" in final_state
