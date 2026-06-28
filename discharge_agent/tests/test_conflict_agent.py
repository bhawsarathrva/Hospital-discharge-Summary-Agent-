"""
tests/test_conflict_agent.py
Tests for the root level ConflictAgent class.
"""

from agents.conflict_agent import ConflictAgent


class DummyLLM:
    def __init__(self, response_text):
        self.response_text = response_text
        self.prompts_received = []

    def complete(self, prompt, max_tokens=None, temperature=None):
        self.prompts_received.append(prompt)
        return self.response_text


def test_conflict_agent_rule_based_diagnoses():
    agent = ConflictAgent(llm_client=None)
    state = {
        "parsed_documents": [
            {
                "file_path": "admission.pdf",
                "page_number": 1,
                "entities": {"diagnoses": ["Acute Gastroenteritis"]},
            },
            {
                "file_path": "discharge.pdf",
                "page_number": 1,
                "entities": {"diagnoses": ["Myocardial Infarction"]},
            },
        ]
    }

    updated_state = agent.run(state)
    conflicts = updated_state["conflicts"]

    assert len(conflicts) > 0
    assert any(c["field"] == "diagnoses" for c in conflicts)
    assert any("admission.pdf" in c["note_a_source"] for c in conflicts)
    assert any("discharge.pdf" in c["note_b_source"] for c in conflicts)


def test_conflict_agent_rule_based_no_conflict():
    agent = ConflictAgent(llm_client=None)
    state = {
        "parsed_documents": [
            {
                "file_path": "admission.pdf",
                "page_number": 1,
                "entities": {"diagnoses": ["Diabetes"]},
            },
            {
                "file_path": "discharge.pdf",
                "page_number": 1,
                "entities": {"diagnoses": ["Diabetes Mellitus"]},
            },
        ]
    }

    updated_state = agent.run(state)
    conflicts = updated_state["conflicts"]

    # "Diabetes" and "Diabetes Mellitus" are compatible (one contains substring of another)
    assert len(conflicts) == 0


def test_conflict_agent_llm_based_semantic_conflict():
    mock_llm_response = """[
        {
            "field": "principal_diagnosis",
            "note_a_source": "admission.pdf:p1",
            "note_a_value": "Ischemic Stroke",
            "note_b_source": "discharge.pdf:p1",
            "note_b_value": "Hemorrhagic Stroke",
            "description": "Mutually exclusive stroke types: Ischemic vs Hemorrhagic"
        }
    ]"""
    llm = DummyLLM(mock_llm_response)
    agent = ConflictAgent(llm_client=llm)

    state = {
        "parsed_documents": [
            {
                "file_path": "admission.pdf",
                "page_number": 1,
                "entities": {"diagnoses": ["Ischemic Stroke"]},
            },
            {
                "file_path": "discharge.pdf",
                "page_number": 1,
                "entities": {"diagnoses": ["Hemorrhagic Stroke"]},
            },
        ]
    }

    updated_state = agent.run(state)
    conflicts = updated_state["conflicts"]

    # Should detect 2 conflicts (rule-based conflict for diagnoses, and LLM-based for principal_diagnosis)
    assert len(conflicts) >= 1
    assert any(c["field"] == "principal_diagnosis" for c in conflicts)
    assert any("Mutually exclusive stroke types" in c["description"] for c in conflicts)
