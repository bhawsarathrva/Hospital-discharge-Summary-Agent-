"""
tests/test_tools.py
Unit tests for all tools.
Run: pytest tests/test_tools.py -v
"""

from models.patient import Medication, MedicationStatus, PatientDocument
from tools.base import ToolStatus
from tools.conflict_detector import ConflictDetectorTool
from tools.document_parser import DocumentParserTool
from tools.drug_interaction import DrugInteractionTool
from tools.escalation import EscalationTool
from tools.lab_extractor import LabExtractorTool
from tools.medication_reconciler import MedicationReconcilerTool


# ── DocumentParser ──────────────────────────────────────────────────────────


class TestDocumentParser:
    def setup_method(self):
        self.tool = DocumentParserTool(llm_client=None)

    def test_classifies_admission_note(self):
        doc = PatientDocument(
            file_path="test.pdf",
            page_number=1,
            note_type="unknown",
            raw_text="ADMISSION RECORD Chief Complaints: Fever since 3 days. History of Present Illness.",
        )
        result = self.tool.run(documents=[doc], use_llm=False)
        assert result.ok
        classified = result.data[0]
        assert classified["note_type"] == "admission_note"

    def test_classifies_lab_result(self):
        doc = PatientDocument(
            file_path="test.pdf",
            page_number=1,
            note_type="unknown",
            raw_text="CLINICAL PATHOLOGY REPORT Haematology Complete Blood Count Haemoglobin 11.4 gm/dl",
        )
        result = self.tool.run(documents=[doc], use_llm=False)
        assert result.ok
        assert result.data[0]["note_type"] == "lab_result"

    def test_empty_documents_returns_not_found(self):
        result = self.tool.run(documents=[], use_llm=False)
        assert result.status == ToolStatus.NOT_FOUND

    def test_unreadable_page_handled(self):
        doc = PatientDocument(
            file_path="test.pdf",
            page_number=1,
            note_type="unknown",
            raw_text="",
            read_error="Page extraction failed",
        )
        result = self.tool.run(documents=[doc], use_llm=False)
        assert result.ok  # Should succeed with partial data


# ── LabExtractor ────────────────────────────────────────────────────────────


class TestLabExtractor:
    def setup_method(self):
        self.tool = LabExtractorTool()

    def test_extracts_haemoglobin(self):
        doc = {
            "file_path": "test.pdf",
            "page_number": 1,
            "note_type": "lab_result",
            "raw_text": "HAEMATOLOGY REPORT Haemoglobin (HB) 11.4 gm/dl 13.5 - 17.5",
            "entities": {},
        }
        result = self.tool.run(documents=[doc])
        assert result.ok
        labs = result.data["labs"]
        hb_labs = [
            l
            for l in labs
            if "Haemoglobin" in l.test_name or "haemoglobin" in l.test_name.lower()
        ]
        assert len(hb_labs) > 0
        assert hb_labs[0].value is not None

    def test_extracts_creatinine(self):
        doc = {
            "file_path": "test.pdf",
            "page_number": 1,
            "note_type": "lab_result",
            "raw_text": "BIOCHEMISTRY REPORT Serum Creatinine 1.65 mg/dL Reference: 0.7 - 1.4",
            "entities": {},
        }
        result = self.tool.run(documents=[doc])
        assert result.ok
        labs = result.data["labs"]
        creat = [l for l in labs if "creatinine" in l.test_name.lower()]
        assert len(creat) > 0

    def test_detects_pending_result(self):
        doc = {
            "file_path": "test.pdf",
            "page_number": 1,
            "note_type": "lab_result",
            "raw_text": "Urine culture and sensitivity sent- report awaited.",
            "entities": {},
        }
        result = self.tool.run(documents=[doc])
        assert result.ok
        assert len(result.data["pending"]) > 0

    def test_flags_abnormal_sodium(self):
        doc = {
            "file_path": "test.pdf",
            "page_number": 1,
            "note_type": "lab_result",
            "raw_text": "S.Sodium 128.00 mmol/L Reference: 135-150",
            "entities": {},
        }
        result = self.tool.run(documents=[doc])
        assert result.ok
        labs = result.data["labs"]
        sodium = [l for l in labs if "sodium" in l.test_name.lower()]
        if sodium:
            assert sodium[0].is_abnormal is True


# ── MedicationReconciler ────────────────────────────────────────────────────


class TestMedicationReconciler:
    def setup_method(self):
        self.tool = MedicationReconcilerTool()

    def test_detects_added_medication(self):
        adm = [Medication(name="Metformin", dose="500mg")]
        dis = [
            Medication(name="Metformin", dose="500mg"),
            Medication(name="Amoxicillin", dose="500mg"),
        ]
        result = self.tool.run(admission_meds=adm, discharge_meds=dis)
        assert result.ok
        changes = result.data
        added = [c for c in changes if c.change_type == MedicationStatus.ADDED]
        assert any("amoxicillin" in c.medication.name.lower() for c in added)

    def test_detects_stopped_medication(self):
        adm = [
            Medication(name="Warfarin", dose="5mg"),
            Medication(name="Aspirin", dose="75mg"),
        ]
        dis = [Medication(name="Warfarin", dose="5mg")]
        result = self.tool.run(admission_meds=adm, discharge_meds=dis)
        assert result.ok
        changes = result.data
        stopped = [c for c in changes if c.change_type == MedicationStatus.STOPPED]
        assert any("aspirin" in c.medication.name.lower() for c in stopped)

    def test_flags_stopped_without_reason(self):
        adm = [Medication(name="Ramipril", dose="5mg", change_reason=None)]
        dis = []
        result = self.tool.run(admission_meds=adm, discharge_meds=dis)
        assert result.ok
        flags = result.metadata.get("flags", [])
        assert any("ramipril" in f.lower() for f in flags)

    def test_empty_lists_returns_not_found(self):
        result = self.tool.run(admission_meds=[], discharge_meds=[])
        assert result.status == ToolStatus.NOT_FOUND


# ── ConflictDetector ────────────────────────────────────────────────────────


class TestConflictDetector:
    def setup_method(self):
        self.tool = ConflictDetectorTool()

    def test_detects_diagnosis_conflict(self):
        docs = [
            {
                "file_path": "admission_note.pdf",
                "page_number": 1,
                "entities": {"diagnoses": ["Acute Gastroenteritis"]},
            },
            {
                "file_path": "discharge_summary.pdf",
                "page_number": 1,
                "entities": {"diagnoses": ["Diabetic Ketoacidosis"]},
            },
        ]
        result = self.tool.run(extracted_documents=docs)
        assert result.ok
        conflicts = result.data
        assert len(conflicts) > 0

    def test_no_conflict_same_diagnosis(self):
        docs = [
            {
                "file_path": "note1.pdf",
                "page_number": 1,
                "entities": {"diagnoses": ["Urinary Tract Infection"]},
            },
            {
                "file_path": "note2.pdf",
                "page_number": 1,
                "entities": {"diagnoses": ["Urinary Tract Infection"]},
            },
        ]
        result = self.tool.run(extracted_documents=docs)
        assert result.ok
        assert len(result.data) == 0

    def test_empty_docs_returns_not_found(self):
        result = self.tool.run(extracted_documents=[])
        assert result.status == ToolStatus.NOT_FOUND


# ── DrugInteraction ─────────────────────────────────────────────────────────


class TestDrugInteraction:
    def setup_method(self):
        self.tool = DrugInteractionTool()

    def test_detects_warfarin_aspirin(self):
        result = self.tool.run(medication_names=["Warfarin", "Aspirin", "Metformin"])
        assert result.ok
        interactions = result.data
        high = [i for i in interactions if i["severity"] == "HIGH"]
        assert len(high) > 0

    def test_no_interaction_safe_combo(self):
        result = self.tool.run(medication_names=["Paracetamol", "Omeprazole"])
        assert result.ok
        assert len(result.data) == 0

    def test_empty_list_returns_not_found(self):
        result = self.tool.run(medication_names=[])
        assert result.status == ToolStatus.NOT_FOUND

    def test_high_severity_requires_escalation(self):
        result = self.tool.run(medication_names=["Meropenem", "Valproate"])
        assert result.ok
        high = [i for i in result.data if i["severity"] == "HIGH"]
        assert all(i["requires_escalation"] for i in high)


# ── EscalationTool ───────────────────────────────────────────────────────────


class TestEscalation:
    def setup_method(self):
        self.tool = EscalationTool()

    def test_records_flag(self):
        result = self.tool.run(
            severity="high",
            field="principal_diagnosis",
            message="Conflicting diagnoses found",
            source_evidence="admission_note vs discharge_summary",
        )
        assert result.ok
        flags = self.tool.get_all_flags()
        assert len(flags) == 1
        assert "principal_diagnosis" in flags[0]

    def test_critical_flag_in_critical_list(self):
        self.tool.reset()
        self.tool.run(
            severity="critical",
            field="drug_interactions",
            message="HIGH drug interaction",
        )
        critical = self.tool.get_critical_flags()
        assert len(critical) == 1

    def test_reset_clears_flags(self):
        self.tool.run(severity="high", field="test", message="test")
        self.tool.reset()
        assert len(self.tool.get_all_flags()) == 0

    def test_invalid_severity_defaults_to_informational(self):
        result = self.tool.run(severity="unknown_level", field="test", message="test")
        assert result.ok
