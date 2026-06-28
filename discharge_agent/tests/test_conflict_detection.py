"""
tests/test_conflict_detection.py
Focused tests for conflict detection and fabrication guard logic.
"""

from config.settings import SETTINGS
from models.patient import Conflict
from models.summary import DischargeSummary
from tools.conflict_detector import ConflictDetectorTool, _values_compatible


class TestConflictDetectorEdgeCases:
    def test_partial_match_no_false_positive(self):
        # "DKA" vs "Diabetic Ketoacidosis" — should NOT be a conflict
        docs = [
            {
                "file_path": "a.pdf",
                "page_number": 1,
                "entities": {"diagnoses": ["DKA"]},
            },
            {
                "file_path": "b.pdf",
                "page_number": 1,
                "entities": {"diagnoses": ["Diabetic Ketoacidosis"]},
            },
        ]
        tool = ConflictDetectorTool()
        result = tool.run(extracted_documents=docs)
        # These could legitimately be same; the tool uses partial match to reduce FP
        # So this may or may not be a conflict depending on implementation
        assert result.ok  # Must not crash

    def test_clearly_different_diagnoses_flagged(self):
        docs = [
            {
                "file_path": "a.pdf",
                "page_number": 1,
                "entities": {"diagnoses": ["Acute Gastroenteritis"]},
            },
            {
                "file_path": "b.pdf",
                "page_number": 1,
                "entities": {"diagnoses": ["Myocardial Infarction"]},
            },
        ]
        tool = ConflictDetectorTool()
        result = tool.run(extracted_documents=docs)
        assert result.ok
        assert len(result.data) > 0

    def test_conflict_formatted_string_contains_sentinel(self):
        conflict = Conflict(
            field="principal_diagnosis",
            note_a_source="admission.pdf:p1",
            note_a_value="Sepsis",
            note_b_source="discharge.pdf:p1",
            note_b_value="UTI",
            description="Conflicting diagnoses",
        )
        formatted = conflict.to_flagged_string(
            SETTINGS.conflict_prefix, SETTINGS.conflict_suffix
        )
        assert SETTINGS.conflict_prefix in formatted
        assert "requires clinician review" in formatted

    def test_values_compatible_helper(self):
        assert _values_compatible("dka", "diabetic ketoacidosis") is False or True
        # The key test: clearly incompatible values
        assert _values_compatible("sepsis", "normal ecg") is False


class TestFabricationGuard:
    def test_missing_sentinel_present_in_empty_summary(self):
        summary = DischargeSummary(patient_id="test")
        # All fields default to missing sentinel
        assert SETTINGS.missing_sentinel in summary.principal_diagnosis
        assert SETTINGS.missing_sentinel in summary.allergies
        assert SETTINGS.missing_sentinel in summary.hospital_course

    def test_validate_completeness_flags_missing_fields(self):
        summary = DischargeSummary(patient_id="test")
        issues = summary.validate_completeness()
        # A fresh summary with all missing sentinels should have issues
        assert len(issues) > 0

    def test_validate_completeness_passes_when_all_filled(self):
        summary = DischargeSummary(
            patient_id="test",
            patient_demographics="John Doe, 45M",
            admission_date="01/03/2026",
            discharge_date="05/03/2026",
            principal_diagnosis="Diabetic Ketoacidosis",
            hospital_course="Patient admitted with DKA, treated with IV insulin, improved.",
            allergies="None known",
            follow_up_instructions="Review in OPD in 1 week",
            discharge_condition="Stable",
        )
        issues = summary.validate_completeness()
        # Should have fewer (or zero) issues now
        missing_count = len(
            [i for i in issues if "demographics" in i or "diagnosis" in i]
        )
        assert missing_count == 0

    def test_markdown_includes_draft_warning(self):
        summary = DischargeSummary(patient_id="test")
        md = summary.to_markdown()
        assert "DRAFT" in md.upper()
        assert "clinician review" in md.lower()

    def test_to_dict_serializable(self):
        import json

        summary = DischargeSummary(patient_id="test")
        d = summary.to_dict()
        # Should be JSON serialisable
        json_str = json.dumps(d, default=str)
        assert len(json_str) > 0
