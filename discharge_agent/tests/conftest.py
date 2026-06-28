"""
tests/conftest.py
Shared pytest fixtures.
"""

import sys
import tempfile
from pathlib import Path

import pytest

# Ensure the project root and workspace root are on PYTHONPATH
project_root = str(Path(__file__).resolve().parent.parent)
workspace_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)


@pytest.fixture
def temp_patient_dir():
    """Empty temp directory simulating a patient folder with no PDFs."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir


@pytest.fixture
def sample_lab_doc():
    """A parsed document dict representing a lab result page."""
    return {
        "file_path": "labs.pdf",
        "page_number": 1,
        "note_type": "lab_result",
        "raw_text": (
            "HAEMATOLOGY REPORT\n"
            "Haemoglobin (HB) 10.7 gm/dl 13.5-17.5\n"
            "Total Count (WBC) 11560 Cells/cumm 4000-11000\n"
            "BIOCHEMISTRY REPORT\n"
            "Serum Creatinine 1.65 mg/dL 0.7-1.4\n"
            "S.SODIUM 128.00 mmol/L 135-150\n"
            "Random Blood Sugar (RBS) 443 mg/dL 80-140\n"
            "Urine culture and sensitivity sent- report awaited.\n"
        ),
        "entities": {},
    }


@pytest.fixture
def sample_admission_doc():
    """A parsed document dict representing an admission note."""
    return {
        "file_path": "admission.pdf",
        "page_number": 1,
        "note_type": "admission_note",
        "raw_text": (
            "CASE RECORD / ADMISSION RECORD (1)\n"
            "Chief Complaints: Fever, Generalized weakness Since 3 days, Myalgia.\n"
            "Past History / Drug History: K/c/o T2DM on Ayurvedic Medication\n"
            "HbA1c - 13.9% (outside report)\n"
            "Allergic History: Not Known\n"
        ),
        "entities": {
            "patient_age": "45",
            "allergies": "Not Known",
            "diagnoses": ["Uncontrolled T2DM", "DKA"],
        },
    }


@pytest.fixture
def sample_medication_list():
    """Sample medication objects."""
    from models.patient import Medication

    return [
        Medication(name="Metformin", dose="500mg", route="oral", frequency="1-0-1"),
        Medication(name="Glimepiride", dose="2mg", route="oral", frequency="1-0-0"),
        Medication(name="Pantoprazole", dose="40mg", route="IV", frequency="1-0-0"),
    ]
