"""
tests/test_db.py
Unit tests for the MongoDB helper utilities.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch


# Adjust PYTHONPATH
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils import db


def test_handles_offline_or_missing_config():
    """Verify functions return False gracefully when MongoDB is not configured or offline."""
    with patch("utils.db.get_db", return_value=None):
        assert db.save_patient_summary("p1", {"data": "test"}) is False
        assert db.save_patient_state("p1", {"data": "test"}) is False
        assert db.save_execution_trace("p1", {"data": "test"}) is False


def test_save_patient_summary_calls_pymongo():
    """Verify save_patient_summary correctly indexes and calls update_one using pymongo."""
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_db.__getitem__.return_value = mock_collection

    with patch("utils.db.get_db", return_value=mock_db):
        res = db.save_patient_summary("patient_123", {"field": "value"})
        assert res is True

        # Verify collection access
        mock_db.__getitem__.assert_called_with("patient_summaries")

        # Verify index creation
        mock_collection.create_index.assert_called_with("patient_id", unique=True)

        # Verify update_one call
        mock_collection.update_one.assert_called_once()
        call_args, call_kwargs = mock_collection.update_one.call_args
        assert call_args[0] == {"patient_id": "patient_123"}
        assert call_args[1]["$set"]["patient_id"] == "patient_123"
        assert call_args[1]["$set"]["summary"] == {"field": "value"}
        assert "updated_at" in call_args[1]["$set"]


def test_save_patient_state_calls_pymongo():
    """Verify save_patient_state serializes and stores the state."""
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_db.__getitem__.return_value = mock_collection

    with patch("utils.db.get_db", return_value=mock_db):
        res = db.save_patient_state(
            "patient_123", {"demographics": {"name": "Bob"}, "_draft_summary": "secret"}
        )
        assert res is True

        mock_db.__getitem__.assert_called_with("patient_states")
        mock_collection.create_index.assert_called_with("patient_id", unique=True)

        mock_collection.update_one.assert_called_once()
        call_args, call_kwargs = mock_collection.update_one.call_args
        assert call_args[0] == {"patient_id": "patient_123"}
        # Check that private fields like _draft_summary are stripped
        assert "demographics" in call_args[1]["$set"]["state"]
        assert "_draft_summary" not in call_args[1]["$set"]["state"]


def test_save_execution_trace_calls_pymongo():
    """Verify save_execution_trace stores trace data."""
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_db.__getitem__.return_value = mock_collection

    with patch("utils.db.get_db", return_value=mock_db):
        res = db.save_execution_trace("patient_123", {"steps": []})
        assert res is True

        mock_db.__getitem__.assert_called_with("execution_traces")
        mock_collection.update_one.assert_called_once()
        call_args = mock_collection.update_one.call_args[0]
        assert call_args[0] == {"patient_id": "patient_123"}
        assert call_args[1]["$set"]["trace"] == {"steps": []}
