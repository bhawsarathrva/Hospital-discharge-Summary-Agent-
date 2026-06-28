import os
import sys
import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project paths are in sys.path if run directly
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv

load_dotenv()

try:
    import pymongo

    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False

_client = None


def get_mongo_client() -> Optional[pymongo.MongoClient]:
    global _client
    if not PYMONGO_AVAILABLE:
        return None

    mongo_uri = os.environ.get("MONGO_URI")
    if not mongo_uri:
        # Silently return None if MongoDB is not configured
        return None

    if _client is not None:
        return _client

    try:
        # Create client with a 3-second connection timeout to avoid hanging the app
        _client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
        # Check connection
        _client.admin.command("ping")
        return _client
    except Exception as exc:
        print(f"[WARNING] Failed to connect to MongoDB at {mongo_uri}: {exc}")
        _client = None
        return None


def get_db():
    client = get_mongo_client()
    if not client:
        return None
    db_name = os.environ.get("MONGO_DB_NAME", "discharge_summary_agent")
    return client[db_name]


def save_patient_summary(patient_id: str, summary_data: Dict[str, Any]) -> bool:
    """Save or update the generated patient summary in MongoDB."""
    db = get_db()
    if db is None:
        return False
    try:
        collection = db["patient_summaries"]
        collection.create_index("patient_id", unique=True)

        doc = {
            "patient_id": patient_id,
            "summary": summary_data,
            "updated_at": datetime.datetime.now(datetime.timezone.utc),
        }
        collection.update_one({"patient_id": patient_id}, {"$set": doc}, upsert=True)
        print(f"[DB] Successfully saved summary for patient '{patient_id}' to MongoDB.")
        return True
    except Exception as exc:
        print(
            f"[ERROR] Failed to save summary for patient '{patient_id}' to MongoDB: {exc}"
        )
        return False


def save_patient_state(patient_id: str, state_data: Dict[str, Any]) -> bool:
    """Save or update the intermediate/final compiled patient state in MongoDB."""
    db = get_db()
    if db is None:
        return False
    try:
        collection = db["patient_states"]
        collection.create_index("patient_id", unique=True)

        # Prepare serializable state data by stripping out private keys starting with '_'
        clean_state = {k: v for k, v in state_data.items() if not k.startswith("_")}

        doc = {
            "patient_id": patient_id,
            "state": clean_state,
            "updated_at": datetime.datetime.now(datetime.timezone.utc),
        }
        collection.update_one({"patient_id": patient_id}, {"$set": doc}, upsert=True)
        print(
            f"[DB] Successfully saved compiled state for patient '{patient_id}' to MongoDB."
        )
        return True
    except Exception as exc:
        print(
            f"[ERROR] Failed to save compiled state for patient '{patient_id}' to MongoDB: {exc}"
        )
        return False


def save_execution_trace(patient_id: str, trace_data: Dict[str, Any]) -> bool:
    """Save or update the agent execution trace in MongoDB."""
    db = get_db()
    if db is None:
        return False
    try:
        collection = db["execution_traces"]
        collection.create_index("patient_id", unique=True)

        doc = {
            "patient_id": patient_id,
            "trace": trace_data,
            "updated_at": datetime.datetime.now(datetime.timezone.utc),
        }
        collection.update_one({"patient_id": patient_id}, {"$set": doc}, upsert=True)
        print(
            f"[DB] Successfully saved execution trace for patient '{patient_id}' to MongoDB."
        )
        return True
    except Exception as exc:
        print(
            f"[ERROR] Failed to save execution trace for patient '{patient_id}' to MongoDB: {exc}"
        )
        return False
