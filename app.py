from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import argparse
import json
import os
import traceback
from pathlib import Path
from dotenv import load_dotenv

workspace_root = str(Path(__file__).resolve().parent)
discharge_agent_dir = str(Path(__file__).resolve().parent / "discharge_agent")
if discharge_agent_dir not in sys.path:
    sys.path.insert(0, discharge_agent_dir)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from workflows.graph import create_discharge_workflow
from discharge_agent.utils.llm_client import LLMClient
from config.settings import SETTINGS

import importlib.util


def _load_root_tool(name: str):
    root = Path(__file__).parent
    spec = importlib.util.spec_from_file_location(
        f"root_tools_{name}", str(root / "tools" / f"{name}.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


logger_mod = _load_root_tool("logger")
get_logger = logger_mod.get_logger

load_dotenv()

logger = get_logger("app")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Hospital Discharge Summary Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--patient",
        nargs="+",
        default=None,
        help="One or more patient directories containing PDFs (defaults to all subdirectories under 'discharge_agent/data/')",
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help="Output root directory (default: outputs/)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Run without LLM (keyword extraction only)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the default Claude model string",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    patient_dirs = args.patient
    if not patient_dirs:
        data_dir = Path("discharge_agent/data")
        if not data_dir.exists():
            data_dir = Path("data")
        if data_dir.exists() and data_dir.is_dir():
            patient_dirs = [
                str(p) for p in data_dir.iterdir() if p.is_dir() and p.name != "outputs"
            ]

        if not patient_dirs:
            logger.error(
                "No patient directories specified via --patient, and default 'data/' directory is empty or does not exist."
            )
            sys.exit(1)

    llm_client = None
    if not args.no_llm:
        model_name = args.model or SETTINGS.model
        if model_name.startswith("gemini-"):
            api_key = os.environ.get("GOOGLE_API_KEY", "") or os.environ.get(
                "GEMINI_API_KEY", ""
            )
            if not api_key:
                logger.warning(
                    "GOOGLE_API_KEY or GEMINI_API_KEY not set. Running in keyword-only mode (no LLM extraction)."
                )
        else:
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                logger.warning(
                    "ANTHROPIC_API_KEY not set. Running in keyword-only mode (no LLM extraction)."
                )

        if api_key:
            llm_client = LLMClient(api_key=api_key, model=model_name)
            logger.info(f"Using LLM model: {llm_client.model}")

    # Compile the graph workflow
    logger.info("Compiling StateGraph Workflow...")
    workflow = create_discharge_workflow(llm_client)

    results = []
    out_root = Path(args.output)
    summaries_dir = out_root / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    for p_dir in patient_dirs:
        p_path = Path(p_dir)
        patient_id = p_path.name
        logger.info(f"Processing patient: {patient_id} ({p_dir})")

        # Define initial state
        state = {
            "patient_id": patient_id,
            "patient_dir": str(p_path),
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

        try:
            # Run graph workflow
            final_state = workflow.invoke(state)
            summary_dict = final_state.get("_draft_summary")

            if summary_dict:
                # Save structured JSON summary
                summary_json_path = summaries_dir / f"{patient_id}_summary.json"
                with open(summary_json_path, "w", encoding="utf-8") as f:
                    json.dump(summary_dict, f, indent=2, default=str)

                logger.info(
                    f"Successfully generated summary for {patient_id} -> {summary_json_path}"
                )

                # Save to MongoDB if configured
                try:
                    from discharge_agent.utils.db import (
                        save_patient_summary,
                        save_patient_state,
                    )

                    save_patient_summary(patient_id, summary_dict)
                    save_patient_state(patient_id, final_state)
                except Exception as db_exc:
                    logger.warning(f"Could not save patient data to MongoDB: {db_exc}")

                results.append(
                    {
                        "patient_id": patient_id,
                        "status": "success",
                        "output": str(summary_json_path),
                    }
                )
            else:
                logger.error(
                    f"Workflow finished but no draft summary was assembled for {patient_id}"
                )
                results.append({"patient_id": patient_id, "status": "assembly_failed"})

        except Exception as exc:
            logger.error(f"Fatal error processing patient {patient_id}: {exc}")
            traceback.print_exc()
            results.append(
                {"patient_id": patient_id, "status": "fatal_error", "error": str(exc)}
            )

    # Write run manifest
    manifest_path = out_root / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"All processing complete. Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
