#!/usr/bin/env python3
from __future__ import annotations

import sys
# Configure stdout/stderr to use UTF-8 to avoid encoding crashes on Windows
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
from config.settings import SETTINGS

load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Discharge Summary Agent — Part 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--patient",
        nargs="+",
        default=None,
        help="One or more patient directories containing PDFs (defaults to all subdirectories under 'data/')",
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help="Output root directory (default: outputs/)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress step-by-step console output",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Run without LLM (keyword extraction only — for testing)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the default Claude model string",
    )
    return parser.parse_args()


def run_patient(
    patient_dir: str,
    output_dir: str,
    llm_client,
    verbose: bool = True,
) -> dict:
    """Run the agent for one patient. Returns result metadata."""
    from agent.loop import AgentLoop

    patient_path = Path(patient_dir)
    if not patient_path.exists():
        print(f"[ERROR] Patient directory not found: {patient_dir}")
        return {"patient_id": patient_dir, "status": "directory_not_found"}

    patient_id = patient_path.name
    out_root = Path(output_dir)
    traces_dir = out_root / "traces"
    summaries_dir = out_root / "summaries"
    traces_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Processing patient: {patient_id}")
    print(f"  Source: {patient_dir}")
    print(f"{'='*60}")

    loop = AgentLoop(llm_client=llm_client, verbose=verbose)

    try:
        summary, trace = loop.run(
            patient_id=patient_id,
            patient_dir=str(patient_path),
        )
    except Exception as exc:
        print(f"[FATAL] Agent loop crashed for {patient_id}: {exc}")
        traceback.print_exc()
        return {"patient_id": patient_id, "status": "fatal_error", "error": str(exc)}

    # 1. JSON summary
    summary_json_path = summaries_dir / f"{patient_id}_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2, default=str)

    # 2. Markdown summary
    summary_md_path = summaries_dir / f"{patient_id}_summary.md"
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(summary.to_markdown())

    # 3. JSON trace
    trace_path = traces_dir / f"{patient_id}_trace.json"
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(trace.to_dict(), f, indent=2, default=str)

    # 4. Readable trace text
    trace_txt_path = traces_dir / f"{patient_id}_trace.txt"
    with open(trace_txt_path, "w", encoding="utf-8") as f:
        f.write(trace.to_readable())

    print(f"\n  ✅ Summary   → {summary_json_path}")
    print(f"  ✅ Markdown  → {summary_md_path}")
    print(f"  ✅ Trace     → {trace_path}")
    print(f"  ✅ Trace txt → {trace_txt_path}")
    print(f"\n  Flags raised : {len(summary.clinician_flags)}")
    print(f"  Conflicts    : {len(summary.conflicts_detected)}")
    print(f"  Pending labs : {len(summary.pending_results)}")
    print(f"  Fab scan     : {'PASSED' if summary.fabrication_scan_passed else '⚠️  ISSUES'}")

    return {
        "patient_id": patient_id,
        "status": "success",
        "flags": len(summary.clinician_flags),
        "conflicts": len(summary.conflicts_detected),
        "pending_results": len(summary.pending_results),
        "fabrication_scan_passed": summary.fabrication_scan_passed,
        "outputs": {
            "summary_json": str(summary_json_path),
            "summary_md": str(summary_md_path),
            "trace_json": str(trace_path),
            "trace_txt": str(trace_txt_path),
        },
    }


def main():
    args = parse_args()

    patient_dirs = args.patient
    if not patient_dirs:
        # Default to all subdirectories under 'data' directory
        data_dir = Path("data")
        if data_dir.exists() and data_dir.is_dir():
            patient_dirs = [str(p) for p in data_dir.iterdir() if p.is_dir() and p.name != "outputs"]
        
        if not patient_dirs:
            print("[ERROR] No patient directories specified via --patient, and default 'data/' directory is empty or does not exist.")
            sys.exit(1)

    llm_client = None
    if not args.no_llm:
        model_name = args.model or SETTINGS.model
        if model_name.startswith("gemini-"):
            api_key = os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                print(
                    "[WARNING] GOOGLE_API_KEY or GEMINI_API_KEY not set. "
                    "Running in keyword-only mode (no LLM extraction).\n"
                    "Set the key or use --no-llm to suppress this warning."
                )
        else:
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                print(
                    "[WARNING] ANTHROPIC_API_KEY not set. "
                    "Running in keyword-only mode (no LLM extraction).\n"
                    "Set the key or use --no-llm to suppress this warning."
                )
        
        if api_key:
            from utils.llm_client import LLMClient
            llm_client = LLMClient(
                api_key=api_key,
                model=model_name,
            )
            print(f"[LLM] Using model: {llm_client.model}")

    results = []
    for patient_dir in patient_dirs:
        result = run_patient(
            patient_dir=patient_dir,
            output_dir=args.output,
            llm_client=llm_client,
            verbose=not args.quiet,
        )
        results.append(result)

    print(f"\n{'='*60}")
    print(f"  RUN COMPLETE — {len(results)} patient(s) processed")
    print(f"{'='*60}")
    for r in results:
        status_icon = "✅" if r["status"] == "success" else "❌"
        print(
            f"  {status_icon} {r['patient_id']}: "
            f"flags={r.get('flags', 'N/A')}, "
            f"conflicts={r.get('conflicts', 'N/A')}, "
            f"pending={r.get('pending_results', 'N/A')}"
        )

    # Save run manifest
    manifest_path = Path(args.output) / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Manifest → {manifest_path}")

    # Exit code: 0 if all succeeded
    if all(r["status"] == "success" for r in results):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
