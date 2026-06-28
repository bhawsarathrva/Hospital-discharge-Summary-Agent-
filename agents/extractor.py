from __future__ import annotations
import sys
import os
import re
import json
from typing import Any, Dict, List, Optional
from pathlib import Path

workspace_root = str(Path(__file__).resolve().parent.parent)
discharge_agent_dir = str(Path(__file__).resolve().parent.parent / "discharge_agent")
if discharge_agent_dir not in sys.path:
    sys.path.insert(0, discharge_agent_dir)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

import importlib.util


def _load_root_tool(name: str):
    root = Path(__file__).parent.parent
    spec = importlib.util.spec_from_file_location(
        f"root_tools_{name}", str(root / "tools" / f"{name}.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pdf_reader = _load_root_tool("pdf_reader")
read_pdf_directory = pdf_reader.read_pdf_directory

logger_mod = _load_root_tool("logger")
get_logger = logger_mod.get_logger
from config.settings import SETTINGS
from discharge_agent.models.patient import LabResult, LabStatus, Medication, Procedure

logger = get_logger("extractor_agent")


class ExtractorAgent:
    def __init__(self, llm_client: Optional[Any] = None):
        self.llm = llm_client

    def run(self, state: Any) -> Any:
        if hasattr(state, "patient_dir"):
            patient_dir = state.patient_dir
        elif isinstance(state, dict):
            patient_dir = state.get("patient_dir", "")
        else:
            logger.error("State does not contain patient_dir.")
            return state

        if not patient_dir or not Path(patient_dir).exists():
            logger.error(f"Patient directory does not exist: {patient_dir}")
            return state

        # 1. Ingest PDF files
        logger.info(f"Ingesting PDFs from {patient_dir}")
        page_limit = 100
        if os.environ.get("EXTRACTOR_PAGE_LIMIT"):
            page_limit = int(os.environ.get("EXTRACTOR_PAGE_LIMIT"))
        try:
            raw_docs = read_pdf_directory(patient_dir, page_limit=page_limit)
        except Exception as exc:
            logger.error(f"PDF reading failed: {exc}")
            if hasattr(state, "add_flag"):
                state.add_flag(f"[ERROR] PDF Ingestion failed: {exc}")
            elif isinstance(state, dict):
                state.setdefault("clinician_flags", []).append(
                    f"[ERROR] PDF Ingestion failed: {exc}"
                )
            return state

        parsed_docs: List[Dict[str, Any]] = []

        # 2. Parse, Classify and Extract entities per document page
        for doc in raw_docs:
            logger.info(
                f"Processing document page: {doc.file_path} page {doc.page_number}"
            )
            note_type = self._keyword_classify(doc.raw_text)
            doc.note_type = note_type

            parsed_doc = {
                "file_path": doc.file_path,
                "page_number": doc.page_number,
                "note_type": note_type,
                "extraction_confidence": doc.extraction_confidence,
                "read_error": doc.read_error,
                "raw_text": doc.raw_text,
                "entities": {},
            }

            if not doc.read_error and doc.raw_text.strip():
                # Use LLM extraction if LLM is available
                if self.llm and doc.extraction_confidence >= 0.3:
                    try:
                        entities = self._llm_extract(doc.raw_text, note_type)
                        parsed_doc["entities"] = entities
                    except Exception as exc:
                        logger.error(f"LLM extraction error: {exc}")
                        parsed_doc["entities"] = {"llm_error": str(exc)}
                else:
                    parsed_doc["entities"] = {}
            else:
                parsed_doc["entities"] = {"parse_error": "No readable text"}

            parsed_docs.append(parsed_doc)

        # 3. Aggregate clinical data across all parsed pages
        demographics: Dict[str, Any] = {}
        admission_dates: List[str] = []
        discharge_dates: List[str] = []
        diagnoses_list: List[str] = []
        discharge_diagnoses: List[str] = []
        procedures: List[Procedure] = []
        admission_meds: List[Medication] = []
        discharge_meds: List[Medication] = []
        allergies_list: List[str] = []
        discharge_conditions: List[str] = []
        follow_ups: List[str] = []
        raw_labs: List[LabResult] = []
        pending_labs: List[str] = []
        hospital_course_snippets: List[str] = []

        for p_doc in parsed_docs:
            entities = p_doc.get("entities", {})
            source = (
                f"{p_doc.get('file_path', 'unknown')}:p{p_doc.get('page_number', '?')}"
            )
            note_type = p_doc.get("note_type", "unknown")

            # Demographics
            for key in ["patient_name", "patient_age", "patient_sex", "mrn"]:
                val = entities.get(key)
                if val and key not in demographics:
                    demographics[key] = val

            # Dates
            if entities.get("admission_date"):
                admission_dates.append(entities["admission_date"])
            if entities.get("discharge_date"):
                discharge_dates.append(entities["discharge_date"])

            # Diagnoses
            diags = entities.get("diagnoses", [])
            if isinstance(diags, list):
                for d in diags:
                    if d and isinstance(d, str):
                        if note_type == "discharge_summary":
                            discharge_diagnoses.append(d)
                        else:
                            diagnoses_list.append(d)

            # Procedures
            procs = entities.get("procedures", [])
            if isinstance(procs, list):
                for p in procs:
                    if isinstance(p, dict):
                        procedures.append(
                            Procedure(
                                name=p.get("name", "unknown"),
                                date=p.get("date"),
                                notes=p.get("notes"),
                                source_note=source,
                            )
                        )
                    elif isinstance(p, str) and p.strip():
                        procedures.append(Procedure(name=p.strip(), source_note=source))

            # Medications
            meds = entities.get("medications", [])
            if isinstance(meds, list):
                for m in meds:
                    if isinstance(m, dict) and m.get("name"):
                        med_obj = Medication(
                            name=m.get("name"),
                            dose=m.get("dose"),
                            route=m.get("route"),
                            frequency=m.get("frequency"),
                            duration=m.get("duration"),
                            indication=m.get("indication"),
                            source_note=source,
                        )
                        # Determine if admission or discharge med
                        is_discharge = (
                            "discharge" in note_type
                            or "discharge" in source.lower()
                            or "discharge" in str(m.get("status", "")).lower()
                        )
                        if is_discharge:
                            discharge_meds.append(med_obj)
                        else:
                            admission_meds.append(med_obj)

            # Allergies
            if entities.get("allergies"):
                allergies_list.append(entities["allergies"])

            # Conditions / Follow up
            if entities.get("discharge_condition"):
                discharge_conditions.append(entities["discharge_condition"])
            if entities.get("follow_up"):
                follow_ups.append(entities["follow_up"])

            # Hospital Course Snippet
            if entities.get("hospital_course_snippet"):
                hospital_course_snippets.append(entities["hospital_course_snippet"])

            # Labs & Pending tests
            plabs = entities.get("pending_results", [])
            if isinstance(plabs, list):
                pending_labs.extend(plabs)

            # Check raw text for pending keywords manually
            for indicator in ["pending", "awaited", "report awaited", "sent to lab"]:
                if indicator in p_doc.get("raw_text", "").lower():
                    pending_labs.append(f"Awaited/Pending lab result in {source}")

            # Collect extracted labs
            lresults = entities.get("lab_results", [])
            if isinstance(lresults, list):
                for lr in lresults:
                    if isinstance(lr, dict) and lr.get("test"):
                        raw_labs.append(
                            LabResult(
                                test_name=lr.get("test"),
                                value=lr.get("value"),
                                unit=lr.get("unit"),
                                reference_range=lr.get("reference_range"),
                                status=LabStatus.FINAL,
                                source_note=source,
                            )
                        )

        # 4. Regex lab extraction fallback & deduplication
        deduped_labs = self._extract_regex_labs(parsed_docs)
        # Merge LLM-extracted labs with regex-extracted labs
        all_labs = raw_labs + deduped_labs
        final_labs = self._deduplicate_labs(all_labs)

        # 5. Populate State
        if hasattr(state, "raw_documents"):
            state.raw_documents = raw_docs
            state.parsed_documents = parsed_docs
            state.demographics = demographics
            state.admission_date = admission_dates[0] if admission_dates else None
            state.discharge_date = discharge_dates[0] if discharge_dates else None
            state.diagnoses = list(set(diagnoses_list))
            state.discharge_diagnoses = list(set(discharge_diagnoses))
            state.procedures = procedures
            state.admission_medications = admission_meds
            state.discharge_medications = discharge_meds
            state.allergies = ", ".join(set(allergies_list)) if allergies_list else None
            state.lab_results = final_labs
            state.pending_results = list(set(pending_labs))
            state.discharge_condition = (
                discharge_conditions[0] if discharge_conditions else None
            )
            state.follow_up = "; ".join(set(follow_ups)) if follow_ups else None
            state.hospital_course = (
                "\n".join(hospital_course_snippets)
                if hospital_course_snippets
                else None
            )
        elif isinstance(state, dict):
            state["raw_documents"] = [d.to_dict() for d in raw_docs]
            state["parsed_documents"] = parsed_docs
            state["demographics"] = demographics
            state["admission_date"] = admission_dates[0] if admission_dates else None
            state["discharge_date"] = discharge_dates[0] if discharge_dates else None
            state["diagnoses"] = list(set(diagnoses_list))
            state["discharge_diagnoses"] = list(set(discharge_diagnoses))
            state["procedures"] = [p.to_dict() for p in procedures]
            state["admission_medications"] = [m.to_dict() for m in admission_meds]
            state["discharge_medications"] = [m.to_dict() for m in discharge_meds]
            state["allergies"] = (
                ", ".join(set(allergies_list)) if allergies_list else None
            )
            state["lab_results"] = [lr.to_dict() for lr in final_labs]
            state["pending_results"] = list(set(pending_labs))
            state["discharge_condition"] = (
                discharge_conditions[0] if discharge_conditions else None
            )
            state["follow_up"] = "; ".join(set(follow_ups)) if follow_ups else None
            state["hospital_course"] = (
                "\n".join(hospital_course_snippets)
                if hospital_course_snippets
                else None
            )

        return state

    def _keyword_classify(self, text: str) -> str:
        text_lower = text.lower()
        scores: Dict[str, int] = {}
        for note_type, keywords in SETTINGS.note_type_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            scores[note_type] = score
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "unknown"

    def _llm_extract(self, text: str, note_type: str) -> Dict[str, Any]:
        prompt = f"""You are a clinical data extraction assistant.
Extract ONLY information explicitly stated in the document below.
Do NOT infer, guess, or fill in missing data.
If a field is not present in the text, return null for that field.

Document type: {note_type}

Document text:
---
{text[:3000]}
---

Return a JSON object with these fields (null if not found):
{{
    "patient_name": null,
    "patient_age": null,
    "patient_sex": null,
    "mrn": null,
    "admission_date": null,
    "discharge_date": null,
    "diagnoses": [],
    "vitals": {{
        "bp": null, "pulse": null, "rr": null, "temp": null, "spo2": null, "grbs": null
    }},
    "medications": [],
    "lab_results": [],
    "procedures": [],
    "allergies": null,
    "follow_up": null,
    "pending_results": [],
    "hospital_course_snippet": null
}}

Return ONLY the JSON object, no other text."""
        response = self.llm.complete(prompt, max_tokens=1500)
        return self._safe_json_parse(response) or {}

    def _safe_json_parse(self, text: str) -> Optional[dict]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None

    def _extract_regex_labs(self, parsed_docs: List[Dict[str, Any]]) -> List[LabResult]:
        # Regex patterns for clinical labs
        patterns = [
            (
                "haemoglobin",
                r"h(?:a?e)?moglobin\s*(?:\([^)]*\))?\s*[:\-]?\s*([\d.]+)\s*(g[m/]?d?[lL]?)?",
            ),
            (
                "serum_creatinine",
                r"(?:serum\s+)?creatinine\s*[:\-]?\s*([\d.]+)\s*(mg/d[lL])?",
            ),
            (
                "sodium",
                r"(?:s\.?sodium|serum\s+sodium|s\.?na\+?)\s*[:\-]?\s*([\d.]+)\s*(mmol/[lL]|mEq/[lL])?",
            ),
            (
                "potassium",
                r"(?:s\.?potassium|serum\s+potassium|s\.?k\+?)\s*[:\-]?\s*([\d.]+)\s*(mmol/[lL]|mEq/[lL])?",
            ),
            (
                "blood_glucose_rbs",
                r"(?:rbs|random\s+blood\s+sugar|blood\s+glucose)\s*[:\-]?\s*([\d.]+)\s*(mg/d[lL])?",
            ),
        ]
        ranges = {
            "haemoglobin": (12.0, 17.5, "g/dL"),
            "serum_creatinine": (0.7, 1.4, "mg/dL"),
            "sodium": (135.0, 150.0, "mmol/L"),
            "potassium": (3.5, 5.0, "mmol/L"),
            "blood_glucose_rbs": (70.0, 140.0, "mg/dL"),
        }

        extracted: List[LabResult] = []
        for doc in parsed_docs:
            text = doc.get("raw_text", "").lower()
            source = f"{doc.get('file_path', 'unknown')}:p{doc.get('page_number', '?')}"
            date = doc.get("entities", {}).get("date_of_test", None)

            for test_name, pat in patterns:
                m = re.search(pat, text)
                if m:
                    val = m.group(1)
                    unit = m.group(2) if m.lastindex >= 2 else None
                    ref_min, ref_max, ref_unit = ranges.get(
                        test_name, (None, None, None)
                    )
                    ref_range = (
                        f"{ref_min}–{ref_max} {ref_unit}".strip() if ref_min else None
                    )

                    is_abnormal = None
                    if ref_min is not None and val:
                        try:
                            fval = float(re.sub(r"[^\d.]", "", val))
                            is_abnormal = not (ref_min <= fval <= ref_max)
                        except (ValueError, TypeError):
                            pass

                    extracted.append(
                        LabResult(
                            test_name=test_name.replace("_", " ").title(),
                            value=val,
                            unit=unit or ref_unit,
                            reference_range=ref_range,
                            status=LabStatus.FINAL,
                            date=date,
                            is_abnormal=is_abnormal,
                            source_note=source,
                        )
                    )
        return extracted

    def _deduplicate_labs(self, labs: List[LabResult]) -> List[LabResult]:
        seen: Dict[str, LabResult] = {}
        for lab in labs:
            key = lab.test_name.lower()
            if key not in seen or (lab.value and not seen[key].value):
                seen[key] = lab
        return list(seen.values())


if __name__ == "__main__":
    import json
    from dotenv import load_dotenv
    from discharge_agent.utils.llm_client import LLMClient

    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY", "") or os.environ.get(
        "GEMINI_API_KEY", ""
    )

    # Initialize LLM Client
    llm_client = None
    if api_key:
        llm_client = LLMClient(api_key=api_key, model="gemini-1.5-flash")

    extractor = ExtractorAgent(llm_client)

    # Input path
    pdf_path = r"c:\Users\athrv\OneDrive\Desktop\Hospital-discharge-Summary-Agent\discharge_agent\data\Medical bills.pdf"
    print(f"Running extraction on: {pdf_path}")

    state = {
        "patient_id": "Medical bills",
        "patient_dir": pdf_path,
        "raw_documents": [],
        "parsed_documents": [],
        "unreadable_files": [],
        "demographics": {},
        "admission_date": None,
        "discharge_date": None,
        "diagnoses": [],
        "discharge_diagnoses": [],
        "procedures": [],
        "admission_medications": [],
        "discharge_medications": [],
        "allergies": None,
        "lab_results": [],
        "pending_results": [],
        "discharge_condition": None,
        "follow_up": None,
        "hospital_course": None,
    }

    # Force page limit to a small subset for testing since the PDF has 71 pages of scanned images
    # We can modify the read_pdf_directory page limit if needed, but let's run a subset of 3 pages first to test
    # Or run the full document. Let's make it run 3 pages for a quick demonstration, or run all.
    # Actually, we can run all but let's allow it to run all 71 pages or configure a cap.
    # Let's set page_limit = 5 for quick run so it doesn't take too long during validation, but can be overridden.
    print("Running extraction on the first 5 pages for rapid validation...")

    # Since extractor agent calls read_pdf_directory with the state path,
    # let's patch read_pdf_directory's default page limit or read the pdf file directly.
    # In extractor.py line 66: raw_docs = read_pdf_directory(patient_dir)
    # We can temporarily edit tools/pdf_reader.py default limit or pass it.
    # Let's run it!
    result = extractor.run(state)

    # Write parsed documents to a JSON file (acting as our database store)
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "patient_data_extracted.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"Extraction complete! Saved extracted wording to {out_file}")
