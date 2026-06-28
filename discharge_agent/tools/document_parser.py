"""
tools/document_parser.py
Classifies each PatientDocument into a note type and runs LLM-based
structured extraction of clinical entities.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from config.settings import SETTINGS
from models.patient import PatientDocument
from tools.base import BaseTool, ToolResult, ToolStatus


class DocumentParserTool(BaseTool):
    name = "document_parser"
    description = (
        "Classify and extract structured clinical data from raw document text. "
        "Returns note_type, key entities (diagnoses, dates, vitals, meds, labs), "
        "and confidence. Does NOT invent data — missing fields are None."
    )

    def __init__(self, llm_client=None):
        self.llm = llm_client  # Injected; can be None for keyword-only classification

    def _run(
        self,
        documents: List[PatientDocument],
        use_llm: bool = True,
    ) -> ToolResult:
        if not documents:
            return ToolResult(
                status=ToolStatus.NOT_FOUND,
                error="No documents provided to parse",
            )

        classified: List[Dict[str, Any]] = []
        failed: List[str] = []

        for doc in documents:
            try:
                result = self._classify_and_extract(doc, use_llm=use_llm)
                classified.append(result)
            except Exception as exc:
                failed.append(f"{doc.file_path}:p{doc.page_number}: {exc}")

        status = (
            ToolStatus.SUCCESS
            if not failed
            else (ToolStatus.PARTIAL if classified else ToolStatus.FAILED)
        )
        return ToolResult(
            status=status,
            data=classified,
            metadata={"classified": len(classified), "failed": len(failed)},
            error=f"Failed pages: {failed}" if failed else None,
        )

    def _classify_and_extract(
        self,
        doc: PatientDocument,
        use_llm: bool = True,
    ) -> Dict[str, Any]:
        """
        First: keyword-based classification (fast, no API call).
        Then: LLM extraction of entities (if llm available and confidence low).
        """
        # Step 1: Keyword classification
        note_type = self._keyword_classify(doc.raw_text)
        doc.note_type = note_type

        base_result = {
            "file_path": doc.file_path,
            "page_number": doc.page_number,
            "note_type": note_type,
            "extraction_confidence": doc.extraction_confidence,
            "read_error": doc.read_error,
            "raw_text": doc.raw_text,
            "entities": {},
        }

        if doc.read_error or not doc.raw_text.strip():
            base_result["entities"] = {"parse_error": "No readable text"}
            return base_result

        # Step 2: LLM extraction (if available)
        if use_llm and self.llm and doc.extraction_confidence >= 0.3:
            entities = self._llm_extract(doc.raw_text, note_type)
            base_result["entities"] = entities

        return base_result

    def _keyword_classify(self, text: str) -> str:
        """Classify note type using keyword matching."""
        text_lower = text.lower()
        scores: Dict[str, int] = {}

        for note_type, keywords in SETTINGS.note_type_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            scores[note_type] = score

        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "unknown"

    def _llm_extract(self, text: str, note_type: str) -> Dict[str, Any]:
        """
        Use LLM to extract structured entities from document text.
        GUARDRAIL: prompt explicitly forbids invention.
        """
        if not self.llm:
            return {}

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

        try:
            response = self.llm.complete(prompt, max_tokens=1500)
            # Safe JSON parse
            return _safe_json_parse(response) or {}
        except Exception as exc:
            return {"llm_extraction_error": str(exc)}

    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "documents": {
                    "type": "array",
                    "description": "List of PatientDocument objects",
                    "required": True,
                },
                "use_llm": {
                    "type": "boolean",
                    "description": "Whether to use LLM for entity extraction",
                    "required": False,
                },
            },
        }


def _safe_json_parse(text: str) -> Optional[dict]:
    """Extract and parse JSON from LLM response text."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find JSON block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None
