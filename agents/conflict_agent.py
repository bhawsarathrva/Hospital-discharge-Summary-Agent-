from __future__ import annotations
import sys
import re
from typing import Any, Dict, List, Optional
from pathlib import Path

# Ensure workspace root and discharge_agent path are available
workspace_root = str(Path(__file__).resolve().parent.parent)
discharge_agent_dir = str(Path(__file__).resolve().parent.parent / "discharge_agent")
if discharge_agent_dir not in sys.path:
    sys.path.insert(0, discharge_agent_dir)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from discharge_agent.models.patient import Conflict


class ConflictAgent:
    def __init__(self, llm_client: Optional[Any] = None):
        self.llm = llm_client

    def run(self, state: Any) -> Any:
        """
        Run conflict detection on the provided agent state.
        Supports both dataclass AgentState and dictionary-based states.
        """
        # Determine how to read and write to state
        if hasattr(state, "parsed_documents"):
            parsed_docs = state.parsed_documents
        elif isinstance(state, dict):
            parsed_docs = state.get("parsed_documents", [])
        else:
            parsed_docs = []

        if not parsed_docs:
            return state

        # 1. Deterministic Rule-Based Conflict Detection
        rule_conflicts = self._detect_rule_conflicts(parsed_docs)

        # 2. LLM-Based Semantic Conflict Detection (if LLM is available)
        llm_conflicts = []
        if self.llm and len(parsed_docs) >= 2:
            llm_conflicts = self._detect_llm_conflicts(parsed_docs)

        # Merge conflicts (avoid duplicates)
        all_conflicts = list(rule_conflicts)
        existing_signatures = {
            (c.field, c.note_a_value.lower().strip(), c.note_b_value.lower().strip())
            for c in all_conflicts
        }

        for c in llm_conflicts:
            sig = (
                c.field,
                c.note_a_value.lower().strip(),
                c.note_b_value.lower().strip(),
            )
            reverse_sig = (
                c.field,
                c.note_b_value.lower().strip(),
                c.note_a_value.lower().strip(),
            )
            if (
                sig not in existing_signatures
                and reverse_sig not in existing_signatures
            ):
                all_conflicts.append(c)
                existing_signatures.add(sig)

        # Update the state
        if hasattr(state, "conflicts"):
            state.conflicts = all_conflicts
            # Also add to clinician flags if needed
            for conflict in all_conflicts:
                flag = f"[CONFLICT: {conflict.field}] {conflict.description}"
                if hasattr(state, "add_flag"):
                    state.add_flag(flag)
                elif hasattr(state, "clinician_flags"):
                    if flag not in state.clinician_flags:
                        state.clinician_flags.append(flag)
        elif isinstance(state, dict):
            state["conflicts"] = [c.to_dict() for c in all_conflicts]
            flags = state.setdefault("clinician_flags", [])
            for conflict in all_conflicts:
                flag = f"[CONFLICT: {conflict.field}] {conflict.description}"
                if flag not in flags:
                    flags.append(flag)

        return state

    def _detect_rule_conflicts(
        self, parsed_docs: List[Dict[str, Any]]
    ) -> List[Conflict]:
        conflicts: List[Conflict] = []

        # Build field -> [(source, value)] map
        field_values: Dict[str, List[tuple[str, str]]] = {}

        for doc in parsed_docs:
            source = f"{doc.get('file_path', 'unknown')}:p{doc.get('page_number', '?')}"
            entities = doc.get("entities", {})

            # Diagnoses
            for diag in entities.get("diagnoses", []):
                if isinstance(diag, str) and diag.strip():
                    field_values.setdefault("diagnoses", []).append(
                        (source, diag.strip())
                    )

            # Other key fields
            for field_name in [
                "allergies",
                "admission_date",
                "discharge_date",
                "discharge_condition",
            ]:
                val = entities.get(field_name)
                if val and isinstance(val, str) and val.strip():
                    field_values.setdefault(field_name, []).append(
                        (source, val.strip())
                    )

        # Detect conflicts per field
        for field_name, source_vals in field_values.items():
            if len(source_vals) < 2:
                continue

            seen: Dict[str, str] = {}  # normalised_value -> first source
            for source, raw_value in source_vals:
                norm = self._normalise_value(raw_value)
                if norm in seen:
                    continue  # Same value — no conflict

                for prev_norm, prev_source in seen.items():
                    if not self._values_compatible(norm, prev_norm):
                        # Find original prev_value
                        prev_raw = raw_value
                        for s, v in source_vals:
                            if s == prev_source:
                                prev_raw = v
                                break

                        conflicts.append(
                            Conflict(
                                field=field_name,
                                note_a_source=prev_source,
                                note_a_value=prev_raw,
                                note_b_source=source,
                                note_b_value=raw_value,
                                description=(
                                    f"Field '{field_name}' has conflicting values across notes: "
                                    f"'{prev_source}' states '{prev_raw}' vs '{source}' states '{raw_value}'"
                                ),
                            )
                        )
                seen[norm] = source

        return conflicts

    def _normalise_value(self, v: str) -> str:
        return re.sub(r"\s+", " ", v.lower().strip())

    def _values_compatible(self, a: str, b: str) -> bool:
        """True if both values are essentially the same or one contains the other."""
        return a == b or a in b or b in a

    def _detect_llm_conflicts(
        self, parsed_docs: List[Dict[str, Any]]
    ) -> List[Conflict]:
        conflicts: List[Conflict] = []

        # Prepare a text summary of documents for the LLM to inspect
        doc_summaries = []
        for i, doc in enumerate(parsed_docs):
            source = f"{doc.get('file_path', 'unknown')}:p{doc.get('page_number', '?')}"
            entities = doc.get("entities", {})
            doc_summaries.append(
                f"Document {i + 1} [Source: {source}]:\n"
                f"- Diagnoses: {', '.join(entities.get('diagnoses', []))}\n"
                f"- Allergies: {entities.get('allergies', 'Not documented')}\n"
                f"- Admission Date: {entities.get('admission_date', 'Not documented')}\n"
                f"- Discharge Date: {entities.get('discharge_date', 'Not documented')}\n"
                f"- Discharge Condition: {entities.get('discharge_condition', 'Not documented')}\n"
            )

        summaries_text = "\n\n".join(doc_summaries)
        separator = "=" * 60
        prompt = f"""
You are a Clinical Conflict Auditor. Your task is to identify conflicts, discrepancies, or contradictions between different medical notes for the same patient.
For example, if one note says a patient was admitted on 2026-06-01 and another says 2026-06-02, that is a conflict.
Similarly, if one note says "No known drug allergies" and another says "Allergic to Penicillin", that is a conflict.
If one note diagnoses "Ischemic Stroke" and another diagnoses "Hemorrhagic Stroke" (which are mutually exclusive), that is a conflict.

Here is the extracted clinical data from different documents:

{separator}
{summaries_text}
{separator}

Instructions:
- Carefully compare the fields (diagnoses, allergies, dates, conditions) across all documents.
- Identify any clear conflicts or contradictions. Do NOT flag minor variations in phrasing (e.g. "type 2 diabetes" vs "diabetes mellitus type 2" is NOT a conflict).
- Only flag clinical contradictions or data mismatch.
- Return the list of conflicts in a structured JSON format (a list of objects).
- Each conflict object MUST contain:
  - "field": the field name (e.g. "principal_diagnosis", "allergies", "admission_date", "discharge_date")
  - "note_a_source": source name of note A (e.g. "admission_note.pdf:p1")
  - "note_a_value": the value stated in note A
  - "note_b_source": source name of note B (e.g. "progress_note.pdf:p2")
  - "note_b_value": the value stated in note B
  - "description": a clear explanation of why this is a conflict.

If there are no conflicts, return an empty list `[]`.

Response MUST be a valid JSON array and nothing else. Do not add markdown backticks.
"""
        response_text = self.llm.complete(
            prompt, max_tokens=1000, temperature=0.0
        ).strip()

        # Clean JSON markdown fences if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        if not response_text:
            return conflicts

        try:
            import json

            data = json.loads(response_text)
            if isinstance(data, list):
                for item in data:
                    field_name = item.get("field", "unknown")
                    note_a_source = item.get("note_a_source", "unknown")
                    note_a_value = item.get("note_a_value", "unknown")
                    note_b_source = item.get("note_b_source", "unknown")
                    note_b_value = item.get("note_b_value", "unknown")
                    desc = item.get("description", "")
                    conflicts.append(
                        Conflict(
                            field=field_name,
                            note_a_source=note_a_source,
                            note_a_value=note_a_value,
                            note_b_source=note_b_source,
                            note_b_value=note_b_value,
                            description=desc or f"Conflicting {field_name} values.",
                        )
                    )
        except Exception:
            # Fallback gracefully on parsing errors
            pass

        return conflicts
