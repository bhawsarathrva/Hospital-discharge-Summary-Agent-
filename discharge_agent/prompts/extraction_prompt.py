"""
prompts/extraction_prompt.py
"""

EXTRACTION_PROMPT_TEMPLATE = """
Extract clinical data from the following document text.

## Document Information
- File: {file_path}
- Page: {page_number}
- Note Type: {note_type}
- OCR Confidence: {confidence}

## Document Text
---
{text}
---

## Instructions
Extract ONLY what is explicitly written in the text above.
Do NOT infer, guess, or add information not present.
For any field not found, return null.

Return JSON with this structure:
{{
    "patient_name": null,
    "patient_age": null,
    "patient_sex": null,
    "mrn": null,
    "admission_date": null,
    "discharge_date": null,
    "diagnoses": [],
    "chief_complaint": null,
    "past_history": null,
    "vitals": {{
        "bp": null,
        "pulse": null,
        "rr": null,
        "temp": null,
        "spo2": null,
        "grbs": null,
        "weight": null
    }},
    "medications": [
        {{
            "name": null,
            "dose": null,
            "route": null,
            "frequency": null,
            "duration": null
        }}
    ],
    "lab_results": [
        {{
            "test": null,
            "value": null,
            "unit": null,
            "reference_range": null,
            "status": "final|pending|unknown"
        }}
    ],
    "procedures": [],
    "allergies": null,
    "follow_up": null,
    "pending_results": [],
    "hospital_course_snippet": null,
    "discharge_condition": null,
    "discharge_destination": null
}}
"""

HOSPITAL_COURSE_PROMPT_TEMPLATE = """
Synthesise the hospital course narrative from the following clinical notes.

## Source Notes (in chronological order)
{notes_text}

## Instructions
- Summarise what happened during the admission in plain, factual language.
- Include: presenting complaint, key investigations, treatment given, clinical progress, reason for discharge.
- Do NOT add any information not present in the notes.
- If there are gaps in the timeline, state "Documentation gap — details unavailable for this period."
- Do NOT make clinical recommendations or interpretations.
- Keep it concise: 150-300 words.
- If there is insufficient information, state "Insufficient documentation to reconstruct hospital course."

Return plain text (not JSON).
"""
