"""
prompts/reconciliation_prompt.py
"""

RECONCILIATION_PROMPT_TEMPLATE = """
Review the medication reconciliation results below and write a clear clinical summary.

## Admission Medications
{admission_meds}

## Discharge Medications
{discharge_meds}

## Detected Changes
{changes}

## Instructions
- List all changes clearly: ADDED, STOPPED, DOSE CHANGED, CONTINUED
- For any change WITHOUT a documented reason, write: "[REASON NOT DOCUMENTED — requires clinician review]"
- Do NOT infer or guess reasons for medication changes
- Flag any potential safety concerns for clinician review

Return plain text suitable for the Medication Changes section of the discharge summary.
"""
