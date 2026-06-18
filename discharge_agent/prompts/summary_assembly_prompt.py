"""
prompts/summary_assembly_prompt.py
"""

SUMMARY_ASSEMBLY_PROMPT_TEMPLATE = """
You are assembling a final discharge summary draft from extracted data.

## Extracted Data
{extracted_data}

## Conflicts
{conflicts}

## Pending Items
{pending_items}

## Clinician Flags
{flags}

## Instructions
Assemble the discharge summary with these rules:
1. Use ONLY the extracted data above — do NOT add any new information
2. For missing fields, use exactly: "[MISSING — requires clinician review]"
3. For conflicting data, use: "[CONFLICT: <description> — requires clinician review]"
4. For pending labs: "[PENDING — result not yet available]"
5. The summary is ALWAYS a draft — include a clear header stating this

Return a JSON object matching the DischargeSummary structure with all required sections.
Every field must be populated from the extracted data or marked as MISSING/PENDING/CONFLICT.
"""
