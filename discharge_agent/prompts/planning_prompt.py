"""
prompts/planning_prompt.py
"""

PLANNING_PROMPT_TEMPLATE = """
You are building a discharge summary for a patient.

## Available Documents
The following source documents have been loaded:
{document_list}

## Task
Create a step-by-step plan to extract all required information for the discharge summary.

Required sections: {required_sections}

For each section, identify:
1. Which document(s) likely contain the information
2. What tool to call
3. What to do if the information is missing

## Plan Format
Return a JSON object:
{{
  "steps": [
    {{
      "step_id": 1,
      "goal": "Extract patient demographics",
      "tool": "document_parser",
      "target_documents": ["admission_note"],
      "fallback": "Mark as MISSING if not found in any note"
    }}
  ],
  "identified_risks": [
    "Handwritten notes may have low OCR confidence"
  ]
}}

Do NOT invent any data. Only plan to extract what the documents might contain.
"""
