# Discharge Summary Agent — Part 1

> **Agentic AI system that reads raw patient source notes (PDFs) and produces a structured, clinically safe discharge summary draft for clinician review.**

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [Agent Loop Design](#agent-loop-design)
5. [No-Fabrication Guardrail](#no-fabrication-guardrail)
6. [Failure & Conflict Handling](#failure--conflict-handling)
7. [Medication Reconciliation](#medication-reconciliation)
8. [Tool Definitions](#tool-definitions)
9. [Observability & Step Traces](#observability--step-traces)
10. [n8n Workflow](#n8n-workflow)
11. [Output Format](#output-format)
12. [Limitations & Future Work](#limitations--future-work)

---

## Overview

This system is built as an **n8n workflow** implementing a true agentic loop. Given a patient's folder of source-note PDFs, the agent:

- Plans which information it still needs at every step
- Calls tools to extract, check, or escalate — deciding which tool to use based on current state
- Enforces a hard **no-fabrication** guardrail: any fact it cannot source is marked `⚠ MISSING` and flagged for clinician review
- Detects conflicts between notes and surfaces them rather than silently resolving them
- Reconciles admission vs discharge medications and flags unexplained changes
- Emits a full step-by-step trace for every run
- Enforces a hard iteration cap (default: 20 steps) so the agent can never run forever

The output is always a **DRAFT for clinician review** — never an auto-finalized clinical document.

---

## Project Structure

```
discharge-summary-agent/
├── n8n/
│   └── workflow.json              # Importable n8n workflow export
├── scripts/
│   ├── pdf_extract.py             # PDF text extraction (pdfplumber / PyMuPDF)
│   ├── fact_extractor.py          # Regex + LLM-assisted field parser
│   ├── conflict_detector.py       # Cross-note field diffing
│   ├── med_reconcile.py           # Admission vs discharge medication diff
│   └── drug_interaction_mock.py   # Mock drug-interaction API tool
├── prompts/
│   ├── planner_system.txt         # System prompt for the Planner LLM
│   └── summary_generator.txt      # System prompt for the Summary Generator LLM
├── patients/
│   └── <patient_id>/              # One folder per patient, containing source PDFs
├── outputs/
│   └── <patient_id>/
│       ├── discharge_summary.json # Structured summary draft
│       └── trace.json             # Full step-by-step agent trace
├── .env.example                   # API key template (never commit .env)
├── requirements.txt
└── README.md
```

---

## Quick Start

### Prerequisites

- [n8n](https://n8n.io/) v1.30+ (self-hosted or cloud)
- Python 3.10+ (for PDF extraction scripts)
- An Anthropic API key (Claude model access)

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` includes: `pdfplumber`, `pymupdf`, `anthropic`, `python-dotenv`

### 2. Configure environment

```bash
cp .env.example .env
# Fill in:
# ANTHROPIC_API_KEY=sk-ant-...
# MAX_AGENT_STEPS=20
# FABRICATION_CONFIDENCE_THRESHOLD=0.70
```

> **Important:** Never commit `.env` or any file containing API keys.

### 3. Import the n8n workflow

1. Open your n8n instance
2. Go to **Workflows → Import from file**
3. Select `n8n/workflow.json`
4. Set the `ANTHROPIC_API_KEY` credential in the workflow's HTTP Request nodes

### 4. Add patient PDFs

Place source PDFs for each patient under `patients/<patient_id>/`. The system expects any combination of:

- `admission_note.pdf`
- `progress_note_*.pdf`
- `lab_results.pdf`
- `medications_admission.pdf`
- `medications_discharge.pdf`

File names are flexible — the agent reads all PDFs in the folder and classifies their content automatically.

### 5. Trigger a run

Send a POST request to the workflow's webhook endpoint:

```bash
curl -X POST http://localhost:5678/webhook/discharge-agent \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "PT-001",
    "pdf_paths": ["patients/PT-001/admission_note.pdf", "patients/PT-001/lab_results.pdf"]
  }'
```

Outputs are written to `outputs/<patient_id>/`.

---

## Agent Loop Design

The agent runs an iterative **plan → act → check → accumulate** loop, bounded by a hard step cap.

```
INIT state: { step=0, facts={}, flags=[], trace=[], pending_sections=ALL }

LOOP (while step < MAX_STEPS and pending_sections not empty):
  1. INCREMENT step
  2. PLANNER LLM  →  reasons over current facts{} and decides next action
  3. EMIT trace entry  (reasoning · action · inputs · result)
  4. ROUTE to tool:
       extract_fact   →  Fact Extractor (PDF parse + LLM)
       drug_check     →  Drug-Interaction Tool (mock API)
       escalate       →  Flag node (append to flags[])
       done           →  EXIT loop early
  5. ERROR HANDLER  →  retry ×2, then fallback to MISSING
  6. NO-FABRICATION CHECK  →  discard if not source-traceable
  7. ACCUMULATE into facts{} (or mark MISSING)
  8. CHECK completeness → loop or exit

POST-LOOP:
  Conflict Detection → Medication Reconciliation → Summary Generator → Output
```

### Why a real loop and not a fixed pipeline?

A fixed pipeline cannot adapt to missing documents, partial notes, or unexpected content. The planner re-evaluates at every step, so if a lab result is absent it will flag it as pending rather than attempting to fill it in, and if a new conflict emerges mid-extraction it can escalate immediately.

### Hard step cap

The IF node at the top of every loop iteration checks `step < MAX_STEPS` (default: 20). If the cap is hit, the agent emits a `cap_hit` flag and proceeds directly to output with whatever facts it has collected so far — it never loops indefinitely.

---

## No-Fabrication Guardrail

This is the most critical safety property of the system.

### How it works

Before any extracted value is written to `facts{}`, it passes a **No-Fabrication Check**:

1. **Source traceability** — the value must be accompanied by `{ source_doc, page, snippet }`. Values without a source reference are rejected.
2. **Confidence threshold** — LLM-extracted values must carry a confidence score ≥ 0.70 (configurable). Below threshold → `MISSING`.
3. **Hallucination marker scan** — values containing vague language (`"likely"`, `"probably"`, `"may be"`, `"appears to"`) are rejected as unsourced inferences.
4. **Pending vs absent distinction** — if a document explicitly states a result is pending, the field is marked `⏳ PENDING – awaiting result` rather than `MISSING`. These are meaningfully different for the clinician.

### What happens on failure

```json
{
  "field": "discharge_diagnosis",
  "value": "⚠ MISSING – not found in source documents",
  "flag": {
    "type": "MISSING",
    "severity": "HIGH",
    "message": "Discharge diagnosis could not be sourced. Clinician must complete before finalisation."
  }
}
```

The Summary Generator LLM is instructed to pass these markers through verbatim — it never fills in or softens a `MISSING` field.

---

## Failure & Conflict Handling

### Tool and document failures

Every tool call is wrapped in an error handler that:

1. Retries up to 2 times with a 1-second backoff
2. On continued failure, logs `{ error, tool, inputs }` to the trace
3. Marks the affected field as `MISSING` with reason `tool_error` or `document_unreadable`
4. Continues the loop — never crashes the run

### Conflicting information

After the main loop, the **Conflict Detector** diffs facts extracted from different source documents. If two notes disagree on the same field, the output records both values and flags the conflict:

```json
{
  "field": "principal_diagnosis",
  "conflict": true,
  "values": [
    { "value": "Community-acquired pneumonia", "source": "admission_note.pdf", "page": 1 },
    { "value": "Aspiration pneumonia", "source": "progress_note_day3.pdf", "page": 2 }
  ],
  "flag": {
    "type": "CONFLICT",
    "severity": "HIGH",
    "message": "Conflicting diagnoses found across notes. Clinician must resolve before finalisation."
  }
}
```

The agent never arbitrarily picks one value — both are surfaced.

---

## Medication Reconciliation

The **Medication Reconciliation** node diffs `admission_meds[]` against `discharge_meds[]` and classifies every change:

| Change type | Behaviour |
|---|---|
| New medication added | Logged with dose/frequency; flagged if no documented reason |
| Medication stopped | Logged; flagged if no documented reason |
| Dose / frequency changed | Logged with before/after values; flagged if no documented reason |
| Unchanged | Logged without flag |

Any change without a documented reason in the notes generates a `RECONCILIATION_NEEDED` flag rather than being silently accepted. This is surfaced in the `discharge_medications` section of the output.

---

## Tool Definitions

### `extract_fact`

Parses a specific clinical field from a specific PDF chunk using regex patterns and a focused LLM call. Returns `{ field, value, source_doc, page, snippet, confidence }`.

### `drug_check` (mock)

Accepts `{ medications: string[] }` and returns `{ interactions: [...] }`. In the mock implementation, a small hardcoded lookup table simulates interactions. Replace with a live API (e.g. OpenFDA, DrugBank) by swapping the HTTP Request node URL. On any interaction found, the agent automatically escalates to the clinician flag list.

### `escalate`

Appends a structured flag to `state.flags[]`. Always called by the planner — never automatically triggered without a reasoning step. Flags include `type`, `severity`, `field`, and a human-readable `message`.

---

## Observability & Step Traces

Every run produces a `trace.json` alongside the discharge summary. Each entry records:

```json
{
  "step": 4,
  "reasoning": "Lab results PDF parsed. Sodium value found on page 2. Potassium result marked as pending in the document. Moving to extract discharge medications next.",
  "action": "extract_fact",
  "inputs": { "field": "potassium", "source": "lab_results.pdf" },
  "result": { "value": "⏳ PENDING – result not yet available", "confidence": 1.0 },
  "next_decision": "Mark potassium as pending and proceed to medications"
}
```

This trace is returned in the API response and written to disk. It is the primary debugging and audit tool — reviewers can follow exactly why the agent made every decision.

---

## n8n Workflow

The workflow (`n8n/workflow.json`) is structured as follows:

| Node | Type | Purpose |
|---|---|---|
| Webhook | Trigger | Receives patient_id and pdf_paths |
| PDF Ingestion | Execute Command | Runs pdf_extract.py, returns text per file |
| Initialise State | Set | Creates state object |
| Step cap check | IF | Guards against infinite loops |
| Planner LLM | HTTP Request | Calls Claude with current state, returns action |
| Emit trace | Code | Appends to trace[] |
| Action router | Switch | Routes to fact extractor / drug check / escalate |
| Tool error handler | Code | Retry + fallback logic |
| No-fabrication check | IF | Validates source traceability and confidence |
| Accumulate facts | Set | Merges new fact into state.facts{} |
| Completeness check | IF | Exits loop or re-plans |
| Conflict detection | Code | Cross-note diffing |
| Med reconciliation | Code | Admission vs discharge diff |
| Summary generator | HTTP Request | Calls Claude to render final draft |
| Respond | Respond to Webhook | Returns summary + trace + flags |

---

## Output Format

```json
{
  "status": "DRAFT – for clinician review only. Do not use as a finalised clinical document.",
  "generated_at": "2025-06-03T10:42:00Z",
  "patient": {
    "name": "...",
    "dob": "...",
    "mrn": "..."
  },
  "admission_date": "...",
  "discharge_date": "...",
  "principal_diagnosis": "...",
  "secondary_diagnoses": ["..."],
  "hospital_course": "...",
  "procedures": ["..."],
  "discharge_medications": [
    { "name": "...", "dose": "...", "change": "NEW – no documented reason ⚠" }
  ],
  "allergies": ["..."],
  "follow_up": "...",
  "pending_results": ["⏳ Potassium – result not yet available"],
  "discharge_condition": "...",
  "flags": [
    {
      "type": "CONFLICT",
      "severity": "HIGH",
      "field": "principal_diagnosis",
      "message": "Conflicting diagnoses across notes. Clinician must resolve."
    },
    {
      "type": "RECONCILIATION_NEEDED",
      "severity": "MEDIUM",
      "field": "metformin",
      "message": "Metformin stopped at discharge with no documented reason."
    }
  ],
  "agent_trace": [ ... ]
}
```

---

## Limitations & Future Work

### Current limitations

- **PDF quality dependence** — scanned or image-only PDFs are not supported in this version. Extraction relies on selectable text. Adding OCR (e.g. Tesseract via a pre-processing step) would address this.
- **Single-language support** — the system assumes English-language notes. Multi-language support would require language detection and translation preprocessing.
- **Mock drug-interaction tool** — the `drug_check` tool uses a small hardcoded lookup. A production deployment must integrate a validated, maintained drug database (OpenFDA, DrugBank, or equivalent).
- **No authentication or audit trail** — the n8n webhook has no auth in the current setup. Production deployment requires auth middleware and a persistent, tamper-evident audit log of every run.
- **No human-in-the-loop pause** — the agent runs to completion without waiting for clinician input mid-run. A production system might benefit from a review checkpoint before the final summary is generated.
- **Confidence calibration** — the 0.70 confidence threshold is a starting heuristic. It has not been validated against real clinical notes and will need tuning.

### What would be done with more time

- Replace the mock drug-interaction tool with a live, validated API integration
- Add OCR preprocessing for scanned documents
- Build a structured evaluation harness: given gold-standard summaries, measure field-level extraction accuracy
- Implement Part 2 (learning from doctor edits) using a contextual bandit over prompt strategies, rewarded by reduced normalised edit distance on a held-out patient set
- Add a human-in-the-loop pause node in n8n where a clinician can correct extracted facts before the final summary is generated, feeding corrections back into `facts{}`
- Integrate with an EHR system (FHIR R4) for direct structured data input rather than relying solely on free-text PDFs

---

> This system is a research and demonstration prototype. It is not a certified medical device and must not be used for real clinical decision-making without appropriate validation, regulatory review, and clinician oversight.
