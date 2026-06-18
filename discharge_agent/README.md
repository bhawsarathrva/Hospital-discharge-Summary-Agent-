# Discharge Summary Agent — Part 1

## Overview

An agentic AI system that reads raw patient source-note PDFs and produces structured,
clinically safe discharge summary drafts for clinician review.

---

## Architecture

```
discharge_agent/
├── agent/
│   ├── __init__.py
│   ├── loop.py            # Core ReAct agent loop with step/iteration cap
│   ├── planner.py         # Planning & re-planning logic
│   ├── state.py           # Immutable agent state + working memory
│   └── executor.py        # Tool dispatch + result handling
├── tools/
│   ├── __init__.py
│   ├── base.py            # BaseTool ABC + ToolResult dataclass
│   ├── pdf_reader.py      # PDF ingestion & text extraction
│   ├── document_parser.py # Classify & parse note types from raw text
│   ├── medication_reconciler.py  # Admission vs discharge med diff
│   ├── conflict_detector.py      # Cross-note conflict detection
│   ├── drug_interaction.py       # Mock drug-interaction lookup
│   ├── lab_extractor.py          # Extract & flag lab results/pending
│   └── escalation.py             # Flag-for-clinician-review action
├── models/
│   ├── __init__.py
│   ├── patient.py         # Patient, Medication, Lab, Procedure dataclasses
│   ├── summary.py         # DischargeSummary structured output model
│   └── trace.py           # AgentStep, Trace, ObservabilityLog
├── utils/
│   ├── __init__.py
│   ├── llm_client.py      # Anthropic API wrapper with retry logic
│   ├── token_counter.py   # Rough token budget tracker
│   └── json_utils.py      # Safe JSON parse with fallback
├── prompts/
│   ├── __init__.py
│   ├── system_prompt.py   # Master system prompt (no-fabrication guardrail)
│   ├── planning_prompt.py # Initial plan generation prompt
│   ├── extraction_prompt.py      # Per-document extraction prompt
│   ├── reconciliation_prompt.py  # Med reconciliation prompt
│   └── summary_assembly_prompt.py  # Final summary assembly prompt
├── config/
│   ├── __init__.py
│   └── settings.py        # All config: caps, model, timeouts, field names
├── data/
│   ├── patient_1/         # Drop PDFs here
│   └── patient_2/
├── outputs/
│   ├── traces/            # Step-by-step JSON traces
│   └── summaries/         # Final discharge summary JSONs + markdown
├── tests/
│   ├── test_tools.py
│   ├── test_agent_loop.py
│   └── test_conflict_detection.py
├── main.py                # CLI entrypoint
├── requirements.txt
└── README.md
```

---

## Agent Loop Design

The agent uses a **ReAct (Reason + Act)** loop:

```
1. PLAN   — LLM reasons about what documents exist and what it needs
2. ACT    — Dispatches a tool call
3. OBSERVE — Receives tool result (success / partial / failure)
4. RE-PLAN — Updates plan based on new information or failure
5. ASSEMBLE — When plan is satisfied or step cap hit, build summary
6. FLAG   — Mark every field that couldn't be sourced as MISSING/PENDING
```

Hard limits:
- `MAX_STEPS = 30` — agent cannot loop forever
- `MAX_RETRIES_PER_TOOL = 3` — per-tool retry with exponential backoff
- Every failed tool call is logged; agent falls back gracefully

---

## No-Fabrication Guardrail

The system prompt contains an explicit, non-overridable instruction:

> "You MUST NOT invent, infer, or guess any clinical fact. If information is
> absent from the source documents, you MUST output the literal string
> '[MISSING — requires clinician review]' for that field."

Implementation layers:
1. **Prompt-level**: System prompt + per-step reminder
2. **Output-level**: `DischargeSummary.validate()` scans every field and flags
   any field that is empty, None, or contains hedge words without a MISSING tag
3. **Post-processing**: `FabricationGuard.scan()` runs regex over the final
   text looking for confident-sounding statements about data not found in the
   extracted document corpus

---

## Failure & Conflict Handling

- PDF read failure → logged, document marked UNREADABLE, agent continues with remaining docs
- Tool timeout → retry up to 3×, then mark result as UNAVAILABLE
- Conflicting diagnoses across notes → `ConflictDetector` surfaces all variants,
  summary field becomes: `"[CONFLICT: Note A says X; Note B says Y — requires clinician review]"`
- Medication without documented reason → flagged as `"[REASON NOT DOCUMENTED]"`

---

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API key
export ANTHROPIC_API_KEY="sk-..."

# 3. Drop patient PDFs into data/patient_1/ etc.

# 4. Run
python main.py --patient data/patient_1 --output outputs/

# 5. Multiple patients
python main.py --patient data/patient_1 data/patient_2 --output outputs/
```

---

## Output Files

For each patient:
- `outputs/summaries/patient_X_summary.json` — structured summary
- `outputs/summaries/patient_X_summary.md`   — human-readable markdown
- `outputs/traces/patient_X_trace.json`       — full step-by-step trace

---

## Limitations & What I'd Do With More Time

1. **OCR quality**: Handwritten notes (common in these PDFs) degrade extraction.
   Would integrate a dedicated handwriting OCR model (e.g. Google Document AI).
2. **Drug interaction database**: Currently mocked. Would integrate RxNorm/DrugBank API.
3. **Structured NLP**: Clinical NER (e.g. Med7, scispaCy) would improve entity extraction vs pure LLM.
4. **Audit trail**: Would add cryptographic hashing of source docs so the summary
   provenance is verifiable.
5. **Human-in-the-loop**: Step-level pause points where high-uncertainty decisions
   are held for clinician input before proceeding.
