discharge_agent/
├── main.py                          # CLI runner (--patient PT-001 / --all)
├── requirements.txt
├── .env.example
│
├── models/
│   └── schemas.py                   # All Pydantic models: AgentState, ClinicalFact, 
│                                    # DischargeSummary, flags, traces, medications
│
├── agents/
│   ├── agent_loop.py                # ★ Core loop: plan→act→guardrail→accumulate
│   ├── guardrail.py                 # No-fabrication check (confidence, source, hallucination scan)
│   ├── llm_client.py                # Anthropic API wrapper with retry
│   └── mock_llm.py                  # Offline demo mode — full pipeline without API key
│
├── tools/
│   ├── pdf_ingestion.py             # pdfplumber + PyMuPDF fallback, document classifier
│   ├── drug_interaction.py          # Mock drug-interaction API with 9 known pairs
│   └── reconciliation.py           # Conflict detection + med reconciliation
│
├── prompts/
│   └── templates.py                 # Planner, Extractor, Summary Generator system prompts
│
├── scripts/
│   ├── generate_synthetic_patients.py  # Creates PT-001 (John Smith) + PT-002 (Margaret Chen)
│   └── output_writer.py             # Saves .json + .txt + trace.json + flags.json
│
├── patients/
│   ├── PT-001/   (4 PDFs — CAP, conflict diagnosis, pending sputum)
│   └── PT-002/   (4 PDFs — heart failure, digoxin+amiodarone interaction, 6 med changes)
│
└── outputs/
    ├── PT-001/discharge_summary.{json,txt}, trace.json, flags.json
    └── PT-002/discharge_summary.{json,txt}, trace.json, flags.json