"""
agents/planner.py
Planner coordinates and executes the multi-agent pipeline:
ExtractorAgent -> ConflictAgent -> MedicationAgent -> SafetyAgent -> SummaryAgent
"""

from __future__ import annotations
from typing import Any, Optional

import importlib.util
from pathlib import Path


def _load_root_tool(name: str):
    root = Path(__file__).parent.parent
    spec = importlib.util.spec_from_file_location(
        f"root_tools_{name}", str(root / "tools" / f"{name}.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


logger_mod = _load_root_tool("logger")
get_logger = logger_mod.get_logger

logger = get_logger("planner_agent")
from agents.extractor import ExtractorAgent
from agents.conflict_agent import ConflictAgent
from agents.medication_agent import MedicationAgent
from agents.safety_agent import SafetyAgent
from agents.summary_agent import SummaryAgent


class Planner:
    """
    Planner orchestrates the multi-agent workflow.
    It sequentially runs the clinical extraction, conflict audit, medication reconciliation,
    clinical safety validation, and summary assembly steps.
    """

    def __init__(self, llm_client: Optional[Any] = None):
        self.llm = llm_client
        self.extractor = ExtractorAgent(llm_client)
        self.conflict_detector = ConflictAgent(llm_client)
        self.med_reconciler = MedicationAgent(llm_client)
        self.safety_auditor = SafetyAgent(llm_client)
        self.summary_assembler = SummaryAgent(llm_client)

    def run(self, state: Any) -> Any:
        """
        Execute the full multi-agent pipeline sequentially on the state.
        """
        logger.info("Starting Multi-Agent Discharge Summary Pipeline")

        # 1. Extraction Phase
        logger.info("=== Phase 1: Clinical Entity Extraction ===")
        state = self.extractor.run(state)

        # 2. Conflict Detection Phase
        logger.info("=== Phase 2: Cross-Note Conflict Detection ===")
        state = self.conflict_detector.run(state)

        # 3. Medication Reconciliation Phase
        logger.info("=== Phase 3: Medication Reconciliation ===")
        state = self.med_reconciler.run(state)

        # 4. Safety Audit & Escalation Phase
        logger.info("=== Phase 4: Safety Audit & Clinical Escalations ===")
        state = self.safety_auditor.run(state)

        # 5. Summary Assembly & Guardrails Phase
        logger.info("=== Phase 5: Discharge Summary Assembly ===")
        state = self.summary_assembler.run(state)

        logger.info("Multi-Agent Pipeline Complete!")
        return state
