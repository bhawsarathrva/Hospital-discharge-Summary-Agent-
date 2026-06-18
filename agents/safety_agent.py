"""
agents/safety_agent.py
SafetyAgent checks medications for dangerous interactions
and flags any unresolved conflicts or critical pending results.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

def _load_root_tool(name: str):
    root = Path(__file__).parent.parent
    spec = importlib.util.spec_from_file_location(f"root_tools_{name}", str(root / "tools" / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

logger_mod = _load_root_tool("logger")
get_logger = logger_mod.get_logger

drug_checker = _load_root_tool("drug_checker")
check_drug_interactions = drug_checker.check_drug_interactions

escalation_tool = _load_root_tool("escalation_tool")
format_clinician_flag = escalation_tool.format_clinician_flag

logger = get_logger("safety_agent")

class SafetyAgent:
    """
    SafetyAgent runs clinical safety checks.
    It runs medication-medication interaction analysis, flags high-severity interactions,
    and aggregates clinical safety warnings for the final summary draft.
    """
    def __init__(self, llm_client: Optional[Any] = None):
        self.llm = llm_client

    def run(self, state: Any) -> Any:
        """
        Runs safety checks on the provided agent state.
        Supports both AgentState objects and dictionary-based states.
        """
        # 1. Retrieve data from state
        discharge_meds = []
        pending_results = []
        conflicts = []

        if hasattr(state, "discharge_medications"):
            discharge_meds = state.discharge_medications
            pending_results = state.pending_results
            conflicts = state.conflicts
        elif isinstance(state, dict):
            discharge_meds = state.get("discharge_medications", [])
            pending_results = state.get("pending_results", [])
            conflicts = state.get("conflicts", [])

        # Extract medication names
        med_names = []
        for m in discharge_meds:
            if isinstance(m, dict):
                name = m.get("name")
            else:
                name = getattr(m, "name", None)
            if name:
                med_names.append(name)

        # 2. Check drug interactions
        interaction_flags = []
        clinician_flags = []

        if med_names:
            logger.info(f"Checking drug-drug interactions for: {med_names}")
            interactions = check_drug_interactions(med_names)
            for item in interactions:
                severity = item.get("severity", "LOW")
                desc = item.get("description", "")
                
                # Format safety warning
                formatted_flag = format_clinician_flag(
                    severity=severity,
                    field="medications",
                    message=desc,
                )
                
                if severity == "HIGH":
                    clinician_flags.append(formatted_flag)
                interaction_flags.append(formatted_flag)

        # 3. Check for clinical conflicts
        for conflict in conflicts:
            if isinstance(conflict, dict):
                field = conflict.get("field", "unknown")
                desc = conflict.get("description", "Discrepancy detected.")
            else:
                field = getattr(conflict, "field", "unknown")
                desc = getattr(conflict, "description", "Discrepancy detected.")

            formatted_flag = format_clinician_flag(
                severity="HIGH",
                field=field,
                message=desc,
            )
            clinician_flags.append(formatted_flag)

        # 4. Check for pending lab results
        for pending in pending_results:
            formatted_flag = format_clinician_flag(
                severity="INFORMATIONAL",
                field="pending_labs",
                message=f"Pending result: {pending}",
            )
            clinician_flags.append(formatted_flag)

        # Write updates to state
        if hasattr(state, "clinician_flags"):
            for flag in clinician_flags:
                state.add_flag(flag)
            state.drug_interaction_flags = list(set(interaction_flags))
        elif isinstance(state, dict):
            existing_flags = state.setdefault("clinician_flags", [])
            for flag in clinician_flags:
                if flag not in existing_flags:
                    existing_flags.append(flag)
            state["drug_interaction_flags"] = list(set(interaction_flags))

        return state
