from __future__ import annotations

from typing import Any, List, Optional
from config.settings import SETTINGS
from discharge_agent.models.patient import Medication, MedicationStatus
from discharge_agent.models.summary import MedicationChange

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

logger = get_logger("medication_agent")


class MedicationAgent:
    def __init__(self, llm_client: Optional[Any] = None):
        self.llm = llm_client

    def run(self, state: Any) -> Any:
        raw_admission_meds = []
        raw_discharge_meds = []

        if hasattr(state, "admission_medications"):
            raw_admission_meds = state.admission_medications
            raw_discharge_meds = state.discharge_medications
        elif isinstance(state, dict):
            raw_admission_meds = state.get("admission_medications", [])
            raw_discharge_meds = state.get("discharge_medications", [])

        # Convert dictionaries to Medication dataclass if needed
        admission_meds = self._convert_to_med_objects(raw_admission_meds)
        discharge_meds = self._convert_to_med_objects(raw_discharge_meds)

        if not admission_meds and not discharge_meds:
            logger.info("No medications to reconcile.")
            return state

        changes: List[MedicationChange] = []
        flags: List[str] = []

        # Build maps for reconciliation
        adm_by_name = {self._norm(m.name): m for m in admission_meds}
        dis_by_name = {self._norm(m.name): m for m in discharge_meds}

        all_names = set(adm_by_name) | set(dis_by_name)

        for name in all_names:
            adm = adm_by_name.get(name)
            dis = dis_by_name.get(name)

            if adm and not dis:
                # Stopped at discharge
                mc = MedicationChange(
                    medication=adm,
                    change_type=MedicationStatus.STOPPED,
                    reason_documented=bool(adm.change_reason),
                    flag_message=(
                        f"{SETTINGS.flag_prefix} '{adm.name}' was stopped "
                        f"with no documented reason — requires clinician review]"
                        if not adm.change_reason
                        else None
                    ),
                )
                changes.append(mc)
                if not adm.change_reason:
                    flags.append(
                        f"Medication '{adm.name}' stopped with no documented reason."
                    )

            elif not adm and dis:
                # Newly added at discharge
                mc = MedicationChange(
                    medication=dis,
                    change_type=MedicationStatus.ADDED,
                    reason_documented=bool(dis.change_reason),
                    flag_message=(
                        f"{SETTINGS.flag_prefix} '{dis.name}' was added "
                        f"with no documented indication — requires clinician review]"
                        if not dis.change_reason
                        else None
                    ),
                )
                changes.append(mc)
                if not dis.change_reason:
                    flags.append(
                        f"Medication '{dis.name}' added with no documented reason/indication."
                    )

            elif adm and dis:
                # Check for changes
                change_type, detail = self._detect_change(adm, dis)
                if change_type:
                    mc = MedicationChange(
                        medication=dis,
                        change_type=change_type,
                        reason_documented=bool(dis.change_reason),
                        flag_message=(
                            f"{SETTINGS.flag_prefix} '{dis.name}' changed ({detail}) "
                            f"with no documented reason — requires clinician review]"
                            if not dis.change_reason
                            else None
                        ),
                    )
                    changes.append(mc)
                    if not dis.change_reason:
                        flags.append(
                            f"Medication '{dis.name}' changed ({detail}) with no documented reason."
                        )
                else:
                    # Continued unchanged
                    mc = MedicationChange(
                        medication=dis,
                        change_type=MedicationStatus.CONTINUED,
                        reason_documented=True,
                    )
                    changes.append(mc)

        # Write updates to state
        if hasattr(state, "medication_changes"):
            state.medication_changes = changes
            for f in flags:
                state.add_flag(f)
        elif isinstance(state, dict):
            state["medication_changes"] = [c.to_dict() for c in changes]
            existing_flags = state.setdefault("clinician_flags", [])
            for f in flags:
                if f not in existing_flags:
                    existing_flags.append(f)

        return state

    def _convert_to_med_objects(self, meds: List[Any]) -> List[Medication]:
        med_objs = []
        for m in meds:
            if isinstance(m, Medication):
                med_objs.append(m)
            elif isinstance(m, dict):
                med_objs.append(
                    Medication(
                        name=m.get("name", ""),
                        dose=m.get("dose"),
                        route=m.get("route"),
                        frequency=m.get("frequency"),
                        duration=m.get("duration"),
                        indication=m.get("indication"),
                        change_reason=m.get("change_reason"),
                        source_note=m.get("source_note"),
                    )
                )
        return med_objs

    def _norm(self, name: str) -> str:
        return name.lower().strip().replace(".", "").replace("-", " ")

    def _detect_change(
        self, adm: Medication, dis: Medication
    ) -> tuple[Optional[MedicationStatus], Optional[str]]:
        if adm.dose and dis.dose and self._norm(adm.dose) != self._norm(dis.dose):
            return MedicationStatus.DOSE_CHANGED, f"dose: {adm.dose} → {dis.dose}"
        if adm.route and dis.route and self._norm(adm.route) != self._norm(dis.route):
            return MedicationStatus.ROUTE_CHANGED, f"route: {adm.route} → {dis.route}"
        if (
            adm.frequency
            and dis.frequency
            and self._norm(adm.frequency) != self._norm(dis.frequency)
        ):
            return (
                MedicationStatus.UNKNOWN_CHANGE,
                f"frequency: {adm.frequency} → {dis.frequency}",
            )
        return None, None
