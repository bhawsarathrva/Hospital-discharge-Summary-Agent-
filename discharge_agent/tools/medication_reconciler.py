"""
tools/medication_reconciler.py
Compares admission vs discharge medication lists and surfaces:
  - Added medications (no admission match)
  - Stopped medications (no discharge match)
  - Dose/route/frequency changes
  - Changes with no documented reason (flagged)
"""

from __future__ import annotations

from typing import List

from config.settings import SETTINGS
from models.patient import Medication, MedicationStatus
from models.summary import MedicationChange
from tools.base import BaseTool, ToolResult, ToolStatus


class MedicationReconcilerTool(BaseTool):
    name = "medication_reconciler"
    description = (
        "Compare admission and discharge medication lists. "
        "Returns a list of MedicationChange objects. "
        "Flags any change that lacks a documented reason."
    )

    def _run(
        self,
        admission_meds: List[Medication],
        discharge_meds: List[Medication],
    ) -> ToolResult:
        if not admission_meds and not discharge_meds:
            return ToolResult(
                status=ToolStatus.NOT_FOUND,
                error="Both admission and discharge medication lists are empty",
            )

        changes: List[MedicationChange] = []
        flags: List[str] = []

        adm_by_name = {_norm(m.name): m for m in admission_meds}
        dis_by_name = {_norm(m.name): m for m in discharge_meds}

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
                    flags.append(f"Medication STOPPED without reason: {adm.name}")

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
                    flags.append(f"Medication ADDED without indication: {dis.name}")

            elif adm and dis:
                # Check for changes
                change_type, detail = _detect_change(adm, dis)
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
                            f"Medication CHANGED ({detail}) without reason: {dis.name}"
                        )
                else:
                    # Continued unchanged
                    mc = MedicationChange(
                        medication=dis,
                        change_type=MedicationStatus.CONTINUED,
                        reason_documented=True,
                    )
                    changes.append(mc)

        return ToolResult(
            status=ToolStatus.SUCCESS,
            data=changes,
            metadata={
                "total_changes": len(changes),
                "flags_raised": len(flags),
                "flags": flags,
            },
        )

    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "admission_meds": {
                    "type": "array",
                    "description": "Medication list at admission",
                    "required": True,
                },
                "discharge_meds": {
                    "type": "array",
                    "description": "Medication list at discharge",
                    "required": True,
                },
            },
        }


def _norm(name: str) -> str:
    """Normalize medication name for comparison."""
    return name.lower().strip().replace(".", "").replace("-", " ")


def _detect_change(adm: Medication, dis: Medication):
    """Detect if dose, route, or frequency changed."""
    if adm.dose and dis.dose and _norm(adm.dose) != _norm(dis.dose):
        return MedicationStatus.DOSE_CHANGED, f"dose: {adm.dose} → {dis.dose}"
    if adm.route and dis.route and _norm(adm.route) != _norm(dis.route):
        return MedicationStatus.ROUTE_CHANGED, f"route: {adm.route} → {dis.route}"
    if adm.frequency and dis.frequency and _norm(adm.frequency) != _norm(dis.frequency):
        return (
            MedicationStatus.UNKNOWN_CHANGE,
            f"frequency: {adm.frequency} → {dis.frequency}",
        )
    return None, None
