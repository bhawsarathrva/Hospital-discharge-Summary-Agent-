"""
tools/drug_interaction.py
Mock drug-interaction lookup tool.
In production this would call RxNorm / DrugBank / OpenFDA APIs.
The mock always returns a deterministic result based on known interaction pairs,
so the agent can test its escalation logic end-to-end.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from tools.base import BaseTool, ToolResult, ToolStatus


# Known interaction pairs (normalised lowercase drug names)
# Format: (drug_a, drug_b) → (severity, description)
KNOWN_INTERACTIONS: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("warfarin", "aspirin"): (
        "HIGH",
        "Warfarin + Aspirin: Increased bleeding risk. Monitor INR closely.",
    ),
    ("metformin", "contrast"): (
        "HIGH",
        "Metformin + IV Contrast: Risk of lactic acidosis. Hold metformin 48h before/after contrast.",
    ),
    ("meropenem", "valproate"): (
        "HIGH",
        "Meropenem + Valproate: Meropenem significantly reduces valproate levels. "
        "Risk of seizure breakthrough.",
    ),
    ("gentamicin", "furosemide"): (
        "MODERATE",
        "Gentamicin + Furosemide: Increased ototoxicity and nephrotoxicity risk.",
    ),
    ("lantus", "metformin"): (
        "LOW",
        "Insulin Glargine + Metformin: Monitor blood glucose — additive hypoglycaemic effect.",
    ),
    ("ssri", "tramadol"): (
        "HIGH",
        "SSRI + Tramadol: Risk of serotonin syndrome.",
    ),
    ("ciprofloxacin", "antacid"): (
        "MODERATE",
        "Ciprofloxacin + Antacids: Antacids reduce ciprofloxacin absorption. "
        "Separate administration by 2 hours.",
    ),
}

# Severity ranking
SEVERITY_RANK = {"HIGH": 3, "MODERATE": 2, "LOW": 1}


class DrugInteractionTool(BaseTool):
    name = "drug_interaction_lookup"
    description = (
        "Check a list of medications for known drug-drug interactions. "
        "Returns flagged interactions with severity levels. "
        "HIGH severity interactions must be escalated. "
        "This is a mock — in production, would call RxNorm/DrugBank API."
    )

    def _run(
        self,
        medication_names: List[str],
    ) -> ToolResult:
        if not medication_names:
            return ToolResult(
                status=ToolStatus.NOT_FOUND,
                error="No medication names provided",
            )

        normalised = [_normalise_drug_name(m) for m in medication_names]
        interactions = []

        # Check all pairs
        for i in range(len(normalised)):
            for j in range(i + 1, len(normalised)):
                a, b = normalised[i], normalised[j]
                # Check both orderings
                result = (
                    KNOWN_INTERACTIONS.get((a, b))
                    or KNOWN_INTERACTIONS.get((b, a))
                    or _fuzzy_match(a, b)
                )
                if result:
                    severity, description = result
                    interactions.append(
                        {
                            "drug_a": medication_names[i],
                            "drug_b": medication_names[j],
                            "severity": severity,
                            "description": description,
                            "requires_escalation": severity == "HIGH",
                        }
                    )

        # Sort by severity
        interactions.sort(
            key=lambda x: SEVERITY_RANK.get(x["severity"], 0), reverse=True
        )

        high_count = sum(1 for i in interactions if i["severity"] == "HIGH")

        return ToolResult(
            status=ToolStatus.SUCCESS,
            data=interactions,
            metadata={
                "checked_medications": len(medication_names),
                "interactions_found": len(interactions),
                "high_severity_count": high_count,
                "source": "mock_database",
                "disclaimer": "MOCK DATA — do not use for real clinical decisions",
            },
        )

    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "medication_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of medication names to check",
                    "required": True,
                }
            },
        }


def _normalise_drug_name(name: str) -> str:
    """Lowercase, strip, remove brand name prefixes."""
    return name.lower().strip().split()[0] if name.strip() else ""


def _fuzzy_match(a: str, b: str) -> Optional[Tuple[str, str]]:
    """Partial-match against known interaction keys."""
    for (ka, kb), v in KNOWN_INTERACTIONS.items():
        if (ka in a or a in ka) and (kb in b or b in kb):
            return v
        if (kb in a or a in kb) and (ka in b or b in ka):
            return v
    return None
