"""
tools/drug_checker.py
Mock drug-interaction lookup tool.
Identifies potential interactions in a list of medications.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

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
        "Meropenem + Valproate: Meropenem significantly reduces valproate levels. Risk of seizure breakthrough.",
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
        "Ciprofloxacin + Antacids: Antacids reduce ciprofloxacin absorption. Separate administration by 2 hours.",
    ),
}

SEVERITY_RANK = {"HIGH": 3, "MODERATE": 2, "LOW": 1}


def check_drug_interactions(medication_names: List[str]) -> List[dict]:
    """Check a list of medication names for known interactions."""
    normalised = [_normalise_drug_name(m) for m in medication_names]
    interactions = []

    for i in range(len(normalised)):
        for j in range(i + 1, len(normalised)):
            a, b = normalised[i], normalised[j]
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

    interactions.sort(key=lambda x: SEVERITY_RANK.get(x["severity"], 0), reverse=True)
    return interactions


def _normalise_drug_name(name: str) -> str:
    return name.lower().strip().split()[0] if name.strip() else ""


def _fuzzy_match(a: str, b: str) -> Optional[Tuple[str, str]]:
    for (ka, kb), v in KNOWN_INTERACTIONS.items():
        if (ka in a or a in ka) and (kb in b or b in kb):
            return v
        if (kb in a or a in kb) and (ka in b or b in ka):
            return v
    return None
