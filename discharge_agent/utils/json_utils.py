"""
utils/json_utils.py
Safe JSON parsing utilities for LLM outputs.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional


def safe_json_parse(text: str) -> Optional[dict]:
    """
    Attempt to parse JSON from LLM output.
    Handles: direct JSON, JSON in markdown code blocks, JSON embedded in prose.
    Returns None if all attempts fail.
    """
    if not text or not text.strip():
        return None

    # 1. Direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown code fences
    stripped = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # 3. Find first {...} block
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # 4. Find largest {...} block (greedy)
    matches = re.findall(r"\{.*\}", text, re.DOTALL)
    for m in sorted(matches, key=len, reverse=True):
        try:
            return json.loads(m)
        except json.JSONDecodeError:
            continue

    return None


def extract_list_from_text(text: str, key: str) -> list:
    """Extract a list from a JSON object by key, or return []."""
    parsed = safe_json_parse(text)
    if parsed and isinstance(parsed, dict):
        val = parsed.get(key, [])
        if isinstance(val, list):
            return val
    return []
