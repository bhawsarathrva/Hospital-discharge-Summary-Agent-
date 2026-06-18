"""
utils/token_counter.py
Rough token budget tracker (no tiktoken needed).
Uses character-based approximation: ~4 chars per token for English text.
"""
from __future__ import annotations

CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


class TokenBudget:
    def __init__(self, budget: int):
        self.budget = budget
        self.used = 0

    def consume(self, text: str) -> bool:
        """Return True if budget allows, False if exceeded."""
        tokens = estimate_tokens(text)
        self.used += tokens
        return self.used <= self.budget

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self.used)

    @property
    def exhausted(self) -> bool:
        return self.used >= self.budget
