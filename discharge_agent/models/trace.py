"""
models/trace.py
Agent step tracing and observability.
Every step is recorded: reasoning → tool → inputs → result → next decision.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class StepStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRIED = "retried"


@dataclass
class AgentStep:
    step_number: int
    timestamp: float = field(default_factory=time.time)

    # ReAct components
    reasoning: str = ""                  # LLM's thought before acting
    tool_name: Optional[str] = None      # Tool selected
    tool_inputs: Dict[str, Any] = field(default_factory=dict)
    tool_result: Optional[Any] = None
    tool_status: StepStatus = StepStatus.SUCCESS

    # After observing result
    next_decision: str = ""              # What the agent decided to do next
    flags_raised: List[str] = field(default_factory=list)

    # Metadata
    retry_count: int = 0
    duration_s: float = 0.0
    tokens_used: int = 0

    def to_dict(self) -> dict:
        return {
            "step": self.step_number,
            "timestamp": self.timestamp,
            "reasoning": self.reasoning,
            "tool_name": self.tool_name,
            "tool_inputs": self.tool_inputs,
            "tool_result": self.tool_result
                if not isinstance(self.tool_result, bytes)
                else "<binary>",
            "tool_status": self.tool_status.value,
            "next_decision": self.next_decision,
            "flags_raised": self.flags_raised,
            "retry_count": self.retry_count,
            "duration_s": round(self.duration_s, 3),
            "tokens_used": self.tokens_used,
        }


@dataclass
class AgentTrace:
    patient_id: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    steps: List[AgentStep] = field(default_factory=list)
    termination_reason: str = ""   # "plan_complete" | "step_cap_reached" | "error"
    total_tokens: int = 0
    total_flags: int = 0

    def add_step(self, step: AgentStep) -> None:
        self.steps.append(step)
        self.total_tokens += step.tokens_used
        self.total_flags += len(step.flags_raised)

    def finish(self, reason: str) -> None:
        self.end_time = time.time()
        self.termination_reason = reason

    @property
    def duration_s(self) -> float:
        if self.end_time:
            return round(self.end_time - self.start_time, 2)
        return round(time.time() - self.start_time, 2)

    def to_dict(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_s": self.duration_s,
            "total_steps": len(self.steps),
            "termination_reason": self.termination_reason,
            "total_tokens": self.total_tokens,
            "total_flags": self.total_flags,
            "steps": [s.to_dict() for s in self.steps],
        }

    def to_readable(self) -> str:
        """Human-readable trace for debugging."""
        lines = [
            f"=== AGENT TRACE: {self.patient_id} ===",
            f"Duration: {self.duration_s}s | Steps: {len(self.steps)} | "
            f"Tokens: {self.total_tokens} | Flags: {self.total_flags}",
            f"Terminated: {self.termination_reason}",
            "",
        ]
        for step in self.steps:
            lines += [
                f"── Step {step.step_number} ──────────────────────────────",
                f"  REASON : {step.reasoning[:200]}...",
                f"  ACTION : {step.tool_name}({json.dumps(step.tool_inputs, default=str)[:150]})",
                f"  STATUS : {step.tool_status.value} | retries={step.retry_count} | {step.duration_s:.2f}s",
                f"  RESULT : {str(step.tool_result)[:200]}",
                f"  NEXT   : {step.next_decision[:150]}",
            ]
            if step.flags_raised:
                for f in step.flags_raised:
                    lines.append(f"  🚩 FLAG: {f}")
            lines.append("")
        return "\n".join(lines)
