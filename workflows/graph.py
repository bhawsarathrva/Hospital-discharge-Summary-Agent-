"""
workflows/graph.py
State graph definition for the multi-agent discharge summary pipeline.
Compiles a lightweight Directed Acyclic Graph (DAG) workflow runner.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional
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

logger = get_logger("workflow_graph")


class StateGraph:
    """
    StateGraph defines a lightweight agentic state transition engine.
    """

    def __init__(self):
        self.nodes: Dict[str, Callable[[Any], Any]] = {}
        self.edges: List[tuple[str, str]] = []
        self.entry_point: Optional[str] = None

    def add_node(self, name: str, action: Callable[[Any], Any]) -> None:
        """Register an agent or tool execution node in the graph."""
        self.nodes[name] = action

    def add_edge(self, start: str, end: str) -> None:
        """Register a transition edge from one node to another."""
        self.edges.append((start, end))

    def set_entry_point(self, name: str) -> None:
        """Set the starting execution node of the graph."""
        self.entry_point = name

    def compile(self) -> CompiledGraph:
        """Compile and validate the state graph into a runnable graph."""
        if not self.entry_point:
            raise ValueError("Entry point must be set before compiling.")
        if self.entry_point not in self.nodes:
            raise ValueError(
                f"Entry point '{self.entry_point}' is not registered as a node."
            )
        return CompiledGraph(self)


class CompiledGraph:
    """
    CompiledGraph executes compiled state transition loops.
    """

    def __init__(self, graph: StateGraph):
        self.graph = graph

    def invoke(self, initial_state: Any) -> Any:
        """Invoke the graph execution loop with the initial state."""
        current_node = self.graph.entry_point
        state = initial_state

        while current_node:
            logger.info(f"Executing workflow node: {current_node}")
            action = self.graph.nodes[current_node]
            try:
                state = action(state)
            except Exception as exc:
                logger.error(f"Error executing node '{current_node}': {exc}")
                raise exc

            # Determine transition (linear / simple edge lookup)
            transitions = [
                end for start, end in self.graph.edges if start == current_node
            ]
            if transitions:
                current_node = transitions[0]  # Move to the next node
            else:
                current_node = None  # Graph terminal state reached

        return state


def create_discharge_workflow(llm_client: Optional[Any] = None) -> CompiledGraph:
    """
    Factory function to instantiate the multi-agent graph workflow.
    """
    from agents.extractor import ExtractorAgent
    from agents.conflict_agent import ConflictAgent
    from agents.medication_agent import MedicationAgent
    from agents.safety_agent import SafetyAgent
    from agents.summary_agent import SummaryAgent

    workflow = StateGraph()

    # Instantiate the agent handlers
    extractor = ExtractorAgent(llm_client)
    conflict_detector = ConflictAgent(llm_client)
    med_reconciler = MedicationAgent(llm_client)
    safety_auditor = SafetyAgent(llm_client)
    summary_assembler = SummaryAgent(llm_client)

    # Register graph nodes
    workflow.add_node("extractor", extractor.run)
    workflow.add_node("conflict_detector", conflict_detector.run)
    workflow.add_node("medication_reconciler", med_reconciler.run)
    workflow.add_node("safety_auditor", safety_auditor.run)
    workflow.add_node("summary_assembler", summary_assembler.run)

    # Register linear graph transitions
    workflow.add_edge("extractor", "conflict_detector")
    workflow.add_edge("conflict_detector", "medication_reconciler")
    workflow.add_edge("medication_reconciler", "safety_auditor")
    workflow.add_edge("safety_auditor", "summary_assembler")

    workflow.set_entry_point("extractor")
    return workflow.compile()
