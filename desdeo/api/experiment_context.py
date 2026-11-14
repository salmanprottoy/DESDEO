"""Read-only evidence models for one XNIMBUS decision experiment."""

from typing import Any

from pydantic import BaseModel, Field

from desdeo.api.evidence import EvidenceFact
from desdeo.problem import Problem
from desdeo.tools import SolverResults


class ExperimentStateSnapshot(BaseModel):
    """Authorized, immutable inputs extracted from one stored XNIMBUS state."""

    state_id: int
    parent_state_id: int | None
    kind: str
    sequence: int
    on_selected_lineage: bool
    preference: dict[str, Any] | None = None
    solver_results: list[SolverResults] = Field(default_factory=list)


def build_experiment_facts(
    *,
    problem: Problem,
    session_id: int | None,
    problem_id: int,
    selected_state_id: int,
    selected_solution_index: int,
    snapshots: list[ExperimentStateSnapshot],
) -> list[EvidenceFact]:
    """Build deterministic facts for a complete authorized decision experiment."""
    solution_count = sum(len(snapshot.solver_results) for snapshot in snapshots)
    session_key = str(session_id) if session_id is not None else "none"
    facts = [
        EvidenceFact(
            evidence_id=f"experiment-session-{session_key}-problem-{problem_id}-summary",
            kind="experiment_summary",
            value={
                "session_id": session_id,
                "problem_id": problem_id,
                "selected_state_id": selected_state_id,
                "selected_solution_index": selected_solution_index,
                "state_count": len(snapshots),
                "solution_count": solution_count,
                "read_only": True,
            },
            source={"field": "experiment.authorized_scope"},
        )
    ]

    for snapshot in snapshots:
        facts.append(
            EvidenceFact(
                evidence_id=f"experiment-state-{snapshot.state_id}",
                kind="iteration",
                value={
                    "state_id": snapshot.state_id,
                    "parent_state_id": snapshot.parent_state_id,
                    "kind": snapshot.kind,
                    "sequence": snapshot.sequence,
                    "on_selected_lineage": snapshot.on_selected_lineage,
                    "solution_count": len(snapshot.solver_results),
                },
                source={"state_id": snapshot.state_id, "field": "statedb"},
            )
        )
        if snapshot.preference is not None:
            facts.append(
                EvidenceFact(
                    evidence_id=f"experiment-state-{snapshot.state_id}-preference",
                    kind="preference_history",
                    value={
                        "state_id": snapshot.state_id,
                        "sequence": snapshot.sequence,
                        "preference": snapshot.preference,
                    },
                    source={"state_id": snapshot.state_id, "field": "state.preferences"},
                )
            )
        for solution_index, result in enumerate(snapshot.solver_results):
            facts.append(
                EvidenceFact(
                    evidence_id=f"experiment-state-{snapshot.state_id}-solution-{solution_index}",
                    kind="experiment_solution",
                    value={
                        "state_id": snapshot.state_id,
                        "sequence": snapshot.sequence,
                        "solution_index": solution_index,
                        "objective_values": result.optimal_objectives,
                        "variable_values": result.optimal_variables,
                        "success": result.success,
                        "message": result.message,
                    },
                    source={
                        "state_id": snapshot.state_id,
                        "solution_index": solution_index,
                        "field": "state.solver_results",
                    },
                )
            )

    selected = _find_solution(snapshots, selected_state_id, selected_solution_index)
    if selected is None:
        return facts
    selected_snapshot, selected_result = selected
    directions = {objective.symbol: objective.maximize for objective in problem.objectives}
    for snapshot in snapshots:
        for solution_index, result in enumerate(snapshot.solver_results):
            if snapshot.state_id == selected_state_id and solution_index == selected_solution_index:
                continue
            facts.append(
                _comparison_fact(
                    selected_snapshot=selected_snapshot,
                    selected_solution_index=selected_solution_index,
                    selected_result=selected_result,
                    other_snapshot=snapshot,
                    other_solution_index=solution_index,
                    other_result=result,
                    directions=directions,
                )
            )
    return facts


def _find_solution(
    snapshots: list[ExperimentStateSnapshot],
    state_id: int,
    solution_index: int,
) -> tuple[ExperimentStateSnapshot, SolverResults] | None:
    for snapshot in snapshots:
        if snapshot.state_id == state_id and 0 <= solution_index < len(snapshot.solver_results):
            return snapshot, snapshot.solver_results[solution_index]
    return None


def _comparison_fact(
    *,
    selected_snapshot: ExperimentStateSnapshot,
    selected_solution_index: int,
    selected_result: SolverResults,
    other_snapshot: ExperimentStateSnapshot,
    other_solution_index: int,
    other_result: SolverResults,
    directions: dict[str, bool],
) -> EvidenceFact:
    shared = sorted(set(selected_result.optimal_objectives) & set(other_result.optimal_objectives))
    deltas: dict[str, float] = {}
    better: list[str] = []
    worse: list[str] = []
    equal: list[str] = []
    for symbol in shared:
        selected_value = selected_result.optimal_objectives[symbol]
        other_value = other_result.optimal_objectives[symbol]
        if not isinstance(selected_value, int | float) or not isinstance(other_value, int | float):
            continue
        delta = round(float(selected_value) - float(other_value), 12)
        deltas[symbol] = delta
        if delta == 0:
            equal.append(symbol)
        elif (delta > 0) == directions.get(symbol, True):
            better.append(symbol)
        else:
            worse.append(symbol)
    return EvidenceFact(
        evidence_id=(
            f"experiment-selected-{selected_snapshot.state_id}-{selected_solution_index}-"
            f"vs-{other_snapshot.state_id}-{other_solution_index}"
        ),
        kind="history_comparison",
        value={
            "selected_state_id": selected_snapshot.state_id,
            "selected_sequence": selected_snapshot.sequence,
            "selected_solution_index": selected_solution_index,
            "other_state_id": other_snapshot.state_id,
            "other_sequence": other_snapshot.sequence,
            "other_solution_index": other_solution_index,
            "objective_deltas_selected_minus_other": deltas,
            "selected_better_objectives": better,
            "selected_worse_objectives": worse,
            "equal_objectives": equal,
        },
        source={
            "state_id": selected_snapshot.state_id,
            "compared_state_id": other_snapshot.state_id,
            "field": "derived.experiment_comparison",
        },
    )
