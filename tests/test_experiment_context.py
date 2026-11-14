"""Tests for read-only full-experiment evidence assembly."""

from desdeo.api.experiment_context import ExperimentStateSnapshot, build_experiment_facts
from desdeo.problem.testproblems import river_pollution_problem
from desdeo.tools import SolverResults


def _result(f_1: float, f_2: float, f_3: float, f_4: float) -> SolverResults:
    return SolverResults(
        optimal_variables={"x_1": 0.5, "x_2": 0.6},
        optimal_objectives={"f_1": f_1, "f_2": f_2, "f_3": f_3, "f_4": f_4},
        success=True,
        message="optimal",
    )


def test_build_experiment_facts_captures_lineage_inputs_process_and_results() -> None:
    """One decision experiment contains DM inputs and stored solver results, never generated actions."""
    problem = river_pollution_problem(five_objective_variant=False)
    snapshots = [
        ExperimentStateSnapshot(
            state_id=10,
            parent_state_id=None,
            kind="xnimbus.initialize",
            sequence=1,
            on_selected_lineage=True,
            preference=None,
            solver_results=[_result(5.0, 3.0, 4.0, -2.0)],
        ),
        ExperimentStateSnapshot(
            state_id=11,
            parent_state_id=10,
            kind="xnimbus.solve_candidates",
            sequence=2,
            on_selected_lineage=True,
            preference={"aspiration_levels": {"f_1": 5.5, "f_2": 3.1}},
            solver_results=[
                _result(5.6, 3.2, 4.2, -2.2),
                _result(5.4, 3.3, 4.5, -2.5),
            ],
        ),
    ]

    facts = build_experiment_facts(
        problem=problem,
        session_id=7,
        problem_id=3,
        selected_state_id=11,
        selected_solution_index=0,
        snapshots=snapshots,
    )

    by_id = {fact.evidence_id: fact for fact in facts}
    summary = by_id["experiment-session-7-problem-3-summary"]
    assert summary.kind == "experiment_summary"
    assert summary.value == {
        "session_id": 7,
        "problem_id": 3,
        "selected_state_id": 11,
        "selected_solution_index": 0,
        "state_count": 2,
        "solution_count": 3,
        "read_only": True,
    }
    assert by_id["experiment-state-11"].value["parent_state_id"] == 10
    assert by_id["experiment-state-11-preference"].kind == "preference_history"
    assert by_id["experiment-state-10-solution-0"].value["objective_values"]["f_1"] == 5.0

    comparison = by_id["experiment-selected-11-0-vs-10-0"]
    assert comparison.kind == "history_comparison"
    assert comparison.value["objective_deltas_selected_minus_other"]["f_1"] == 0.6
    assert comparison.value["selected_better_objectives"] == ["f_1", "f_2", "f_3"]
    assert comparison.value["selected_worse_objectives"] == ["f_4"]
    assert not any(fact.kind in {"action", "recommendation", "generated_solution"} for fact in facts)


def test_build_experiment_facts_marks_branches_without_crossing_scope() -> None:
    """Branch membership is data; the experiment builder never follows or invents external states."""
    problem = river_pollution_problem(five_objective_variant=False)
    facts = build_experiment_facts(
        problem=problem,
        session_id=7,
        problem_id=3,
        selected_state_id=12,
        selected_solution_index=0,
        snapshots=[
            ExperimentStateSnapshot(
                state_id=10,
                parent_state_id=None,
                kind="xnimbus.initialize",
                sequence=1,
                on_selected_lineage=True,
                solver_results=[_result(5.0, 3.0, 4.0, -2.0)],
            ),
            ExperimentStateSnapshot(
                state_id=11,
                parent_state_id=10,
                kind="xnimbus.solve_candidates",
                sequence=2,
                on_selected_lineage=False,
                solver_results=[_result(5.1, 3.1, 4.1, -2.1)],
            ),
            ExperimentStateSnapshot(
                state_id=12,
                parent_state_id=10,
                kind="xnimbus.solve_candidates",
                sequence=3,
                on_selected_lineage=True,
                solver_results=[_result(5.2, 3.2, 4.2, -2.2)],
            ),
        ],
    )

    state_11 = next(fact for fact in facts if fact.evidence_id == "experiment-state-11")
    assert state_11.value["on_selected_lineage"] is False
    assert {fact.source["state_id"] for fact in facts if "state_id" in fact.source} <= {10, 11, 12}
