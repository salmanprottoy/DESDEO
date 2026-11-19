"""Tests for deterministic XNIMBUS evidence."""

from desdeo.api.evidence import (
    EvidenceContext,
    EvidenceToolRegistry,
    InsufficientEvidence,
    build_evidence_context,
    compare_solutions,
)
from desdeo.problem.testproblems import river_pollution_problem
from desdeo.tools import SolverResults


def _result(
    *,
    objectives: dict[str, float],
    variables: dict[str, float] | None = None,
    multipliers: dict[str, float] | None = None,
) -> SolverResults:
    return SolverResults(
        optimal_variables=variables or {"x_1": 0.5, "x_2": 0.6},
        optimal_objectives=objectives,
        lagrange_multipliers=multipliers,
        success=True,
        message="optimal",
    )


def test_build_evidence_context_is_scoped_and_contains_solution_facts() -> None:
    """Evidence contains state-scoped objective and explanation facts."""
    problem = river_pollution_problem(five_objective_variant=False)
    context = build_evidence_context(
        state_id=42,
        problem_id=7,
        problem=problem,
        solver_results=[
            _result(
                objectives={"f_1": 5.2, "f_2": 3.0, "f_3": 6.0, "f_4": -2.0},
                multipliers={"f_1_lt": 1.0, "f_2_lt": 2.0, "f_3_lt": 0.0, "f_4_lt": 4.0},
            )
        ],
        actual_solver="test_solver",
    )

    assert isinstance(context, EvidenceContext)
    assert context.state_id == 42
    assert context.problem_id == 7
    objective_fact = next(fact for fact in context.facts if fact.kind == "objective_value")
    assert objective_fact.value == {"symbol": "f_1", "name": "WQCity", "value": 5.2}
    assert objective_fact.source["state_id"] == 42
    assert any(fact.kind == "tradeoff" for fact in context.facts)
    assert any(fact.kind == "active_objective" for fact in context.facts)
    tradeoff = next(fact for fact in context.facts if fact.kind == "tradeoff")
    assert tradeoff.value["semantics"] == "effective_local_matrix_entry"
    assert tradeoff.value["formula"] == "-lambda_column/lambda_row"
    multiplier = next(fact for fact in context.facts if fact.kind == "lagrange_multiplier")
    assert multiplier.source["raw_dual_key"] == "f_1_lt"


def test_near_zero_multipliers_never_produce_tradeoff_facts() -> None:
    """Numerically unstable ratios are omitted and represented as unavailable."""
    problem = river_pollution_problem(five_objective_variant=False)
    context = build_evidence_context(
        state_id=42,
        problem=problem,
        solver_results=[
            _result(
                objectives={"f_1": 5.2, "f_2": 3.0, "f_3": 6.0, "f_4": -2.0},
                multipliers={"f_1_lt": 1.0, "f_2_lt": 5e-8, "f_3_lt": 4e-8, "f_4_lt": 3e-8},
            )
        ],
        actual_solver="test_solver",
    )

    assert not any(fact.kind == "tradeoff" for fact in context.facts)
    availability = next(fact for fact in context.facts if fact.kind == "explanation_availability")
    assert availability.value["available"] is False
    assert availability.value["reason"] == "no_reliable_tradeoffs"


def test_missing_multipliers_are_explicitly_unavailable() -> None:
    """Missing duals are not silently presented as an absence of trade-offs."""
    problem = river_pollution_problem(five_objective_variant=False)
    context = build_evidence_context(
        state_id=42,
        problem=problem,
        solver_results=[_result(objectives={"f_1": 5.2, "f_2": 3.0, "f_3": 6.0, "f_4": -2.0})],
    )

    availability = next(fact for fact in context.facts if fact.kind == "explanation_availability")
    assert availability.value == {"available": False, "reason": "missing_lagrange_multipliers"}


def test_unknown_actual_solver_blocks_quotable_tradeoffs() -> None:
    """The request's stored solver label is never treated as actual solver provenance."""
    problem = river_pollution_problem(five_objective_variant=False)
    context = build_evidence_context(
        state_id=42,
        problem=problem,
        solver_results=[
            _result(
                objectives={"f_1": 5.2, "f_2": 3.0, "f_3": 6.0, "f_4": -2.0},
                multipliers={"f_1_lt": 1.0, "f_2_lt": 2.0, "f_3_lt": 3.0, "f_4_lt": 4.0},
            )
        ],
    )

    assert not any(fact.kind == "tradeoff" for fact in context.facts)
    availability = next(fact for fact in context.facts if fact.kind == "explanation_availability")
    assert availability.value["reason"] == "actual_solver_unknown"


def test_compare_solutions_returns_directional_objective_deltas() -> None:
    """Comparison returns second-minus-first objective values."""
    first = _result(objectives={"f_1": 5.0, "f_2": 2.0})
    second = _result(objectives={"f_1": 6.5, "f_2": 1.5})

    comparison = compare_solutions(first, second)

    assert comparison == {"f_1": 1.5, "f_2": -0.5}


def test_evidence_context_contains_deterministic_comparison_facts() -> None:
    """Differences and better/worse direction are computed before prompting the model."""
    problem = river_pollution_problem(five_objective_variant=False)
    context = build_evidence_context(
        state_id=42,
        problem=problem,
        solver_results=[
            _result(objectives={"f_1": 5.0, "f_2": 3.0, "f_3": 4.0, "f_4": -2.0}),
            _result(objectives={"f_1": 5.5, "f_2": 2.9, "f_3": 4.0, "f_4": -3.0}),
        ],
    )

    fact = next(
        fact for fact in context.facts if fact.evidence_id == "state-42-comparison-solution-0-solution-1-objective-f_1"
    )
    assert fact.value["delta_second_minus_first"] == 0.5
    assert fact.value["better_solution_index"] == 1


def test_compare_solutions_returns_insufficient_evidence_when_objectives_are_missing() -> None:
    """Comparison reports missing objective values explicitly."""
    result = compare_solutions(
        _result(objectives={"f_1": 5.0}),
        _result(objectives={"f_1": 6.0, "f_2": 1.5}),
    )

    assert isinstance(result, InsufficientEvidence)
    assert result.missing == ["objective:f_2"]


def test_evidence_tools_reject_a_state_outside_the_authorized_scope() -> None:
    """A state ID outside the bound context returns insufficient evidence."""
    context = EvidenceContext(state_id=42, problem_id=7, user_id=1, session_id=3)
    tools = EvidenceToolRegistry(
        context,
        authorized_user_id=1,
        authorized_session_id=3,
        authorized_problem_id=7,
    )

    result = tools.get_solution(state_id=99, solution_index=0, user_id=1, session_id=3)

    assert isinstance(result, InsufficientEvidence)
    assert result.missing == ["state:99"]


def test_evidence_tools_reject_a_cross_user_or_session_request() -> None:
    """User and session mismatches cannot read evidence."""
    context = EvidenceContext(state_id=42, problem_id=7, user_id=1, session_id=3)
    tools = EvidenceToolRegistry(
        context,
        authorized_user_id=1,
        authorized_session_id=3,
        authorized_problem_id=7,
    )

    result = tools.get_solution(state_id=42, solution_index=0, user_id=2, session_id=9)

    assert isinstance(result, InsufficientEvidence)
    assert result.missing == ["user:2", "session:9"]


def test_evidence_tools_return_only_facts_for_requested_solution() -> None:
    """A solution tool response is limited to the requested solution."""
    problem = river_pollution_problem(five_objective_variant=False)
    context = build_evidence_context(
        state_id=42,
        problem=problem,
        solver_results=[
            _result(objectives={"f_1": 5.2, "f_2": 3.0}),
            _result(objectives={"f_1": 5.5, "f_2": 2.8}),
        ],
    )
    tools = EvidenceToolRegistry(context, authorized_problem_id=None)

    result = tools.get_solution(state_id=42, solution_index=1)

    assert isinstance(result, list)
    assert result
    assert all(fact.source["solution_index"] == 1 for fact in result)


def _panel_values(problem, multipliers: dict[str, float], scalarization: str, reference_point=None):
    """What the XNIMBUS panel (``/method/xnimbus/get-multipliers-info``) computes for one solution."""
    from desdeo.api.routers.xnimbus import (  # noqa: PLC0415
        MULTIPLIER_TOL,
        compute_effective_multipliers,
        compute_tradeoffs,
        filter_lagrange_multipliers,
    )
    from desdeo.tools.utils import (  # noqa: PLC0415
        flip_maximized_objective_values,
        get_corrected_ideal,
        get_corrected_nadir,
    )

    symbols = [objective.symbol for objective in problem.objectives]
    selected = filter_lagrange_multipliers(multipliers, symbols)
    effective = compute_effective_multipliers(
        selected_multipliers=selected,
        scalarization=scalarization,
        ideal=get_corrected_ideal(problem),
        nadir=get_corrected_nadir(problem),
        reference_point=(
            flip_maximized_objective_values(problem, reference_point) if reference_point is not None else {}
        ),
    )
    active = [symbol for symbol, (_, raw) in selected.items() if abs(raw) > MULTIPLIER_TOL]
    return effective, compute_tradeoffs(effective), active


def test_multipliers_tradeoffs_and_active_objectives_match_the_xnimbus_panel() -> None:
    """The chat quotes the same scaled multipliers and trade-offs the XNIMBUS panel shows."""
    problem = river_pollution_problem(five_objective_variant=False)
    multipliers = {"f_1_lt": 0.4, "f_2_lt": 0.2, "f_3_lt": 0.0, "f_4_lt": 0.1, "f_4_eq": 9.0}
    context = build_evidence_context(
        state_id=42,
        problem=problem,
        solver_results=[_result(objectives={"f_1": 5.2, "f_2": 3.0, "f_3": 6.0, "f_4": -2.0}, multipliers=multipliers)],
        actual_solver="test_solver",
    )
    effective, matrix, active = _panel_values(problem, multipliers, "NIMBUS")

    quoted = {fact.value["symbol"]: fact.value["value"] for fact in context.facts if fact.kind == "lagrange_multiplier"}
    assert quoted == effective
    tradeoffs = {
        (fact.value["row_symbol"], fact.value["column_symbol"]): fact.value["value"]
        for fact in context.facts
        if fact.kind == "tradeoff"
    }
    assert tradeoffs == {(row, column): matrix[row][column] for row in active for column in active if row != column}
    assert [fact.value["symbol"] for fact in context.facts if fact.kind == "active_objective"] == active
    assert active == ["f_1", "f_2", "f_4"]


def test_each_solution_uses_its_own_scalarization_scaling() -> None:
    """Solution order follows XNIMBUS: NIMBUS, STOM, ASF, GUESS; STOM scaling uses the reference point."""
    problem = river_pollution_problem(five_objective_variant=False)
    multipliers = {"f_1_lt": 0.4, "f_2_lt": 0.2, "f_3_lt": 0.3, "f_4_lt": 0.1}
    reference_point = {"f_1": 6.0, "f_2": 3.0, "f_3": 6.5, "f_4": -1.0}
    objectives = {"f_1": 5.2, "f_2": 3.0, "f_3": 6.0, "f_4": -2.0}
    context = build_evidence_context(
        state_id=42,
        problem=problem,
        solver_results=[_result(objectives=objectives, multipliers=multipliers)] * 2,
        actual_solver="test_solver",
        reference_point=reference_point,
    )
    stom, _, _ = _panel_values(problem, multipliers, "STOM", reference_point)

    second = {
        fact.value["symbol"]: fact.value["value"]
        for fact in context.facts
        if fact.kind == "lagrange_multiplier" and fact.source["solution_index"] == 1
    }
    assert second == stom
    assert {
        fact.source["scalarization"]
        for fact in context.facts
        if fact.kind == "lagrange_multiplier" and fact.source["solution_index"] == 1
    } == {"STOM"}


def test_scaling_that_needs_a_missing_reference_point_is_unavailable() -> None:
    """Without the reference point, STOM multipliers cannot be scaled, so no trade-offs are quoted."""
    problem = river_pollution_problem(five_objective_variant=False)
    multipliers = {"f_1_lt": 0.4, "f_2_lt": 0.2, "f_3_lt": 0.3, "f_4_lt": 0.1}
    objectives = {"f_1": 5.2, "f_2": 3.0, "f_3": 6.0, "f_4": -2.0}
    context = build_evidence_context(
        state_id=42,
        problem=problem,
        solver_results=[_result(objectives=objectives, multipliers=multipliers)] * 2,
        actual_solver="test_solver",
    )

    second = [fact for fact in context.facts if fact.source.get("solution_index") == 1]
    assert not any(fact.kind in {"tradeoff", "lagrange_multiplier"} for fact in second)
    availability = next(fact for fact in second if fact.kind == "explanation_availability")
    assert availability.value["available"] is False
    assert availability.value["reason"] == "effective_multipliers_unavailable"
