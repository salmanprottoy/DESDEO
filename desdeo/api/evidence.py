"""Typed, deterministic evidence for grounded XNIMBUS explanations."""

from collections import defaultdict
from itertools import combinations
from math import isfinite
from numbers import Real
from typing import Any

from pydantic import BaseModel, Field

from desdeo.explanations.lagrange import (
    determine_active_objectives,
    filter_constraint_values,
)
from desdeo.problem import Problem
from desdeo.tools import SolverResults

TRADEOFF_MULTIPLIER_THRESHOLD = 1e-5
"""Smallest multiplier magnitude considered reliable for a trade-off ratio."""


class EvidenceFact(BaseModel):
    """A fact with a stable identifier and source provenance."""

    evidence_id: str
    kind: str
    value: dict[str, Any]
    source: dict[str, int | str]


class InsufficientEvidence(BaseModel):
    """A deterministic result describing information required but unavailable."""

    reason: str = "Insufficient evidence."
    missing: list[str] = Field(default_factory=list)


class EvidenceContext(BaseModel):
    """Evidence restricted to one problem and XNIMBUS state."""

    state_id: int
    user_id: int | None = None
    session_id: int | None = None
    problem_id: int | None = None
    facts: list[EvidenceFact] = Field(default_factory=list)


class EvidenceToolRegistry:
    """Read-only tools over one authorized evidence context."""

    def __init__(
        self,
        context: EvidenceContext,
        *,
        authorized_user_id: int | None = None,
        authorized_session_id: int | None = None,
        authorized_problem_id: int | None,
    ):
        """Create a registry bound to one authorized evidence scope."""
        self._context = context
        self._authorized_user_id = authorized_user_id
        self._authorized_session_id = authorized_session_id
        self._authorized_problem_id = authorized_problem_id

    def _authorize(
        self,
        state_id: int,
        problem_id: int | None = None,
        user_id: int | None = None,
        session_id: int | None = None,
    ) -> InsufficientEvidence | None:
        missing: list[str] = []
        if user_id is not None and user_id != self._authorized_user_id:
            missing.append(f"user:{user_id}")
        if session_id is not None and session_id != self._authorized_session_id:
            missing.append(f"session:{session_id}")
        if missing:
            return InsufficientEvidence(missing=missing)
        if state_id != self._context.state_id:
            return InsufficientEvidence(missing=[f"state:{state_id}"])
        if problem_id is not None and problem_id != self._authorized_problem_id:
            return InsufficientEvidence(missing=[f"problem:{problem_id}"])
        if (
            self._authorized_problem_id is not None
            and self._context.problem_id is not None
            and self._authorized_problem_id != self._context.problem_id
        ):
            return InsufficientEvidence(missing=[f"problem:{self._context.problem_id}"])
        return None

    def get_problem_context(
        self,
        *,
        state_id: int,
        problem_id: int | None = None,
        user_id: int | None = None,
        session_id: int | None = None,
    ) -> list[EvidenceFact] | InsufficientEvidence:
        """Return the problem metadata for the authorized state."""
        if (error := self._authorize(state_id, problem_id, user_id, session_id)) is not None:
            return error
        return [fact for fact in self._context.facts if fact.kind == "problem_context"]

    def get_solution(
        self,
        *,
        state_id: int,
        solution_index: int,
        problem_id: int | None = None,
        user_id: int | None = None,
        session_id: int | None = None,
    ) -> list[EvidenceFact] | InsufficientEvidence:
        """Return facts for one solution in the authorized state."""
        if (error := self._authorize(state_id, problem_id, user_id, session_id)) is not None:
            return error
        if solution_index < 0:
            return InsufficientEvidence(missing=[f"solution:{solution_index}"])
        facts = [fact for fact in self._context.facts if fact.source.get("solution_index") == solution_index]
        if not facts:
            return InsufficientEvidence(missing=[f"solution:{solution_index}"])
        return facts

    def get_preference_context(
        self,
        *,
        state_id: int,
        problem_id: int | None = None,
        user_id: int | None = None,
        session_id: int | None = None,
    ) -> list[EvidenceFact] | InsufficientEvidence:
        """Return preference facts for the authorized state."""
        if (error := self._authorize(state_id, problem_id, user_id, session_id)) is not None:
            return error
        facts = [fact for fact in self._context.facts if fact.kind == "preference"]
        return facts or InsufficientEvidence(missing=["preference"])

    def get_lagrange_explanation(
        self,
        *,
        state_id: int,
        solution_index: int,
        problem_id: int | None = None,
        user_id: int | None = None,
        session_id: int | None = None,
    ) -> list[EvidenceFact] | InsufficientEvidence:
        """Return multiplier, trade-off, and active-objective facts."""
        result = self.get_solution(
            state_id=state_id,
            solution_index=solution_index,
            problem_id=problem_id,
            user_id=user_id,
            session_id=session_id,
        )
        if isinstance(result, InsufficientEvidence):
            return result
        facts = [
            fact
            for fact in result
            if fact.kind in {"lagrange_multiplier", "tradeoff", "active_objective", "explanation_availability"}
        ]
        return facts or InsufficientEvidence(missing=["lagrange_explanation"])

    def get_decision_history(
        self,
        *,
        state_id: int,
        history: list[EvidenceFact] | None = None,
        problem_id: int | None = None,
        user_id: int | None = None,
        session_id: int | None = None,
    ) -> list[EvidenceFact] | InsufficientEvidence:
        """Return supplied state-lineage facts without querying or mutating state."""
        if (error := self._authorize(state_id, problem_id, user_id, session_id)) is not None:
            return error
        return history or InsufficientEvidence(missing=["decision_history"])


def build_evidence_context(
    *,
    state_id: int,
    problem: Problem,
    solver_results: list[SolverResults],
    user_id: int | None = None,
    session_id: int | None = None,
    problem_id: int | None = None,
    preference: dict[str, Any] | None = None,
    solution_indices: list[int] | None = None,
    actual_solver: str | None = None,
) -> EvidenceContext:
    """Build deterministic evidence for a state and its solver results."""
    context = EvidenceContext(
        state_id=state_id,
        user_id=user_id,
        session_id=session_id,
        problem_id=problem_id,
    )
    objective_symbols = [objective.symbol for objective in problem.objectives]

    context.facts.append(
        EvidenceFact(
            evidence_id=f"state-{state_id}-problem-context",
            kind="problem_context",
            value={
                "name": problem.name,
                "objectives": [
                    {
                        "symbol": objective.symbol,
                        "name": objective.name,
                        "maximize": objective.maximize,
                        "ideal": objective.ideal,
                        "nadir": objective.nadir,
                    }
                    for objective in problem.objectives
                ],
            },
            source={"state_id": state_id, "field": "problem"},
        )
    )
    if preference is not None:
        context.facts.append(
            EvidenceFact(
                evidence_id=f"state-{state_id}-preference",
                kind="preference",
                value=preference,
                source={"state_id": state_id, "field": "preference"},
            )
        )

    indices = solution_indices or list(range(len(solver_results)))
    if len(indices) != len(solver_results):
        raise ValueError("solution_indices must have one entry per solver result.")

    for solution_index, result in zip(indices, solver_results, strict=True):
        source_base = {"state_id": state_id, "solution_index": solution_index}
        for symbol, value in result.optimal_objectives.items():
            context.facts.append(
                EvidenceFact(
                    evidence_id=f"state-{state_id}-solution-{solution_index}-objective-{symbol}",
                    kind="objective_value",
                    value={"symbol": symbol, "value": value},
                    source={**source_base, "field": f"solver_results.optimal_objectives.{symbol}"},
                )
            )
        for symbol, value in result.optimal_variables.items():
            context.facts.append(
                EvidenceFact(
                    evidence_id=f"state-{state_id}-solution-{solution_index}-variable-{symbol}",
                    kind="variable_value",
                    value={"symbol": symbol, "value": value},
                    source={**source_base, "field": f"solver_results.optimal_variables.{symbol}"},
                )
            )
        context.facts.append(
            EvidenceFact(
                evidence_id=f"state-{state_id}-solution-{solution_index}-solver-status",
                kind="solver_status",
                value={"success": result.success, "message": result.message},
                source={**source_base, "field": "solver_results"},
            )
        )

        raw_multipliers = _numeric_mapping(result.lagrange_multipliers)
        if not raw_multipliers:
            context.facts.append(
                EvidenceFact(
                    evidence_id=f"state-{state_id}-solution-{solution_index}-explanation-availability",
                    kind="explanation_availability",
                    value={"available": False, "reason": "missing_lagrange_multipliers"},
                    source={**source_base, "field": "solver_results.lagrange_multipliers"},
                )
            )
            continue

        filtered, raw_keys = _filter_multipliers_with_sources(raw_multipliers, objective_symbols)
        constraint_values = _numeric_mapping(result.constraint_values)
        filtered_constraints = (
            filter_constraint_values(constraint_values, objective_symbols=objective_symbols)
            if constraint_values
            else {}
        )
        active = determine_active_objectives(
            [filtered],
            constraint_values=[filtered_constraints] if filtered_constraints else None,
            objective_symbols=objective_symbols,
        )[0]
        for symbol, value in filtered.items():
            raw_dual_key = raw_keys.get(symbol, "unavailable")
            context.facts.append(
                EvidenceFact(
                    evidence_id=f"state-{state_id}-solution-{solution_index}-multiplier-{symbol}",
                    kind="lagrange_multiplier",
                    value={
                        "symbol": symbol,
                        "value": value,
                        "reliable_for_tradeoffs": (
                            actual_solver is not None and abs(value) >= TRADEOFF_MULTIPLIER_THRESHOLD
                        ),
                    },
                    source={
                        **source_base,
                        "field": "solver_results.lagrange_multipliers",
                        "raw_dual_key": raw_dual_key,
                        "actual_solver": actual_solver or "unavailable",
                    },
                )
            )

        reliable_symbols = (
            [symbol for symbol in active if abs(filtered.get(symbol, 0.0)) >= TRADEOFF_MULTIPLIER_THRESHOLD]
            if actual_solver is not None
            else []
        )
        tradeoff_count = 0
        for row_symbol in reliable_symbols:
            for column_symbol in reliable_symbols:
                if row_symbol == column_symbol:
                    continue
                value = -filtered[column_symbol] / filtered[row_symbol]
                tradeoff_count += 1
                context.facts.append(
                    EvidenceFact(
                        evidence_id=(
                            f"state-{state_id}-solution-{solution_index}-tradeoff-"
                            f"row-{row_symbol}-column-{column_symbol}"
                        ),
                        kind="tradeoff",
                        value={
                            "row_symbol": row_symbol,
                            "column_symbol": column_symbol,
                            "value": value,
                            "formula": "-lambda_column/lambda_row",
                            "semantics": "raw_local_matrix_entry",
                            "local": True,
                            "reliability": "reliable",
                        },
                        source={
                            **source_base,
                            "field": "derived.from_lagrange_multipliers",
                            "actual_solver": actual_solver or "unavailable",
                        },
                    )
                )

        unavailable_symbols = sorted(set(objective_symbols) - set(reliable_symbols))
        availability_reason = (
            "actual_solver_unknown"
            if actual_solver is None
            else "reliable_local_tradeoffs_available"
            if tradeoff_count > 0
            else "no_reliable_tradeoffs"
        )
        context.facts.append(
            EvidenceFact(
                evidence_id=f"state-{state_id}-solution-{solution_index}-explanation-availability",
                kind="explanation_availability",
                value={
                    "available": tradeoff_count > 0,
                    "reason": availability_reason,
                    "unreliable_or_inactive_objectives": unavailable_symbols,
                    "multiplier_threshold": TRADEOFF_MULTIPLIER_THRESHOLD,
                },
                source={
                    **source_base,
                    "field": "derived.reliability_rule",
                    "actual_solver": actual_solver or "unavailable",
                },
            )
        )
        for symbol in active:
            context.facts.append(
                EvidenceFact(
                    evidence_id=f"state-{state_id}-solution-{solution_index}-active-{symbol}",
                    kind="active_objective",
                    value={"symbol": symbol},
                    source={**source_base, "field": "derived.from_solver_results"},
                )
            )

    _append_comparison_facts(
        context,
        state_id=state_id,
        problem=problem,
        indexed_results=list(zip(indices, solver_results, strict=True)),
    )
    return context


def _append_comparison_facts(
    context: EvidenceContext,
    *,
    state_id: int,
    problem: Problem,
    indexed_results: list[tuple[int, SolverResults]],
) -> None:
    """Append deterministic pairwise objective comparisons to an evidence context."""
    objective_directions = {objective.symbol: objective.maximize for objective in problem.objectives}
    for (first_index, first_result), (second_index, second_result) in combinations(indexed_results, 2):
        shared_symbols = sorted(set(first_result.optimal_objectives) & set(second_result.optimal_objectives))
        for symbol in shared_symbols:
            first_value = first_result.optimal_objectives[symbol]
            second_value = second_result.optimal_objectives[symbol]
            if not isinstance(first_value, Real) or not isinstance(second_value, Real):
                continue
            first_number = float(first_value)
            second_number = float(second_value)
            delta = second_number - first_number
            maximize = objective_directions.get(symbol)
            better_index: int | None = None
            if delta != 0 and maximize is not None:
                second_is_better = delta > 0 if maximize else delta < 0
                better_index = second_index if second_is_better else first_index
            context.facts.append(
                EvidenceFact(
                    evidence_id=(
                        f"state-{state_id}-comparison-solution-{first_index}-solution-{second_index}-objective-{symbol}"
                    ),
                    kind="comparison",
                    value={
                        "symbol": symbol,
                        "first_solution_index": first_index,
                        "second_solution_index": second_index,
                        "first_value": first_number,
                        "second_value": second_number,
                        "delta_second_minus_first": delta,
                        "maximize": maximize,
                        "better_solution_index": better_index,
                    },
                    source={
                        "state_id": state_id,
                        "field": "derived.compare_solutions",
                    },
                )
            )


def _numeric_mapping(values: object) -> dict[str, float]:
    """Return only finite scalar numeric values from a solver mapping."""
    if not isinstance(values, dict):
        return {}
    return {
        str(key): float(value)
        for key, value in values.items()
        if isinstance(value, Real) and not isinstance(value, bool) and isfinite(float(value))
    }


def _filter_multipliers_with_sources(
    raw_multipliers: dict[str, float],
    objective_symbols: list[str],
) -> tuple[dict[str, float], dict[str, str]]:
    """Filter multipliers while retaining the raw dual key selected for provenance."""
    grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for raw_key, value in raw_multipliers.items():
        for symbol in objective_symbols:
            if symbol in raw_key:
                grouped[symbol].append((raw_key, value))
                break

    filtered: dict[str, float] = {}
    raw_keys: dict[str, str] = {}
    for symbol in objective_symbols:
        entries = grouped.get(symbol, [])
        preferred = next((entry for entry in entries if not entry[0].endswith("eq")), None)
        if preferred is None and entries:
            preferred = entries[0]
        if preferred is None:
            filtered[symbol] = 0.0
            continue
        raw_keys[symbol], filtered[symbol] = preferred

    return filtered, raw_keys


def compare_solutions(
    first: SolverResults | None,
    second: SolverResults | None,
) -> dict[str, float] | InsufficientEvidence:
    """Return second-minus-first objective values, or missing evidence."""
    if first is None or second is None:
        return InsufficientEvidence(missing=["first_solution", "second_solution"])

    missing: list[str] = []
    deltas: dict[str, float] = {}
    symbols = sorted(set(first.optimal_objectives) | set(second.optimal_objectives))
    for symbol in symbols:
        first_value = first.optimal_objectives.get(symbol)
        second_value = second.optimal_objectives.get(symbol)
        if not isinstance(first_value, Real) or not isinstance(second_value, Real):
            missing.append(f"objective:{symbol}")
            continue
        deltas[symbol] = float(second_value) - float(first_value)

    if missing:
        return InsufficientEvidence(missing=missing)
    return deltas
