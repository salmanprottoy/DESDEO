"""Typed, deterministic evidence for grounded XNIMBUS explanations."""

from numbers import Real
from typing import Any

from pydantic import BaseModel, Field

from desdeo.explanations.lagrange import (
    compute_tradeoffs,
    determine_active_objectives,
    filter_constraint_values,
    filter_lagrange_multipliers,
)
from desdeo.problem import Problem
from desdeo.tools import SolverResults


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
        facts = [
            fact
            for fact in self._context.facts
            if fact.source.get("solution_index") == solution_index
        ]
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
            if fact.kind in {"lagrange_multiplier", "tradeoff", "active_objective"}
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

    for solution_index, result in enumerate(solver_results):
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

        if result.lagrange_multipliers is None:
            continue
        filtered = filter_lagrange_multipliers(
            result.lagrange_multipliers,
            objective_symbols=objective_symbols,
        )
        tradeoffs = compute_tradeoffs(filtered)
        active = determine_active_objectives(
            [filtered],
            constraint_values=(
                [filter_constraint_values(result.constraint_values, objective_symbols=objective_symbols)]
                if isinstance(result.constraint_values, dict)
                else None
            ),
            objective_symbols=objective_symbols,
        )[0]
        for symbol, value in filtered.items():
            context.facts.append(
                EvidenceFact(
                    evidence_id=f"state-{state_id}-solution-{solution_index}-multiplier-{symbol}",
                    kind="lagrange_multiplier",
                    value={"symbol": symbol, "value": value},
                    source={**source_base, "field": "solver_results.lagrange_multipliers"},
                )
            )
        if tradeoffs is not None:
            context.facts.append(
                EvidenceFact(
                    evidence_id=f"state-{state_id}-solution-{solution_index}-tradeoffs",
                    kind="tradeoff",
                    value={"matrix": tradeoffs},
                    source={**source_base, "field": "derived.from_lagrange_multipliers"},
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

    return context


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
