"""Grounded, read-only conversational explanations for XNIMBUS."""

import hashlib
import json
from collections import defaultdict, deque
from datetime import UTC, datetime
from threading import Lock
from time import monotonic
from typing import Annotated, cast

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlmodel import Session, col, select

from desdeo.api.evidence import EvidenceContext, build_evidence_context
from desdeo.api.experiment_context import ExperimentStateSnapshot, build_experiment_facts
from desdeo.api.grounded_chat import (
    MAX_HISTORY_TURNS,
    PROMPT_VERSION,
    ChatRequest,
    ChatResponse,
    GroundedChatService,
    HistoryTurn,
    ModelCatalogResponse,
)
from desdeo.api.llm_config import (
    CloudConsentRequiredError,
    UnknownProviderError,
    available_model_profiles,
    default_profile_id,
    get_llm_configuration,
)
from desdeo.api.models import ChatConversation, ChatMessage, StateDB
from desdeo.api.models.generic_states import StateKind
from desdeo.api.models.state import ResultInterface
from desdeo.problem import Problem

from .utils import ContextField, SessionContext, SessionContextGuard, get_solver_results_at

router = APIRouter(prefix="/method/xnimbus")

HISTORY_STATUSES = {"received", "verified", "partial", "abstained", "rejected", "blocked", "fallback"}
CHAT_RATE_LIMIT = 10
CHAT_RATE_WINDOW_SECONDS = 60
_CHAT_RATE_EVENTS: dict[int, deque[float]] = defaultdict(deque)
_CHAT_RATE_LOCK = Lock()


def _get_or_create_conversation(request: ChatRequest, context: SessionContext) -> ChatConversation:
    db: Session = context.db_session
    problem_db = context.problem_db
    if problem_db is None or problem_db.id is None or context.user.id is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Chat context is incomplete.")
    if request.conversation_id is None:
        conversation = ChatConversation(
            user_id=context.user.id,
            problem_id=problem_db.id,
            session_id=context.user.active_session_id,
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        return conversation

    existing = db.exec(
        select(ChatConversation).where(
            ChatConversation.id == request.conversation_id,
            ChatConversation.user_id == context.user.id,
            ChatConversation.problem_id == problem_db.id,
            ChatConversation.session_id == context.user.active_session_id,
        )
    ).first()
    if existing is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat conversation not found.")
    return existing


def _history(db: Session, conversation_id: int) -> list[HistoryTurn]:
    messages = list(
        reversed(
            db.exec(
                select(ChatMessage)
                .where(
                    ChatMessage.conversation_id == conversation_id,
                    col(ChatMessage.verification_status).in_(HISTORY_STATUSES),
                )
                .order_by(col(ChatMessage.id).desc())
                .limit(MAX_HISTORY_TURNS)
            ).all()
        )
    )
    return [
        HistoryTurn(role="user" if message.role == "user" else "assistant", content=message.content)
        for message in messages
    ]


def _enforce_chat_rate_limit(user_id: int) -> None:
    """Limit model-backed chat traffic per authenticated user in this API process."""
    now = monotonic()
    cutoff = now - CHAT_RATE_WINDOW_SECONDS
    with _CHAT_RATE_LOCK:
        events = _CHAT_RATE_EVENTS[user_id]
        while events and events[0] <= cutoff:
            events.popleft()
        if len(events) >= CHAT_RATE_LIMIT:
            retry_after = max(1, round(events[0] + CHAT_RATE_WINDOW_SECONDS - now))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="XNIMBUS chat rate limit exceeded.",
                headers={"Retry-After": str(retry_after)},
            )
        events.append(now)


def _is_xnimbus_state(state_row: StateDB) -> bool:
    """Accept XNIMBUS states and generic intermediate descendants of one."""
    current: StateDB | None = state_row
    while current is not None and current.base_state is not None:
        kind = current.base_state.kind
        if kind in {
            StateKind.XNIMBUS_INIT,
            StateKind.XNIMBUS_SOLVE,
            StateKind.XNIMBUS_SAVE,
            StateKind.XNIMBUS_FINAL,
        }:
            return True
        if kind != StateKind.GENERIC_INTERMEDIATE:
            return False
        current = current.parent
    return False


def _fact_sheet_hash(evidence: EvidenceContext) -> str:
    canonical = json.dumps(evidence.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _selected_lineage_ids(state_row: StateDB) -> set[int]:
    """Return same-session/problem ancestors of the selected state, including itself."""
    lineage: set[int] = set()
    current: StateDB | None = state_row
    while current is not None and current.id is not None and current.id not in lineage:
        if current.session_id != state_row.session_id or current.problem_id != state_row.problem_id:
            break
        lineage.add(current.id)
        current = current.parent
    return lineage


def _experiment_snapshots(db: Session, selected_state: StateDB) -> list[ExperimentStateSnapshot]:
    """Extract read-only XNIMBUS snapshots inside the selected authorized experiment scope."""
    rows = db.exec(
        select(StateDB)
        .where(
            StateDB.problem_id == selected_state.problem_id,
            StateDB.session_id == selected_state.session_id,
        )
        .order_by(col(StateDB.id))
    ).all()
    lineage_ids = _selected_lineage_ids(selected_state)
    snapshots: list[ExperimentStateSnapshot] = []
    for row in rows:
        if row.id is None or row.state is None or not _is_xnimbus_state(row):
            continue
        actual_state = cast(ResultInterface, row.state)
        results = []
        for solution_index in range(actual_state.num_solutions):
            try:
                results.append(get_solver_results_at(actual_state, solution_index))
            except (AttributeError, IndexError):
                continue
        preference = getattr(actual_state, "preferences", None)
        if preference is None:
            preference = getattr(actual_state, "reference_point", None)
        if preference is None:
            preference = getattr(actual_state, "previous_preferences", None)
        if isinstance(preference, BaseModel):
            preference = preference.model_dump()
        snapshots.append(
            ExperimentStateSnapshot(
                state_id=row.id,
                parent_state_id=row.parent_id,
                kind=row.base_state.kind.value if row.base_state is not None else "unknown",
                sequence=len(snapshots) + 1,
                on_selected_lineage=row.id in lineage_ids,
                preference=preference if isinstance(preference, dict) else None,
                solver_results=results,
            )
        )
    return snapshots


@router.get("/chat/models", response_model=ModelCatalogResponse)
def chat_models(
    context: Annotated[SessionContext, Depends(SessionContextGuard().get)],
) -> ModelCatalogResponse:
    """List configured server-owned model profiles without exposing credentials."""
    del context
    profiles = available_model_profiles()
    return ModelCatalogResponse(models=profiles, default_profile_id=default_profile_id(profiles))


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    context: Annotated[SessionContext, Depends(SessionContextGuard(require=[ContextField.PROBLEM]).post)],
) -> ChatResponse:
    """Answer from the explicitly selected XNIMBUS state and solution."""
    db: Session = context.db_session
    problem_db = context.problem_db
    if problem_db is None or problem_db.id is None or context.user.id is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Chat context is incomplete.")
    _enforce_chat_rate_limit(context.user.id)

    state_row = db.exec(
        select(StateDB).where(
            StateDB.id == request.state_id,
            StateDB.problem_id == problem_db.id,
            StateDB.session_id == context.user.active_session_id,
        )
    ).first()
    if state_row is None or state_row.state is None or not _is_xnimbus_state(state_row):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="XNIMBUS state not found.")

    actual_state = cast(ResultInterface, state_row.state)
    num_solutions = actual_state.num_solutions
    if request.solution_index >= num_solutions:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Solution {request.solution_index} is not available in the selected state.",
        )
    try:
        solver_results = [get_solver_results_at(actual_state, index) for index in range(num_solutions)]
    except (AttributeError, IndexError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The selected state does not contain a readable solution.",
        ) from error

    try:
        client, profile = get_llm_configuration(
            request.model_profile_id,
            cloud_consent=request.cloud_consent,
        )
    except CloudConsentRequiredError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    except UnknownProviderError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error

    conversation = _get_or_create_conversation(request, context)
    if conversation.id is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Conversation could not be created.",
        )
    history = _history(db, conversation.id)

    preference = getattr(actual_state, "preferences", None)
    if preference is None:
        preference = getattr(actual_state, "previous_preferences", None)
    problem = Problem.from_problemdb(problem_db)
    evidence = build_evidence_context(
        state_id=request.state_id,
        user_id=context.user.id,
        session_id=context.user.active_session_id,
        problem_id=problem_db.id,
        problem=problem,
        solver_results=solver_results,
        solution_indices=list(range(num_solutions)),
        preference=preference.model_dump() if preference is not None else None,
    )
    evidence.facts.extend(
        build_experiment_facts(
            problem=problem,
            session_id=context.user.active_session_id,
            problem_id=problem_db.id,
            selected_state_id=request.state_id,
            selected_solution_index=request.solution_index,
            snapshots=_experiment_snapshots(db, state_row),
        )
    )
    evidence_hash = _fact_sheet_hash(evidence)

    db.add(
        ChatMessage(
            conversation_id=conversation.id,
            role="user",
            content=request.message,
            state_id=request.state_id,
            solution_index=request.solution_index,
            evidence_json=[],
            verification_status="received",
            model_profile_id=profile.profile_id if profile is not None else request.model_profile_id,
        )
    )
    conversation.updated_at = datetime.now(UTC)
    db.add(conversation)

    outcome = GroundedChatService(client=client).respond(
        request.message,
        context=evidence,
        profile=profile,
        history=history,
        focus_solution_index=request.solution_index,
    )

    facts_by_id = {fact.evidence_id: fact for fact in evidence.facts}
    cited_evidence = [facts_by_id[evidence_id] for evidence_id in outcome.cited_evidence_ids]
    db.add(
        ChatMessage(
            conversation_id=conversation.id,
            role="assistant",
            content=outcome.answer.answer,
            state_id=request.state_id,
            solution_index=request.solution_index,
            evidence_json=[fact.model_dump(mode="json") for fact in evidence.facts],
            verification_status=outcome.status,
            intent=outcome.intent,
            rejection_reason=outcome.reason,
            model_profile_id=profile.profile_id if profile is not None else None,
            model_answer_json={
                "attempts": [answer.model_dump(mode="json") for answer in outcome.model_answers],
                "audit": {
                    "provider": profile.provider if profile is not None else None,
                    "model_id": profile.model_id if profile is not None else None,
                    "prompt_version": PROMPT_VERSION,
                    "decoding_parameters": profile.decoding_parameters if profile is not None else None,
                    "fact_sheet_hash": evidence_hash,
                    "cited_evidence_ids": outcome.cited_evidence_ids,
                    "verifier_reasons": outcome.verifier_reasons,
                    "tool_trace": [],
                    "generation_attempts": outcome.attempts,
                    "latency_ms": outcome.latency_ms,
                    "token_usage": outcome.token_usage,
                    "cloud_consent": request.cloud_consent,
                },
            },
        )
    )
    db.commit()

    return ChatResponse(
        answer=outcome.answer.answer,
        claims=outcome.answer.claims,
        evidence=cited_evidence,
        uncertainty=outcome.answer.uncertainty,
        follow_up_questions=outcome.answer.follow_up_questions,
        status=outcome.status,
        intent=outcome.intent,
        verified=outcome.status == "verified",
        conversation_id=conversation.id,
        state_id=request.state_id,
        solution_index=request.solution_index,
        model_profile=profile,
    )
