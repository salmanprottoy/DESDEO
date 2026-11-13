"""Grounded, read-only conversational explanations for XNIMBUS."""

import hashlib
import json
from datetime import UTC, datetime
from typing import Annotated, cast

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, col, select

from desdeo.api.evidence import EvidenceContext, build_evidence_context
from desdeo.api.grounded_chat import (
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
    messages = db.exec(
        select(ChatMessage).where(ChatMessage.conversation_id == conversation_id).order_by(col(ChatMessage.id))
    ).all()
    return [
        HistoryTurn(role="user" if message.role == "user" else "assistant", content=message.content)
        for message in messages
        if message.verification_status in HISTORY_STATUSES
    ]


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
    evidence = build_evidence_context(
        state_id=request.state_id,
        user_id=context.user.id,
        session_id=context.user.active_session_id,
        problem_id=problem_db.id,
        problem=Problem.from_problemdb(problem_db),
        solver_results=solver_results,
        solution_indices=list(range(num_solutions)),
        preference=preference.model_dump() if preference is not None else None,
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
