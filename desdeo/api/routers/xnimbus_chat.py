"""Grounded conversational explanations for XNIMBUS.

The chat endpoint answers questions about the current XNIMBUS state. Answers come from a language
model but are only shown after deterministic verification against evidence built from the state.
The endpoint never changes the optimization session.
"""

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from desdeo.api.evidence import build_evidence_context
from desdeo.api.grounded_chat import ChatRequest, ChatResponse, GroundedChatService, HistoryTurn
from desdeo.api.llm_config import ProviderError, UnknownProviderError, get_llm_configuration
from desdeo.api.models import ChatConversation, ChatMessage, StateDB
from desdeo.problem import Problem

from .utils import ContextField, SessionContext, SessionContextGuard

router = APIRouter(prefix="/method/xnimbus")

HISTORY_STATUSES = {"received", "verified", "partial", "abstained", "rejected", "blocked"}
"""Message statuses replayed to the model as conversation history."""


def _get_or_create_conversation(request: ChatRequest, context: SessionContext) -> ChatConversation:
    db: Session = context.db_session
    if request.conversation_id is None:
        conversation = ChatConversation(
            user_id=context.user.id,
            problem_id=context.problem_db.id,
            session_id=context.user.active_session_id,
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        return conversation

    conversation = db.exec(
        select(ChatConversation).where(
            ChatConversation.id == request.conversation_id,
            ChatConversation.user_id == context.user.id,
            ChatConversation.problem_id == context.problem_db.id,
            ChatConversation.session_id == context.user.active_session_id,
        )
    ).first()
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat conversation not found.")
    return conversation


def _history(db: Session, conversation_id: int) -> list[HistoryTurn]:
    messages = db.exec(
        select(ChatMessage).where(ChatMessage.conversation_id == conversation_id).order_by(ChatMessage.id)
    ).all()
    return [
        HistoryTurn(role="user" if message.role == "user" else "assistant", content=message.content)
        for message in messages
        if message.verification_status in HISTORY_STATUSES
    ]


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    context: Annotated[SessionContext, Depends(SessionContextGuard(require=[ContextField.PROBLEM]).post)],
) -> ChatResponse:
    """Answer a question about an XNIMBUS state with a verified answer or an explicit abstention."""
    db: Session = context.db_session
    state_row = db.exec(
        select(StateDB).where(
            StateDB.id == request.state_id,
            StateDB.problem_id == context.problem_db.id,
            StateDB.session_id == context.user.active_session_id,
        )
    ).first()
    if state_row is None or state_row.state is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="XNIMBUS state not found.")

    state = state_row.state
    solver_results = getattr(state, "solver_results", None)
    if solver_results is None:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="State has no solver results.")
    if not isinstance(solver_results, list):
        solver_results = [solver_results]
    if request.solution_index >= len(solver_results):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Solution {request.solution_index} is not available in the selected state.",
        )

    try:
        client, profile = get_llm_configuration()
    except UnknownProviderError as error:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(error)) from error

    conversation = _get_or_create_conversation(request, context)
    history = _history(db, conversation.id)

    preference = getattr(state, "previous_preferences", None)
    evidence = build_evidence_context(
        state_id=request.state_id,
        user_id=context.user.id,
        session_id=context.user.active_session_id,
        problem_id=context.problem_db.id,
        problem=Problem.from_problemdb(context.problem_db),
        solver_results=solver_results,
        preference=preference.model_dump() if preference is not None else None,
    )

    db.add(
        ChatMessage(
            conversation_id=conversation.id,
            role="user",
            content=request.message,
            state_id=request.state_id,
            solution_index=request.solution_index,
            evidence_json=[],
            verification_status="received",
        )
    )
    conversation.updated_at = datetime.now(UTC)
    db.add(conversation)

    try:
        outcome = GroundedChatService(client=client).respond(
            request.message,
            context=evidence,
            profile=profile,
            history=history,
            focus_solution_index=request.solution_index,
        )
    except ProviderError as error:
        db.add(
            ChatMessage(
                conversation_id=conversation.id,
                role="assistant",
                content="",
                state_id=request.state_id,
                solution_index=request.solution_index,
                evidence_json=[],
                verification_status="provider_error",
                rejection_reason=str(error),
                model_profile_id=profile.profile_id if profile is not None else None,
            )
        )
        db.commit()
        provider = profile.provider if profile is not None else "language model"
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"The configured {provider} provider did not return a valid answer.",
        ) from error

    facts_by_id = {fact.evidence_id: fact for fact in evidence.facts}
    cited_evidence = [facts_by_id[evidence_id] for evidence_id in outcome.cited_evidence_ids]
    db.add(
        ChatMessage(
            conversation_id=conversation.id,
            role="assistant",
            content=outcome.answer.answer,
            state_id=request.state_id,
            solution_index=request.solution_index,
            evidence_json=[fact.model_dump() for fact in cited_evidence],
            verification_status=outcome.status,
            intent=outcome.intent,
            rejection_reason=outcome.reason,
            model_profile_id=profile.profile_id if profile is not None else None,
            model_answer_json=outcome.model_answer.model_dump() if outcome.model_answer is not None else None,
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
    )
