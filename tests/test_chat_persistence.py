"""Tests for persisted grounded-chat context."""

from datetime import UTC, datetime

from sqlmodel import Session, SQLModel, create_engine, select

from desdeo.api.models.chat import ChatConversation, ChatMessage


def test_chat_messages_persist_state_context_and_evidence_snapshot() -> None:
    """Persist state and evidence context with each chat message."""
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        conversation = ChatConversation(user_id=1, problem_id=2, session_id=3)
        session.add(conversation)
        session.commit()
        session.refresh(conversation)

        message = ChatMessage(
            conversation_id=conversation.id,
            role="assistant",
            content="The objective value is supported by evidence.",
            state_id=42,
            solution_index=1,
            evidence_json=[{"evidence_id": "state-42-solution-1-objective-f_1"}],
            verification_status="verified",
            created_at=datetime.now(UTC),
        )
        session.add(message)
        session.commit()

        stored = session.exec(select(ChatMessage)).one()
        assert stored.state_id == 42
        assert stored.solution_index == 1
        assert stored.evidence_json[0]["evidence_id"] == "state-42-solution-1-objective-f_1"
