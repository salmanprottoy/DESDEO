"""Database models for grounded XNIMBUS conversations."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


class ChatConversation(SQLModel, table=True):
    """A conversation bound to one user, problem, and interactive session."""

    __tablename__ = "chat_conversations"

    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    problem_id: int = Field(foreign_key="problemdb.id", index=True)
    session_id: int | None = Field(default=None, foreign_key="interactivesessiondb.id", index=True)
    method: str = Field(default="xnimbus")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ChatMessage(SQLModel, table=True):
    """A message with the exact state and evidence snapshot used for its answer."""

    __tablename__ = "chat_messages"

    id: int | None = Field(default=None, primary_key=True)
    conversation_id: int = Field(foreign_key="chat_conversations.id", index=True)
    role: str
    content: str
    state_id: int
    solution_index: int = 0
    evidence_json: list[dict[str, Any]] = Field(sa_column=Column(JSON, nullable=False))
    verification_status: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
