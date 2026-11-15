"""Tests for non-destructive grounded-chat schema initialization."""

from sqlalchemy import inspect
from sqlmodel import SQLModel, create_engine

from desdeo.api.chat_schema import ensure_chat_schema
from desdeo.api.models import ChatConversation, ChatMessage


def test_chat_schema_initializer_adds_only_missing_chat_tables_and_is_idempotent() -> None:
    """Existing databases gain the two chat tables without recreating other schema."""
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    ChatMessage.__table__.drop(engine)
    ChatConversation.__table__.drop(engine)
    tables_before = set(inspect(engine).get_table_names())

    created = ensure_chat_schema(engine)

    tables_after = set(inspect(engine).get_table_names())
    assert created == ["chat_conversations", "chat_messages"]
    assert tables_after - tables_before == {"chat_conversations", "chat_messages"}
    assert ensure_chat_schema(engine) == []
