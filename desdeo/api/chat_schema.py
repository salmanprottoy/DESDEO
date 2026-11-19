"""Non-destructive schema initialization for grounded XNIMBUS chat."""

from sqlalchemy import inspect
from sqlalchemy.engine import Engine
from sqlmodel import SQLModel

import desdeo.api.models  # noqa: F401

CHAT_TABLE_NAMES = frozenset({"chat_conversations", "chat_messages"})


def ensure_chat_schema(engine: Engine) -> list[str]:
    """Create missing chat tables and return their names."""
    existing = set(inspect(engine).get_table_names())
    missing = [SQLModel.metadata.tables[name] for name in CHAT_TABLE_NAMES if name not in existing]
    if not missing:
        return []

    SQLModel.metadata.create_all(engine, tables=missing, checkfirst=True)
    created = set(inspect(engine).get_table_names()) - existing
    return sorted(created & CHAT_TABLE_NAMES)
