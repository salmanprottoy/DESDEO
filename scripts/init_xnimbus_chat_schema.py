"""Add missing grounded XNIMBUS chat tables without resetting the database."""

from desdeo.api.chat_schema import ensure_chat_schema
from desdeo.api.db import engine


def main() -> None:
    """Initialize the chat schema and report whether any tables were added."""
    created = ensure_chat_schema(engine)
    if created:
        print(f"Created chat tables: {', '.join(created)}")  # noqa: T201
    else:
        print("Chat tables already exist; no schema changes were needed.")  # noqa: T201


if __name__ == "__main__":
    main()
