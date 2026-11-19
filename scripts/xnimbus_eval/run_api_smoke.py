"""Run one real local-model chat turn through the FastAPI endpoint and persistence layer.

This uses an isolated in-memory database, so it never modifies ``test.db``. Ollama must
already be running and the requested model must be installed.

Usage:
    uv run python scripts/xnimbus_eval/run_api_smoke.py --model qwen2.5:7b-instruct
"""

import argparse
import json
import os
from pathlib import Path

from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, col, create_engine, select
from sqlmodel.pool import StaticPool

from desdeo.api.app import app
from desdeo.api.db import get_session
from desdeo.api.models import ChatMessage, ProblemDB, User, UserRole
from desdeo.api.routers.user_authentication import get_password_hash
from desdeo.problem.testproblems import river_pollution_problem


def main() -> None:
    """Create an isolated XNIMBUS session and persist one real grounded answer."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="installed local OpenAI-compatible model id")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    os.environ["XNIMBUS_LOCAL_MODEL"] = args.model
    os.environ["XNIMBUS_LOCAL_BASE_URL"] = args.base_url

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)

    with Session(engine) as db:
        user = User(
            username="xnimbus-smoke",
            password_hash=get_password_hash("xnimbus-smoke"),
            role=UserRole.analyst,
            group="evaluation",
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        problem_db = ProblemDB.from_problem(
            river_pollution_problem(five_objective_variant=False),
            user=user,
        )
        db.add(problem_db)
        db.commit()
        db.refresh(problem_db)

        def isolated_session():
            yield db

        app.dependency_overrides[get_session] = isolated_session
        try:
            with TestClient(app) as client:
                login = client.post(
                    "/login",
                    data={
                        "username": "xnimbus-smoke",
                        "password": "xnimbus-smoke",
                        "grant_type": "password",
                    },
                    headers={"content-type": "application/x-www-form-urlencoded"},
                )
                login.raise_for_status()
                token = login.json()["access_token"]
                headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

                initialized = client.post(
                    "/method/xnimbus/initialize",
                    json={"problem_id": problem_db.id},
                    headers=headers,
                )
                initialized.raise_for_status()
                state_id = initialized.json()["state_id"]

                response = client.post(
                    "/method/xnimbus/chat",
                    json={
                        "problem_id": problem_db.id,
                        "state_id": state_id,
                        "solution_index": 0,
                        "message": "What are the recorded objective values of this solution?",
                        "model_profile_id": f"local:{args.model}",
                        "cloud_consent": False,
                    },
                    headers=headers,
                )
                response.raise_for_status()
                messages = db.exec(select(ChatMessage).order_by(col(ChatMessage.id))).all()
                result = {
                    "response": response.json(),
                    "persisted_assistant_audit": messages[-1].model_answer_json,
                }
        finally:
            app.dependency_overrides.clear()

    rendered = json.dumps(result, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)  # noqa: T201


if __name__ == "__main__":
    main()
