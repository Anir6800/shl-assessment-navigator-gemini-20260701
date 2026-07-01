"""Standalone smoke evaluator for the catalog retriever and chat service."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from main import app


def main() -> None:
    with TestClient(app) as client:
        scenarios = [
            ("vague", {"messages": [{"role": "user", "content": "I need an assessment."}]}),
            (
                "java",
                {"messages": [{"role": "user", "content": "Hiring a Java developer who works with stakeholders."}]},
            ),
            ("compare", {"messages": [{"role": "user", "content": "OPQ32r vs GSA"}]}),
        ]
        for name, payload in scenarios:
            response = client.post("/chat", json=payload)
            print(f"[{name}] {response.status_code} {response.json()['end_of_conversation']}")
            print(response.json()["reply"])
            print("---")


if __name__ == "__main__":
    main()
