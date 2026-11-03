"""Client for the Unit 4 final-benchmark scoring API -- a subset (20
questions, level 1) of the GAIA benchmark.

Confirmed live via a direct GET /questions call before writing anything
against it (same verify-first habit as the rest of this repo): 20
questions, 17 text-only, one each with a .png (chess position), .py
(code to execute), and two .mp3 attachments.

https://huggingface.co/learn/agents-course/en/unit4/hands-on
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import requests

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"


@dataclass
class GaiaQuestion:
    task_id: str
    question: str
    level: str
    file_name: str

    @property
    def has_file(self) -> bool:
        return bool(self.file_name)


class GaiaClient:
    """Thin wrapper around the four scoring-API routes."""

    def __init__(self, api_url: str = DEFAULT_API_URL) -> None:
        self.api_url = api_url

    def fetch_questions(self) -> list[GaiaQuestion]:
        response = requests.get(f"{self.api_url}/questions", timeout=15)
        response.raise_for_status()
        return [
            GaiaQuestion(
                task_id=row["task_id"],
                question=row["question"],
                level=row["Level"],
                file_name=row["file_name"],
            )
            for row in response.json()
        ]

    def fetch_random_question(self) -> GaiaQuestion:
        response = requests.get(f"{self.api_url}/random-question", timeout=15)
        response.raise_for_status()
        row = response.json()
        return GaiaQuestion(
            task_id=row["task_id"],
            question=row["question"],
            level=row["Level"],
            file_name=row["file_name"],
        )

    def download_file(self, task_id: str, file_name: str, dest_dir: Path) -> Path:
        """Downloads the file attached to a task, saved under its real
        filename (GAIA attachments are named `{task_id}.{ext}`)."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / file_name
        response = requests.get(f"{self.api_url}/files/{task_id}", timeout=30)
        response.raise_for_status()
        dest_path.write_bytes(response.content)
        return dest_path

    def submit(self, username: str, agent_code_url: str, answers: list[dict]) -> dict:
        """Posts answers for scoring. `agent_code_url` must be a public
        Space code URL ending in `/tree/main` -- this is what the
        leaderboard links to, so the Space must stay public.

        This is a one-way, publicly-visible action (logs against
        `username` and appears on the leaderboard) -- call it deliberately,
        not as part of a dry run.
        """
        payload = {"username": username, "agent_code": agent_code_url, "answers": answers}
        response = requests.post(f"{self.api_url}/submit", json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
