"""Runs GaiaAgent over the real 20 benchmark questions and prints each
answer for manual inspection -- NOT a submission. Ground truth is withheld
server-side, so there's no local pass/fail check like
unit3_agentic_rag/invitees/eval_alfred.py; this is purely a sanity pass to
catch malformed answers (wrong format, refusals, crashes) before anyone
spends a real submission on them.

Submitting for a real score requires a public Hugging Face Space (the
template at agents-course/Final_Assignment_Template, duplicated) plus a
Gradio login -- GaiaClient.submit() exists for that step, but it's
deliberately not called from here.

https://huggingface.co/learn/agents-course/en/unit4/hands-on
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

from huggingface_agent.unit4_final_benchmark.app import GaiaAgent
from huggingface_agent.unit4_final_benchmark.gaia_client import GaiaClient


def dry_run(limit: int | None = None) -> None:
    client = GaiaClient()
    questions = client.fetch_questions()
    if limit:
        questions = questions[:limit]

    agent = GaiaAgent()
    for q in questions:
        print(f"\n=== {q.task_id} (level {q.level}, file={q.file_name!r}) ===")
        print(f"Q: {q.question[:200]}")
        try:
            answer = agent(q.question, task_id=q.task_id, file_name=q.file_name)
        except Exception as e:
            answer = f"<agent error: {e!s}>"
        print(f"A: {answer}")


if __name__ == "__main__":
    dry_run()
