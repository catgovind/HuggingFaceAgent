"""Evaluation: checks AlfredGalaAgent's answers against verifiable ground
truth, instead of trusting the few hand-picked examples in the README.

Same pattern as langgraph_course/first_graph/eval_spam_classification.py:
labeled cases, each checked against the real agent's output via a simple
substring/regex check -- not an LLM-judges-LLM setup, deliberately, so the
grading itself is deterministic even though the agent's answers aren't.

https://huggingface.co/learn/agents-course/en/unit3/agentic-rag/agent
"""

import re
import sys
from dataclasses import dataclass
from typing import Callable

sys.stdout.reconfigure(encoding="utf-8")

from huggingface_agent.unit3_agentic_rag.invitees.app import AlfredGalaAgent


@dataclass
class EvalCase:
    name: str
    question: str
    check: Callable[[str], bool]
    check_description: str


# Ground truth for guest cases comes straight from invitees.json (the saved
# dataset); ground truth for hub_stats cases was confirmed via a direct,
# independent huggingface_hub.list_models() call before writing this file
# (not assumed from the course text) -- see README.md for both.
EVAL_CASES: list[EvalCase] = [
    EvalCase(
        "guest_ada_lovelace",
        "Tell me about our guest named Ada Lovelace.",
        lambda answer: "ada.lovelace@example.com" in answer.lower(),
        "mentions Ada Lovelace's real email",
    ),
    EvalCase(
        "guest_nikola_tesla",
        "Tell me about our guest Dr. Nikola Tesla.",
        lambda answer: "nikola.tesla@gmail.com" in answer.lower(),
        "mentions Dr. Nikola Tesla's real email",
    ),
    EvalCase(
        "guest_marie_curie",
        "Tell me about our guest Marie Curie.",
        lambda answer: "marie.curie@example.com" in answer.lower(),
        "mentions Marie Curie's real email",
    ),
    EvalCase(
        "hub_stats_facebook",
        "What's the most downloaded model by facebook?",
        lambda answer: "facebook/contriever" in answer,
        "names the real top model (facebook/contriever)",
    ),
    EvalCase(
        "hub_stats_google",
        "What's the most downloaded model by google?",
        lambda answer: "google/vit-base-patch16-224" in answer,
        "names the real top model (google/vit-base-patch16-224)",
    ),
    EvalCase(
        "weather_tool_invoked",
        "What's the weather like in Paris right now?",
        lambda answer: bool(re.search(r"(Rainy|Clear|Windy)", answer))
        and bool(re.search(r"\d+\s*°?C", answer)),
        "actually used weather_info (condition + temperature present), not just guessed",
    ),
]


def run_eval(verbose: bool = True) -> float:
    """Runs every eval case through the real agent (fresh state each time)
    and returns overall pass rate (0.0-1.0)."""
    correct = 0

    for case in EVAL_CASES:
        alfred = AlfredGalaAgent()  # fresh agent per case -- no cross-case memory leakage
        answer = alfred.run(case.question)
        passed = case.check(answer)
        correct += int(passed)

        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"[{status}] {case.name}: expected {case.check_description}")
            if not passed:
                print(f"       answer was: {answer[:300]!r}")

    accuracy = correct / len(EVAL_CASES)
    if verbose:
        print(f"\nPass rate: {correct}/{len(EVAL_CASES)} ({accuracy:.0%})")
    return accuracy


if __name__ == "__main__":
    run_eval()
