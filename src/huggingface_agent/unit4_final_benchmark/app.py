"""GaiaAgent: the Unit 4 final-benchmark agent -- answers a subset (20
questions, level 1) of the GAIA benchmark, graded by exact string match.

Confirmed live before building against it: 17/20 questions are text-only
(Wikipedia-style lookups, logic puzzles, trivia), one has a .png (chess
position), one a .py (run it, report the output), two are .mp3 (voicemail-
style questions). Two more ask about a YouTube video's content, which this
agent cannot watch -- left as a known gap rather than faked.

https://huggingface.co/learn/agents-course/en/unit4/hands-on
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

from smolagents import CodeAgent, InferenceClientModel, SpeechToTextTool, WikipediaSearchTool

from huggingface_agent.unit4_final_benchmark.tools import DownloadTaskFileTool

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"

# Grading is an EXACT MATCH against ground truth -- prompting matters as
# much as tool choice. The course explicitly says not to include the text
# "FINAL ANSWER" in the submitted answer itself.
ANSWER_FORMAT_INSTRUCTIONS = (
    "You are answering questions from the GAIA benchmark. Grading is an "
    "EXACT STRING MATCH against a ground-truth answer, so formatting "
    "matters as much as correctness.\n"
    "- Call final_answer with ONLY the bare answer: no restating the "
    "question, no explanation, no leading/trailing words like 'Answer:' "
    "or 'FINAL ANSWER'.\n"
    "- If the question asks for a number, return digits only, no commas, "
    "no units unless explicitly requested.\n"
    "- If it asks for a name, return just the name, matching the exact "
    "capitalization/spelling likely used by the source (e.g. a Wikipedia "
    "article title).\n"
    "- If it asks for a comma-separated list, use ', ' between items and "
    "no trailing period.\n"
    "- If the question references an attached file, call "
    "download_task_file(task_id, file_name) first, then read the "
    "downloaded path with ordinary Python (pandas for .csv/.xlsx, pypdf "
    "for .pdf, PIL for images, open() for .py/.txt) -- do not guess file "
    "contents.\n"
    "- Prefer wikipedia_search for encyclopedic facts; use web_search only "
    "when wikipedia_search doesn't cover it.\n"
    "- When extracting a fact from retrieved text (a search result, a "
    "visited page, a file), read it yourself and reason about it in your "
    "Thought -- do NOT write brittle string-parsing code (line.split(), "
    "regex on markdown, isdigit() filters) to extract it. That code "
    "usually looks right but silently matches zero or the wrong thing on "
    "real-world formatting it wasn't tested against. Only write parsing "
    "code for data you can verify is regularly structured (e.g. an actual "
    "CSV/JSON/Excel file), never for prose or markdown you scraped.\n"
    "- If you genuinely cannot answer (e.g. a YouTube video you cannot "
    "watch), still call final_answer with your best guess rather than "
    "refusing -- a guess can accidentally match, a refusal never does."
)


class GaiaAgent:
    """A CodeAgent tuned for GAIA's exact-match grading: file-aware,
    Wikipedia-first, and explicit about answer formatting.

    No planning_interval -- removed deliberately. The Unit 3 capstone
    (unit3_agentic_rag/invitees) found that planning_interval's periodic
    re-plan step ignores `instructions` and can make the agent abandon a
    reliable tool for open-ended research; the same risk applies here, so
    it's left out from the start instead of being found the hard way again.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        # Same recurring fix as every other unit in this project: this
        # model's only live inference provider is featherless-ai.
        #
        # flatten_messages_as_text=True: hit a run of 400s here that looked
        # like featherless-ai deterministically rejecting the list-of-parts
        # message content InferenceClientModel sends by default. Retesting
        # the identical raw call later (no code change) succeeded 3/3 --
        # it was transient backend flakiness during that session, not a
        # real format incompatibility (see README.md for the full
        # retraction). Left this flag on anyway since it's a harmless
        # no-op that sends plain-string content instead, in case the
        # flakiness recurs.
        self.model = InferenceClientModel(
            model_id=model_id, provider="featherless-ai", flatten_messages_as_text=True
        )
        self.agent = CodeAgent(
            tools=[DownloadTaskFileTool(), WikipediaSearchTool(), SpeechToTextTool()],
            model=self.model,
            add_base_tools=True,  # web_search, visit_webpage, final_answer
            additional_authorized_imports=["pandas", "pypdf", "openpyxl", "PIL", "json", "csv"],
            max_steps=15,
            instructions=ANSWER_FORMAT_INSTRUCTIONS,
        )

    def __call__(self, question: str, task_id: str = "", file_name: str = "") -> str:
        prompt = question
        if file_name:
            prompt = (
                f"{question}\n\n"
                f"(This question has an attached file: task_id={task_id!r}, "
                f"file_name={file_name!r}. Call download_task_file first.)"
            )
        return str(self.agent.run(prompt))


if __name__ == "__main__":
    agent = GaiaAgent()
    print(agent("How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)?"))
