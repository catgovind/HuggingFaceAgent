"""First graph: "Alfred, the mail sorting butler" -- the course's first
full graph with real LLM nodes: read an email, classify it as spam or
legitimate, draft a reply for legitimate mail, and notify the house.

Uses ChatHuggingFace/HuggingFaceEndpoint instead of the course's ChatOpenAI
(no OpenAI key needed, consistent with every other example in this
project). Classification asks for JSON output rather than the course's
keyword search over free text ("spam" / "reason:") -- that approach breaks
on a legitimate email that happens to contain the word "spam" (e.g. "please
stop the spam calls"); JSON with a fallback on parse failure doesn't.

The course notifies "Mr. Hugg" in its code despite describing "Mr. Wayne"
in the surrounding text -- this version uses Mr. Wayne throughout, which
is also the canonically correct name for Alfred's employer, matching the
Wayne Manor theme already used across this project.

https://huggingface.co/learn/agents-course/en/unit2/langgraph/first_graph
"""

import json
import os
import sys
from typing import Any, Optional, TypedDict

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import END, START, StateGraph

load_dotenv()

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"


class EmailState(TypedDict):
    email: dict[str, Any]  # {"sender", "subject", "body"}
    is_spam: Optional[bool]
    spam_reason: Optional[str]
    email_category: Optional[str]
    email_draft: Optional[str]
    messages: list[dict[str, Any]]


def _build_llm(model_id: str) -> ChatHuggingFace:
    # HuggingFaceEndpoint's provider auto-resolution doesn't find this model
    # unless the provider is named explicitly -- confirmed via model_info():
    # Qwen2.5-Coder-32B-Instruct's only live inferenceProviderMapping right
    # now is 'featherless-ai', which isn't enabled by default on this
    # account (same root cause hit earlier with a vision model). Without
    # this, the call fails with "not supported by any provider you have
    # enabled" even though the model is genuinely live somewhere.
    endpoint = HuggingFaceEndpoint(
        repo_id=model_id,
        provider="featherless-ai",
        huggingfacehub_api_token=os.getenv("HF_TOKEN"),
        temperature=0.01,
    )
    return ChatHuggingFace(llm=endpoint)


def _parse_json_reply(text: str) -> dict:
    """Strips markdown code fences the model sometimes wraps JSON in, then
    parses it -- falling back to a safe default rather than crashing if the
    model didn't produce valid JSON.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"is_spam": False, "spam_reason": None, "category": "information"}


class MailSortingGraph:
    """Wraps Alfred's email-sorting workflow: read -> classify -> (spam:
    stop here) or (legitimate: draft a reply -> notify Mr. Wayne).
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        self.llm = _build_llm(model_id)

        builder = StateGraph(EmailState)
        builder.add_node("read_email", self._read_email)
        builder.add_node("classify_email", self._classify_email)
        builder.add_node("handle_spam", self._handle_spam)
        builder.add_node("draft_response", self._draft_response)
        builder.add_node("notify_mr_wayne", self._notify_mr_wayne)

        builder.add_edge(START, "read_email")
        builder.add_edge("read_email", "classify_email")
        builder.add_conditional_edges(
            "classify_email",
            self._route_email,
            {"spam": "handle_spam", "legitimate": "draft_response"},
        )
        builder.add_edge("handle_spam", END)
        builder.add_edge("draft_response", "notify_mr_wayne")
        builder.add_edge("notify_mr_wayne", END)

        self.graph = builder.compile()

    def _read_email(self, state: EmailState) -> dict:
        e = state["email"]
        print(f"Alfred is processing an email from {e['sender']} with subject: {e['subject']}")
        return {}

    def _classify_email(self, state: EmailState) -> dict:
        e = state["email"]
        prompt = (
            "As Alfred the butler, analyze this email and determine if it is spam or legitimate.\n"
            "Reply ONLY with a JSON object with these keys:\n"
            '- "is_spam": true or false\n'
            '- "spam_reason": short string if spam, otherwise null\n'
            '- "category": one of "inquiry", "complaint", "thank_you", "request", "information" '
            "if legitimate, otherwise null\n\n"
            f"From: {e['sender']}\nSubject: {e['subject']}\nBody: {e['body']}"
        )
        response = self.llm.invoke([HumanMessage(content=prompt)])
        data = _parse_json_reply(response.content)
        return {
            "is_spam": bool(data.get("is_spam")),
            "spam_reason": data.get("spam_reason"),
            "email_category": data.get("category"),
            "messages": state.get("messages", [])
            + [{"role": "user", "content": prompt}, {"role": "assistant", "content": response.content}],
        }

    def _handle_spam(self, state: EmailState) -> dict:
        print(f"Alfred marked the email as spam. Reason: {state.get('spam_reason')}")
        return {}

    def _draft_response(self, state: EmailState) -> dict:
        e = state["email"]
        prompt = (
            "As Alfred the butler, draft a brief, polite preliminary reply to this email. "
            "Mr. Wayne will review and personalize it before sending.\n"
            f"Category: {state.get('email_category') or 'general'}\n"
            f"From: {e['sender']}\nSubject: {e['subject']}\nBody: {e['body']}"
        )
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return {
            "email_draft": response.content,
            "messages": state.get("messages", [])
            + [{"role": "user", "content": prompt}, {"role": "assistant", "content": response.content}],
        }

    def _notify_mr_wayne(self, state: EmailState) -> dict:
        e = state["email"]
        print("\n" + "=" * 50)
        print(f"Sir, you've received an email from {e['sender']}.")
        print(f"Subject: {e['subject']}  |  Category: {state.get('email_category')}")
        print("-" * 50)
        print(state.get("email_draft"))
        print("=" * 50 + "\n")
        return {}

    def _route_email(self, state: EmailState) -> str:
        return "spam" if state["is_spam"] else "legitimate"

    def run(self, email: dict[str, str]) -> EmailState:
        initial: EmailState = {
            "email": email,
            "is_spam": None,
            "spam_reason": None,
            "email_category": None,
            "email_draft": None,
            "messages": [],
        }
        return self.graph.invoke(initial)


if __name__ == "__main__":
    alfred = MailSortingGraph()

    legit_email = {
        "sender": "john.smith@example.com",
        "subject": "Question about your services",
        "body": "I was referred to you and would like to schedule a call next week.",
    }
    spam_email = {
        "sender": "winner@lottery-intl.com",
        "subject": "YOU HAVE WON $5,000,000!!!",
        "body": "Send your bank details and a $100 processing fee to claim your prize.",
    }

    print("--- Legitimate email ---")
    alfred.run(legit_email)

    print("--- Spam email ---")
    alfred.run(spam_email)
