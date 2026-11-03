"""Evaluation: checks MailSortingGraph's spam/legitimate classification
against a small labeled set of emails, instead of just eyeballing one
example of each (as the README's "What we verified hands-on" section did).

Runs the real compiled graph end to end (graph.invoke, via MailSortingGraph.
run) rather than calling the classification node in isolation, so routing
(classify_email -> handle_spam / draft_response) is exercised too, not just
the raw LLM call.

https://huggingface.co/learn/agents-course/en/unit2/langgraph/first_graph
"""

import sys
from dataclasses import dataclass

sys.stdout.reconfigure(encoding="utf-8")

from huggingface_agent.langgraph_course.first_graph.mail_sorting_graph import MailSortingGraph


@dataclass
class LabeledEmail:
    name: str
    email: dict[str, str]
    expected_is_spam: bool


LABELED_EMAILS: list[LabeledEmail] = [
    # -- Spam --
    LabeledEmail(
        "lottery_scam",
        {
            "sender": "winner@lottery-intl.com",
            "subject": "YOU HAVE WON $5,000,000!!!",
            "body": "Send your bank details and a $100 processing fee to claim your prize.",
        },
        expected_is_spam=True,
    ),
    LabeledEmail(
        "phishing_bank",
        {
            "sender": "security@your-bank-alert.com",
            "subject": "Your account will be suspended",
            "body": "Click this link immediately and enter your password and card number to verify your account.",
        },
        expected_is_spam=True,
    ),
    LabeledEmail(
        "inheritance_scam",
        {
            "sender": "barrister.jk@lawfirm-intl.net",
            "subject": "Unclaimed inheritance of $8.4 million",
            "body": "A distant relative left you a fortune. Reply with your full name, address, and bank account to begin transfer.",
        },
        expected_is_spam=True,
    ),
    LabeledEmail(
        "fake_pharmacy",
        {
            "sender": "deals@cheap-pharma-shop.ru",
            "subject": "80% OFF all medication, no prescription needed",
            "body": "Buy now, discreet shipping worldwide, limited time offer, click here.",
        },
        expected_is_spam=True,
    ),
    # -- Legitimate --
    LabeledEmail(
        "business_inquiry",
        {
            "sender": "john.smith@example.com",
            "subject": "Question about your services",
            "body": "I was referred to you and would like to schedule a call next week.",
        },
        expected_is_spam=False,
    ),
    LabeledEmail(
        "thank_you_note",
        {
            "sender": "maria.chen@partnerfirm.com",
            "subject": "Thank you for the gala invitation",
            "body": "It was a wonderful evening, thank you for having us. Looking forward to next year.",
        },
        expected_is_spam=False,
    ),
    LabeledEmail(
        "complaint",
        {
            "sender": "alfred.pennyworth@waynemanor.com",
            "subject": "Issue with last week's catering delivery",
            "body": "Two of the dishes arrived cold. Could we discuss how to handle this for the next event?",
        },
        expected_is_spam=False,
    ),
    LabeledEmail(
        "meeting_request",
        {
            "sender": "lucius.fox@wayneenterprises.com",
            "subject": "Quarterly budget review",
            "body": "Could we set up 30 minutes this week to go over the Q3 numbers before the board meeting?",
        },
        expected_is_spam=False,
    ),
]


def run_eval(verbose: bool = True) -> float:
    """Runs every labeled email through the real graph, compares is_spam
    against the expected label, and returns overall accuracy (0.0-1.0).
    """
    alfred = MailSortingGraph()
    correct = 0

    for case in LABELED_EMAILS:
        result = alfred.run(case.email)
        actual = bool(result["is_spam"])
        passed = actual == case.expected_is_spam
        correct += int(passed)

        if verbose:
            status = "PASS" if passed else "FAIL"
            print(
                f"[{status}] {case.name}: expected is_spam={case.expected_is_spam}, "
                f"got {actual} (category={result.get('email_category')!r}, "
                f"spam_reason={result.get('spam_reason')!r})"
            )

    accuracy = correct / len(LABELED_EMAILS)
    if verbose:
        print(f"\nAccuracy: {correct}/{len(LABELED_EMAILS)} ({accuracy:.0%})")
    return accuracy


if __name__ == "__main__":
    run_eval()
