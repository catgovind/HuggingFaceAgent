"""Alfred, the gala agent -- final assembly: all four tools (invitees +
tools pages), add_base_tools for free web_search/visit_webpage/
final_answer, planning_interval, and conversation memory via reset=False.

https://huggingface.co/learn/agents-course/en/unit3/agentic-rag/invitees
https://huggingface.co/learn/agents-course/en/unit3/agentic-rag/tools
https://huggingface.co/learn/agents-course/en/unit3/agentic-rag/agent
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

from smolagents import CodeAgent, InferenceClientModel

from huggingface_agent.unit3_agentic_rag.invitees.retriever import build_guest_retriever
from huggingface_agent.unit3_agentic_rag.invitees.tools import (
    GuestInfoRetrieverTool,
    HubStatsTool,
    WeatherInfoTool,
)

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"


class AlfredGalaAgent:
    """Alfred: a CodeAgent that can look up real gala guest details, search
    the web, check (dummy) weather, and look up real Hugging Face Hub stats
    -- and remember earlier turns within one conversation.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        guest_tool = GuestInfoRetrieverTool(build_guest_retriever())
        # Provider auto-resolution fails for this model right now -- same
        # root cause hit in langgraph_course/first_graph and
        # smolagents_course/vision_and_browser_agents: check model_info()'s
        # inferenceProviderMapping, its only live provider is
        # featherless-ai, which isn't enabled by default on this account.
        self.model = InferenceClientModel(model_id=model_id, provider="featherless-ai")
        self.agent = CodeAgent(
            # add_base_tools=True already provides web_search, visit_webpage,
            # and final_answer -- an explicit DuckDuckGoSearchTool() would
            # just be silently overridden by it anyway (same tool name,
            # confirmed: the dict of tools is keyed by name).
            tools=[guest_tool, WeatherInfoTool(), HubStatsTool()],
            model=self.model,
            add_base_tools=True,
            planning_interval=3,
        )

    def run(self, question: str, reset: bool = True) -> str:
        """reset=False keeps the previous conversation's context -- lets
        Alfred answer follow-up questions without re-stating everything.
        """
        return str(self.agent.run(question, reset=reset))


if __name__ == "__main__":
    alfred = AlfredGalaAgent()

    print("=== Guest lookup ===")
    print(alfred.run("Tell me about our guest named 'Lady Ada Lovelace'."))

    print("\n=== Follow-up, same conversation (reset=False) ===")
    print(alfred.run("What projects is she currently working on?", reset=False))

    print("\n=== Multi-tool: prep for a conversation with a real guest ===")
    print(
        alfred.run(
            "I need to speak with Dr. Nikola Tesla about his wireless energy work. "
            "Tell me about him, then search the web for a recent development in "
            "wireless power transmission I could mention."
        )
    )
