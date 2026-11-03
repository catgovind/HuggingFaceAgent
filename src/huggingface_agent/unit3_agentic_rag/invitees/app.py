"""Alfred, the gala agent -- integrates all four tools built so far into
a real smolagents CodeAgent: guest lookup (invitees page), web search,
dummy weather, and real Hub stats (tools page).

https://huggingface.co/learn/agents-course/en/unit3/agentic-rag/invitees
https://huggingface.co/learn/agents-course/en/unit3/agentic-rag/tools
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel

from huggingface_agent.unit3_agentic_rag.invitees.retriever import build_guest_retriever
from huggingface_agent.unit3_agentic_rag.invitees.tools import (
    GuestInfoRetrieverTool,
    HubStatsTool,
    WeatherInfoTool,
)

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"


class AlfredGalaAgent:
    """Alfred: a CodeAgent that can look up real gala guest details, search
    the web, check (dummy) weather, and look up real Hugging Face Hub stats.
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
            tools=[guest_tool, DuckDuckGoSearchTool(), WeatherInfoTool(), HubStatsTool()],
            model=self.model,
        )

    def run(self, question: str) -> str:
        return str(self.agent.run(question))


if __name__ == "__main__":
    alfred = AlfredGalaAgent()

    print("=== Guest lookup ===")
    print(alfred.run("Tell me about our guest named 'Lady Ada Lovelace'."))

    print("\n=== Hub stats (real API call) ===")
    print(alfred.run("What's the most downloaded model by facebook?"))
