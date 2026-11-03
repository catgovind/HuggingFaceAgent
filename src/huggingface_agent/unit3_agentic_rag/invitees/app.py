"""Alfred, the gala agent -- integrates the guest retriever tool into a
real smolagents CodeAgent.

https://huggingface.co/learn/agents-course/en/unit3/agentic-rag/invitees
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

from smolagents import CodeAgent, InferenceClientModel

from huggingface_agent.unit3_agentic_rag.invitees.retriever import build_guest_retriever
from huggingface_agent.unit3_agentic_rag.invitees.tools import GuestInfoRetrieverTool

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"


class AlfredGalaAgent:
    """Alfred: a CodeAgent that can look up real gala guest details on demand."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        guest_tool = GuestInfoRetrieverTool(build_guest_retriever())
        # Provider auto-resolution fails for this model right now -- same
        # root cause hit in langgraph_course/first_graph and
        # smolagents_course/vision_and_browser_agents: check model_info()'s
        # inferenceProviderMapping, its only live provider is
        # featherless-ai, which isn't enabled by default on this account.
        self.model = InferenceClientModel(model_id=model_id, provider="featherless-ai")
        self.agent = CodeAgent(tools=[guest_tool], model=self.model)

    def run(self, question: str) -> str:
        return str(self.agent.run(question))


if __name__ == "__main__":
    alfred = AlfredGalaAgent()
    print("Alfred's Response:")
    print(alfred.run("Tell me about our guest named 'Lady Ada Lovelace'."))
