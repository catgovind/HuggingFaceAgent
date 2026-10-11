"""ToolCallingAgent example -- same party-planning tools as code_agents.party_planner,
but driving them through JSON tool calls instead of executable Python code.

https://huggingface.co/learn/agents-course/en/unit2/smolagents/tool_calling_agents
"""

from smolagents import DuckDuckGoSearchTool, InferenceClientModel, ToolCallingAgent

from huggingface_agent.smolagents_course.code_agents.party_planner import (
    SuperheroThemeTool,
    cityCapital,
    suggest_menu,
)


class PartyPlannerToolCallingAgent:
    """Wraps a smolagents ToolCallingAgent using the same tools as PartyPlannerAgent.

    Same tools, same model, same .run(task) interface as
    code_agents.party_planner.PartyPlannerAgent -- the only difference is the
    agent class itself, so any behavior difference comes from JSON tool-calling
    vs. code generation, not from the tools.
    """

    def __init__(
        self,
        # Qwen2.5-Coder-32B-Instruct's HF Inference Providers backend rejects the
        # "tools"/"tool_choice" params ToolCallingAgent needs (verified: 422
        # UNSUPPORTED_OPENAI_PARAMS). Llama-3.3-70B-Instruct supports real
        # function calling there, so it's the default here instead.
        model_id: str = "meta-llama/Llama-3.3-70B-Instruct",
        max_steps: int = 10,
        verbosity_level: int = 2,
    ) -> None:
        self.model = InferenceClientModel(model_id=model_id)
        self.agent = ToolCallingAgent(
            tools=[DuckDuckGoSearchTool(), suggest_menu, cityCapital, SuperheroThemeTool()],
            model=self.model,
            max_steps=max_steps,
            verbosity_level=verbosity_level,
        )

    def run(self, task: str) -> str:
        """Run a single task through the agent and return its final answer."""
        return self.agent.run(task)


if __name__ == "__main__":
    planner = PartyPlannerToolCallingAgent()
    print(planner.run("Give me the best playlist for a villain masquerade party at Wayne Manor."))
