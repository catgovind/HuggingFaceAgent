"""Multi-agent system: a manager CodeAgent delegating web research to a
managed web_agent, instead of one agent holding every tool.

https://huggingface.co/learn/agents-course/en/unit2/smolagents/multi_agent_systems
"""

import sys

# Windows' default console codepage (cp1252) can't print non-ASCII characters
# that real web search results often contain (curly quotes, em dashes, ...),
# which crashes web_agent's delegated calls with UnicodeEncodeError. Force
# UTF-8 on stdout so that output prints correctly instead.
sys.stdout.reconfigure(encoding="utf-8")

from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel, VisitWebpageTool

from huggingface_agent.smolagents_course.code_agents.party_planner import (
    SuperheroThemeTool,
    cityCapital,
    suggest_menu,
)


class PartyPlannerManagerAgent:
    """A manager agent that delegates web research to a managed web_agent.

    The manager keeps the party-planning tools (suggest_menu, cityCapital,
    SuperheroThemeTool) for itself, and hands off anything requiring real web
    browsing to web_agent -- a separate CodeAgent with its own tools and its
    own memory. The manager never sees web_agent's intermediate search/visit
    steps, only the summary it returns -- this is what keeps the manager's
    own context small as tasks grow.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
        max_steps: int = 8,
        verbosity_level: int = 2,
    ) -> None:
        model = InferenceClientModel(model_id=model_id)

        self.web_agent = CodeAgent(
            model=model,
            tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
            name="web_agent",
            description=(
                "Browses the web to research real party ideas, venues, playlists, or other "
                "information. Give it a specific research question; it returns a summary."
            ),
            max_steps=6,
            verbosity_level=1,
        )

        self.agent = CodeAgent(
            model=model,
            tools=[suggest_menu, cityCapital, SuperheroThemeTool()],
            managed_agents=[self.web_agent],
            max_steps=max_steps,
            verbosity_level=verbosity_level,
            # web_agent runs its own multi-step loop when the manager delegates
            # to it, which can easily exceed CodeAgent's default 30s per-step
            # timeout. Raise it so a delegated call has room to finish.
            executor_kwargs={"timeout_seconds": 90},
        )

    def run(self, task: str) -> str:
        """Run a single task through the manager agent and return its final answer."""
        return self.agent.run(task)


if __name__ == "__main__":
    planner = PartyPlannerManagerAgent()
    print(
        planner.run(
            "Plan a villain masquerade party: pick a theme and menu using your own tools, "
            "then have web_agent research one real song that fits a Gotham villain ball, "
            "and combine everything into one final plan."
        )
    )
