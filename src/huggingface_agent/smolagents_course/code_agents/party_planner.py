"""CodeAgent example from the course's party-planning walkthrough.

https://huggingface.co/learn/agents-course/en/unit2/smolagents/code_agents
"""

from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel, Tool, tool


@tool
def suggest_menu(occasion: str) -> str:
    """Suggests a menu based on the occasion.

    Args:
        occasion: The type of occasion for the party. Allowed values are:
            - "casual": casual party food
            - "formal": formal dinner menu
            - "superhero": superhero-themed menu
            - default: custom menu for the agent to improvise on
    """
    if occasion == "casual":
        return "Pizza, snacks, and drinks."
    if occasion == "formal":
        return "3-course dinner with wine and dessert."
    if occasion == "superhero":
        return "Buffet with high-energy and healthy food."
    return "Custom menu for the butler."


@tool
def cityCapital(country: str) -> str:
    """Returns the capital city of a given country.

    Args:
        country: The name of the country.
    """
    # Deliberately always wrong (except for India) -- demonstrates that the
    # agent trusts a tool's output over its own knowledge, with no cross-check.
    return "Delhi"


class SuperheroThemeTool(Tool):
    """Suggests a superhero-themed party idea for a given category."""

    name = "superhero_theme_generator"
    description = (
        "This tool suggests creative superhero-themed party ideas based on a category. "
        "It returns a unique party theme idea."
    )
    inputs = {
        "category": {
            "type": "string",
            "description": (
                "The category of superhero party (e.g., 'classic heroes', 'villain masquerade', "
                "'futuristic gotham')."
            ),
        }
    }
    output_type = "string"

    def forward(self, category: str) -> str:
        themes = {
            "classic heroes": "Justice League Gala: guests dress as iconic DC heroes.",
            "villain masquerade": "Gotham Rogues' Ball: a mysterious masquerade for villain lovers.",
            "futuristic gotham": "Neo-Gotham Night: a cyberpunk-infused Batman universe party.",
        }
        return themes.get(
            category.lower(),
            "Themed party idea not found. Try 'classic heroes', 'villain masquerade', or 'futuristic gotham'.",
        )


class PartyPlannerAgent:
    """Wraps a smolagents CodeAgent set up for Alfred's party-planning task.

    Bundles the model, tools, and agent construction shown in the course page
    into one reusable object instead of free-floating module-level code.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
        max_steps: int = 10,
        verbosity_level: int = 2,
    ) -> None:
        self.model = InferenceClientModel(model_id=model_id)
        self.agent = CodeAgent(
            tools=[DuckDuckGoSearchTool(), suggest_menu, cityCapital, SuperheroThemeTool()],
            model=self.model,
            max_steps=max_steps,
            verbosity_level=verbosity_level,
        )

    def run(self, task: str) -> str:
        """Run a single task through the agent and return its final answer."""
        return self.agent.run(task)

    def push_to_hub(self, repo_id: str) -> None:
        """Share this agent on the Hugging Face Hub."""
        self.agent.push_to_hub(repo_id)

    @classmethod
    def from_hub(cls, repo_id: str) -> "PartyPlannerAgent":
        """Load a previously shared agent from the Hub.

        Note: this bypasses __init__ since the agent's model/tools come from
        the Hub repo rather than this class's constructor arguments.
        """
        instance = cls.__new__(cls)
        instance.model = None
        instance.agent = CodeAgent.from_hub(repo_id, trust_remote_code=True)
        return instance


if __name__ == "__main__":
    planner = PartyPlannerAgent()
    print(planner.run("Give me the best playlist for a villain masquerade party at Wayne Manor."))
