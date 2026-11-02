"""Building blocks of LangGraph: State, Nodes, Edges, StateGraph.

https://huggingface.co/learn/agents-course/en/unit2/langgraph/building_blocks
"""

import random
from typing import Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


class PartyState(TypedDict):
    """State: a plain TypedDict. Every node receives this and returns a
    partial update dict that LangGraph merges back in -- nodes never mutate
    state directly or see each other's local variables, only what's here.
    """

    plan: str


def pick_theme(state: PartyState) -> dict:
    """A node: takes state, returns a partial update."""
    print("---pick_theme---")
    return {"plan": state["plan"] + " Theme chosen: villain masquerade."}


def plan_formal(state: PartyState) -> dict:
    print("---plan_formal---")
    return {"plan": state["plan"] + " Formal 3-course dinner planned."}


def plan_casual(state: PartyState) -> dict:
    print("---plan_casual---")
    return {"plan": state["plan"] + " Casual buffet planned."}


def decide_formality(state: PartyState) -> Literal["plan_formal", "plan_casual"]:
    """A conditional edge's routing function: inspects state (or, here,
    rolls dice the way the course's own example does) and returns the name
    of whichever node should run next. The Literal return type documents
    the only two valid destinations.
    """
    return "plan_formal" if random.random() < 0.5 else "plan_casual"


class PartyPlanningGraph:
    """Wraps the course's State -> Nodes -> Edges -> StateGraph pattern into
    one reusable object: build the graph once in __init__, run() it as many
    times as needed.
    """

    def __init__(self) -> None:
        builder = StateGraph(PartyState)

        builder.add_node("pick_theme", pick_theme)
        builder.add_node("plan_formal", plan_formal)
        builder.add_node("plan_casual", plan_casual)

        builder.add_edge(START, "pick_theme")
        builder.add_conditional_edges("pick_theme", decide_formality)
        builder.add_edge("plan_formal", END)
        builder.add_edge("plan_casual", END)

        self.graph = builder.compile()

    def run(self, initial_plan: str = "") -> str:
        result = self.graph.invoke({"plan": initial_plan})
        return result["plan"]

    def visualize(self, output_path: str = "party_graph.png") -> None:
        """Renders the compiled graph as a PNG (calls the mermaid.ink API)."""
        png_bytes = self.graph.get_graph().draw_mermaid_png()
        with open(output_path, "wb") as f:
            f.write(png_bytes)
        print(f"Graph diagram written to: {output_path}")


if __name__ == "__main__":
    planner = PartyPlanningGraph()
    print("FINAL PLAN:", planner.run("Alfred's notes:"))

    # Not called automatically: draw_mermaid_png() sends the graph's
    # structure to the public mermaid.ink API to render the image. Call
    # planner.visualize() yourself if you're fine with that external call.
