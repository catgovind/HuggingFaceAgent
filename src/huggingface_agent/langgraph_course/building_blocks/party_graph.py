"""Building blocks of LangGraph: State, Nodes, Edges, StateGraph.

Two examples: MoodGraph is the course page's own example as-is (node_1/
node_2/node_3, decide_mood), wrapped in a class; PartyPlanningGraph is the
same pattern adapted to this project's party-planning theme.

https://huggingface.co/learn/agents-course/en/unit2/langgraph/building_blocks
"""

import random
from typing import Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


class MoodState(TypedDict):
    """The course page's own example state, exactly as written -- one
    string field, graph_state."""

    graph_state: str


def node_1(state: MoodState) -> dict:
    print("---Node 1---")
    return {"graph_state": state["graph_state"] + " I am"}


def node_2(state: MoodState) -> dict:
    print("---Node 2---")
    return {"graph_state": state["graph_state"] + " happy!"}


def node_3(state: MoodState) -> dict:
    print("---Node 3---")
    return {"graph_state": state["graph_state"] + " sad!"}


def decide_mood(state: MoodState) -> Literal["node_2", "node_3"]:
    """A random 50/50 split for demonstration -- in practice, inspect state."""
    if random.random() < 0.5:
        return "node_2"
    return "node_3"


class MoodGraph:
    """The course page's own example, wrapped in a class for consistency
    with this project's other examples. __init__ builds and compiles the
    graph exactly as the page shows it (add_node x3, add_edge(START,
    "node_1"), add_conditional_edges("node_1", decide_mood), add_edge to
    END from both branches); run() is the page's graph.invoke(...) call.
    """

    def __init__(self) -> None:
        builder = StateGraph(MoodState)

        builder.add_node("node_1", node_1)
        builder.add_node("node_2", node_2)
        builder.add_node("node_3", node_3)

        builder.add_edge(START, "node_1")
        builder.add_conditional_edges("node_1", decide_mood)
        builder.add_edge("node_2", END)
        builder.add_edge("node_3", END)

        self.graph = builder.compile()

    def run(self, initial_state: str = "Hi, this is Lance.") -> str:
        result = self.graph.invoke({"graph_state": initial_state})
        return result["graph_state"]


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
    print("=== MoodGraph (course page's own example) ===")
    mood = MoodGraph()
    print("RESULT:", mood.run())

    print("\n=== PartyPlanningGraph (adapted) ===")
    planner = PartyPlanningGraph()
    print("FINAL PLAN:", planner.run("Alfred's notes:"))

    # Not called automatically: draw_mermaid_png() sends the graph's
    # structure to the public mermaid.ink API to render the image. Call
    # planner.visualize() yourself if you're fine with that external call.
