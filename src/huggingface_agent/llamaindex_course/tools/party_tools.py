"""Tools: wrapping a Python function and an existing QueryEngine as
LlamaIndex Tools an agent can call.

Scoped to FunctionTool and QueryEngineTool only -- the course page's other
two tool types (ToolSpecs like GmailToolSpec, and MCP tools) need external
credentials or a running server not available here, same reasoning as the
browser-automation half of vision_and_browser_agents being out of scope.

https://huggingface.co/learn/agents-course/en/unit2/llama-index/tools
"""

import sys

# Tamil text may be printed via the wrapped query engine; Windows' default
# console codepage can't encode it (same fix as components/party_query_engine.py).
sys.stdout.reconfigure(encoding="utf-8")

from llama_index.core.tools import FunctionTool, QueryEngineTool

from huggingface_agent.llamaindex_course.components.party_query_engine import PartyQueryEngine


def suggest_menu(occasion: str) -> str:
    """Suggests a party menu for a given occasion.

    Args:
        occasion: One of "casual", "formal", "superhero", or anything else
            (returns a generic suggestion).
    """
    if occasion == "casual":
        return "Pizza, snacks, and drinks."
    if occasion == "formal":
        return "3-course dinner with wine and dessert."
    if occasion == "superhero":
        return "Buffet with high-energy and healthy food."
    return "Custom menu for the butler."


def build_menu_function_tool() -> FunctionTool:
    """1. FunctionTool -- wraps a plain Python function so an agent can call it.

    Compare against the smolagents @tool decorator (code_agents/party_planner.py)
    doing the same job for the same kind of function, in a different framework.
    """
    return FunctionTool.from_defaults(
        suggest_menu,
        name="suggest_menu",
        description="Suggests a party menu for a given occasion (casual, formal, or superhero).",
    )


def build_novel_query_engine_tool(engine: PartyQueryEngine | None = None) -> QueryEngineTool:
    """2. QueryEngineTool -- wraps an existing QueryEngine as a tool.

    Reuses the real PartyQueryEngine from components/ (Tamil novel text,
    multilingual embeddings) instead of building a separate one from scratch
    -- this is literally the course's own point: a query engine from the
    previous unit becomes a tool in this one, no new indexing needed.
    """
    engine = engine or PartyQueryEngine()
    query_engine = engine.index.as_query_engine(llm=engine.llm)
    return QueryEngineTool.from_defaults(
        query_engine,
        name="ponniyin_selvan_lookup",
        description=(
            "Answers questions about the Ponniyin Selvan novel (Tamil source text, "
            "can answer in English or Tamil)."
        ),
    )


if __name__ == "__main__":
    menu_tool = build_menu_function_tool()
    print("FunctionTool direct call:", menu_tool.call(occasion="superhero"))

    novel_tool = build_novel_query_engine_tool()
    print("\nQueryEngineTool direct call:", novel_tool.call("Who is Vandiyathevan?"))
