"""Agents: LlamaIndex's AgentWorkflow -- basic tool-using agents, stateful
conversation via Context, RAG agents, and multi-agent handoff.

Uses ReActAgent (works with any LLM, not just function-calling ones) rather
than FunctionAgent, with streaming=False -- see the module-level note below
for why that second part is load-bearing, not a style choice.

https://huggingface.co/learn/agents-course/en/unit2/llama-index/agents
"""

import asyncio
import sys

sys.stdout.reconfigure(encoding="utf-8")

from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

from huggingface_agent.llamaindex_course.tools.party_tools import (
    build_menu_function_tool,
    build_novel_query_engine_tool,
)

# Real bug, confirmed by tracing it into both libraries' source: ReActAgent
# defaults to streaming=True, whose path (take_step -> _get_streaming_response
# -> llm.astream_chat) hits an async httpx client that's already closed by the
# time a later step reaches it ("RuntimeError: Cannot send a request, as the
# client has been closed."), reproduced with two different models, so it's
# not a function-calling-support issue. Isolated astream_chat() alone works
# fine -- the bug is specifically in how the agent workflow engine drives
# repeated streaming calls. streaming=False routes through _get_response
# (plain llm.achat()) instead, which has no such issue.
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"


def build_basic_agent() -> AgentWorkflow:
    """1. A basic tool-using agent -- one ReActAgent with one FunctionTool."""
    llm = HuggingFaceInferenceAPI(model_name=DEFAULT_MODEL_ID)
    menu_agent = ReActAgent(
        name="menu_planner",
        description="Suggests a party menu for a given occasion.",
        tools=[build_menu_function_tool()],
        llm=llm,
        streaming=False,
    )
    return AgentWorkflow(agents=[menu_agent], root_agent="menu_planner")


def build_rag_agent() -> AgentWorkflow:
    """3. RAG agent -- wraps the real PartyQueryEngine (components/) as a tool."""
    llm = HuggingFaceInferenceAPI(model_name=DEFAULT_MODEL_ID)
    novel_agent = ReActAgent(
        name="novel_lookup",
        description="Answers questions about the Ponniyin Selvan novel text.",
        system_prompt="Use your tool to answer questions about the Ponniyin Selvan novel.",
        tools=[build_novel_query_engine_tool()],
        llm=llm,
        streaming=False,
    )
    return AgentWorkflow(agents=[novel_agent], root_agent="novel_lookup")


def build_multi_agent_system() -> AgentWorkflow:
    """4. Multi-agent system -- two specialized agents, one root, with handoff.

    menu_planner handles party-menu questions directly; novel_lookup handles
    anything about the book. Each only needs its own narrow tool, and
    can_handoff_to lets either one route to the other instead of trying (and
    failing) to answer outside its specialty.
    """
    llm = HuggingFaceInferenceAPI(model_name=DEFAULT_MODEL_ID)

    menu_agent = ReActAgent(
        name="menu_planner",
        description="Suggests a party menu for a given occasion (casual, formal, superhero).",
        system_prompt=(
            "You suggest party menus using your tool. For anything about the "
            "Ponniyin Selvan novel, hand off to novel_lookup."
        ),
        tools=[build_menu_function_tool()],
        llm=llm,
        streaming=False,
        can_handoff_to=["novel_lookup"],
    )
    novel_agent = ReActAgent(
        name="novel_lookup",
        description="Answers questions about the Ponniyin Selvan novel text.",
        system_prompt=(
            "You answer questions about the Ponniyin Selvan novel using your tool. "
            "For anything about party menus, hand off to menu_planner."
        ),
        tools=[build_novel_query_engine_tool()],
        llm=llm,
        streaming=False,
        can_handoff_to=["menu_planner"],
    )
    return AgentWorkflow(agents=[menu_agent, novel_agent], root_agent="menu_planner")


async def demo_memory(agent: AgentWorkflow) -> None:
    """2. State/memory -- a Context ties multiple .run() calls into one conversation."""
    ctx = Context(agent)
    r1 = await agent.run("My favorite party occasion is superhero-themed.", ctx=ctx)
    print("Turn 1:", r1)
    r2 = await agent.run("Suggest a menu for my favorite occasion.", ctx=ctx)
    print("Turn 2:", r2)


async def main() -> None:
    print("=== 1. Basic agent ===")
    basic_agent = build_basic_agent()
    print(await basic_agent.run("Suggest a menu for a casual party."))

    print("\n=== 2. Memory via Context ===")
    await demo_memory(build_basic_agent())

    print("\n=== 3. RAG agent ===")
    rag_agent = build_rag_agent()
    print(await rag_agent.run("Who is Vandiyathevan?"))

    print("\n=== 4. Multi-agent system ===")
    multi_agent = build_multi_agent_system()
    print("Menu question:", await multi_agent.run("Suggest a formal party menu."))
    print("Novel question:", await multi_agent.run("Who is Kundavai?"))


if __name__ == "__main__":
    asyncio.run(main())
