# Agents

Course page: https://huggingface.co/learn/agents-course/en/unit2/llama-index/agents

LlamaIndex's `AgentWorkflow`: basic tool-using agents, stateful conversation via `Context`, RAG agents, and multi-agent systems with handoff.

## Files

- **`party_agents.py`** — `BasicPartyAgent` (also covers memory via `new_context()`), `NovelRAGAgent`, `MultiAgentPartyPlanner`. The RAG and multi-agent classes reuse the real tools from `tools/party_tools.py` rather than rebuilding anything.

## Run it

```
python -m huggingface_agent.llamaindex_course.agents.party_agents
```

## Multi-agent handoff

```mermaid
flowchart TD
    U["user message"] --> R["root_agent: menu_planner"]
    R -->|"party-menu question"| M["menu_planner\n(suggest_menu tool)"]
    R -->|"can_handoff_to"| N["novel_lookup\n(QueryEngineTool over\nPonniyin Selvan text)"]
    M -->|"can_handoff_to"| N
    N -->|"can_handoff_to"| M
    M --> Out(["final_answer"])
    N --> Out
```

Each agent only needs tools for its own specialty; `can_handoff_to` lets it route a question outside that specialty to the other agent instead of trying (and failing) to answer it directly.

## A real bug, found and fixed: `ReActAgent(..., streaming=False)`

`ReActAgent` defaults to `streaming=True`. Traced into both libraries' source to confirm: that path (`take_step` → `_get_streaming_response` → `llm.astream_chat`) reliably hit `RuntimeError: Cannot send a request, as the client has been closed.` — reproduced with two different models, so not a function-calling-support issue. `astream_chat()` called directly, in isolation, works fine — the bug is specifically in how the agent workflow engine drives *repeated* streaming calls across steps. `streaming=False` routes through `_get_response` (plain `llm.achat()`) instead, which has no such issue. All agents in `party_agents.py` set this explicitly.

## A second real bug, found and fixed: `num_output=1024`

The memory demo's first turn (a plain statement about a superhero-themed party) initially hit `WorkflowRuntimeError: Max iterations of 20 reached!` — reproduced on **three separate runs**, always at the identical spot, which ruled out ordinary LLM randomness as the cause. Streaming the raw `AgentOutput` events (not just the final result) showed exactly why: the model's answer — an elaborate multi-section menu — got **cut off mid-generation at the identical character position every time** ("...Captain America's Shield-Shaped"). `HuggingFaceInferenceAPI`'s default `num_output` (its `max_tokens` field) is only **256**, too small for a verbose answer to ever reach the literal `Answer:` marker `ReActOutputParser` requires to end the loop — so the model just kept retrying and regenerating the same truncated response until the iteration cap gave up. Fixed with `num_output=DEFAULT_NUM_OUTPUT` (1024) on every `HuggingFaceInferenceAPI` in this file. Re-verified with a full clean run after the fix: all four parts succeeded in one pass, no truncation, no retries.

## What we verified hands-on

- **Basic agent**: `suggest_menu` tool called correctly for "a casual party."
- **Memory**: a two-turn conversation via one `Context` — turn 2 ("suggest a menu for my favorite occasion") correctly used "superhero" remembered from turn 1, with no repeated statement needed.
- **RAG agent**: correct, grounded answers about Vandiyathevan and Kundavai from the real Tamil novel text (Kundavai's description — "daughter of Chembian Maadhevi... royal lineage" — matches the character's real role in the novel).
- **Multi-agent handoff**: a menu question and a novel question, sent to the same `AgentWorkflow`, each correctly routed to the agent actually equipped to answer it.
