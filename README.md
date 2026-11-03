# HuggingFaceAgent

My hands-on repo for Hugging Face's [AI Agents course](https://huggingface.co/learn/agents-course), built with [smolagents](https://github.com/huggingface/smolagents). Each course unit becomes a runnable example, and I document what broke along the way — the bugs were the real lessons.

## What's inside

- `src/huggingface_agent/app.py` — CodeAgent with time-zone, image, and web-search tools, plus a Gradio UI (`Gradio_UI.py`)
- `src/huggingface_agent/smolagents_course/` — Unit 2, smolagents track, one subpackage per topic (each with its own README):
  - `code_agents` — PartyPlannerAgent (CodeAgent); home of the Delhi experiment
  - `tool_calling_agents` — ToolCallingAgent mirror of the same task, for a direct JSON-vs-code comparison
  - `retrieval_agents` — agentic RAG: the agent decides when to query a local BM25-indexed knowledge base
  - `multi_agent_systems` — manager agent delegating web research to a sub-agent via `managed_agents`
  - `vision_and_browser_agents` — vision agent reasoning over images
  - `tools` — custom tools shared across examples
- `src/huggingface_agent/llamaindex_course/` — Unit 2, LlamaIndex track:
  - `llama_hub` — loading Hub models/components into LlamaIndex
  - `components` — QueryEngine/RAG pipeline over a local document set
  - `tools` — FunctionTool/QueryEngineTool wrapping
  - `agents` — ReActAgent and AgentWorkflow
  - `workflows` — raw `Workflow`/`@step`/`Event` framework, stateful via `Context`
- `src/huggingface_agent/langgraph_course/` — Unit 2, LangGraph track:
  - `when_to_use_langgraph` — notes only, no runnable example
  - `building_blocks` — StateGraph fundamentals (nodes, edges, conditional edges)
  - `first_graph` — ReAct-style assistant↔tools loop (`tools_condition`/`ToolNode`), with `eval_spam_classification.py`
  - `document_analysis_agent` — multi-step document analyst graph
- `src/huggingface_agent/unit3_agentic_rag/invitees/` — Unit 3 capstone: Alfred the gala agent (smolagents), with BM25 guest retrieval, web search, weather, and real Hub-stats tools, plus `eval_alfred.py` scoring it against verifiable ground truth
- `docs/agent-flow.md` — Mermaid diagrams explaining how the agents work
- `tests/` — pytest suite (thin — most verification in this repo is hands-on runs and the per-unit `eval_*.py` scripts, not unit tests)

## Things I found while building

- **Agents trust tool output over their own knowledge, with no cross-check.** Demonstrated with a deliberately wrong `cityCapital` tool that always returns "Delhi" — the agent reports it confidently. (`code_agents/party_planner.py`)
- **A managed agent's entire sub-run counts as one of the manager's execution steps**, which can blow past CodeAgent's default 30s timeout. Fix: `executor_kwargs={"timeout_seconds": 90}`.
- **Windows' default console codepage can't print non-ASCII**, which crashed delegated web-search calls with `UnicodeEncodeError`. Fix: `sys.stdout.reconfigure(encoding="utf-8")`.
- **Model compatibility is real, not theoretical.** Qwen2.5-Coder-32B's inference backend rejects the `tools`/`tool_choice` params `ToolCallingAgent` needs; Llama-3.2-11B-Vision and Qwen2.5-VL-7B both failed before Llama-4-Scout-17B worked for the vision example.
- **You don't need Langfuse to observe an agent.** `tracing.py` adds local OpenTelemetry console tracing for CodeAgent runs.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
cp .env.example .env             # add your HF_TOKEN
python -m huggingface_agent "plan a superhero-themed birthday party"
```

`.env` is git-ignored — never commit your token.

## Course progress

- [x] Unit 2 — smolagents: CodeAgent/ToolCallingAgent, retrieval agents, multi-agent systems, vision agents
- [x] Unit 2 — LlamaIndex: components, tools, agents, workflows
- [x] Unit 2 — LangGraph: building blocks, first graph (+ eval), document analysis agent
- [x] Unit 3 — Agentic RAG capstone: Alfred the gala agent (+ eval, see `unit3_agentic_rag/invitees/README.md` for the real `planning_interval` bug found along the way)
- [~] Unit 4 — Final benchmark (GAIA subset): agent + API client built and verified against the real 20-question set (see `unit4_final_benchmark/README.md` for two real bugs found: a `featherless-ai` 400 and a 404 server-side file-download gap) — not yet submitted via a Hugging Face Space
