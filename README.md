# HuggingFaceAgent

My hands-on repo for Hugging Face's [AI Agents course](https://huggingface.co/learn/agents-course), built with [smolagents](https://github.com/huggingface/smolagents). Each course unit becomes a runnable example, and I document what broke along the way — the bugs were the real lessons.

## What's inside

- `src/huggingface_agent/app.py` — CodeAgent with time-zone, image, and web-search tools, plus a Gradio UI (`Gradio_UI.py`)
- `src/huggingface_agent/smolagents_course/` — one subpackage per course topic, each with its own README and runnable example:
  - `code_agents` — PartyPlannerAgent (CodeAgent); home of the Delhi experiment
  - `tool_calling_agents` — ToolCallingAgent mirror of the same task, for a direct JSON-vs-code comparison
  - `retrieval_agents` — agentic RAG: the agent decides when to query a local BM25-indexed knowledge base
  - `multi_agent_systems` — manager agent delegating web research to a sub-agent via `managed_agents`
  - `vision_and_browser_agents` — vision agent reasoning over images
  - `tools` — custom tools shared across examples
- `docs/agent-flow.md` — Mermaid diagrams explaining how the agents work
- `tests/` — pytest suite

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

- [x] Unit 2.1 — CodeAgent basics, tools, party-planner walkthrough
- [x] Unit 2.2 — ToolCallingAgent vs CodeAgent
- [x] Unit 2.3 — Retrieval agents (agentic RAG with BM25)
- [x] Unit 2.4 — Multi-agent systems (manager + managed web researcher)
- [x] Unit 2.5 — Vision agents
- [ ] Unit 3 — ...
