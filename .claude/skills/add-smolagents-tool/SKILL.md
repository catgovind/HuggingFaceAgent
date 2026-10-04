---
name: add-smolagents-tool
description: Use when adding a new tool/capability to the smolagents CodeAgent in this project (src/huggingface_agent/app.py) — e.g. "add a tool that does X", "give the agent the ability to Y", "create a new smolagents tool". Covers both the quick @tool decorator pattern and the Tool subclass pattern, and how to register the result with the agent.
---

# Add a smolagents tool to this project

This project's agent lives in [src/huggingface_agent/app.py](../../../src/huggingface_agent/app.py).
It is a `smolagents.CodeAgent` whose `tools=[...]` list is the agent's full capability set.
Reusable, standalone tools live as separate files in
[src/huggingface_agent/tools/](../../../src/huggingface_agent/tools/).

## Step 1 — pick a pattern

**`@tool` decorated function** — default choice. Use for a stateless tool with no
setup/config. Define it directly in `app.py`, following `get_current_time_in_timezone`
as the template:

```python
@tool
def my_new_tool(arg1: str, arg2: int) -> str:
    """One-line description of what the tool does.

    Args:
        arg1: what this argument means
        arg2: what this argument means
    """
    ...
    return result
```

Rules that matter to smolagents (not just style):
- The docstring's `Args:` section is sent to the model as the tool's input spec — every
  parameter must be documented there or the agent can't call it correctly.
- Always annotate the return type.
- Catch exceptions you expect and return a string explaining the failure instead of
  raising, so the agent can see the error and retry (see
  `get_current_time_in_timezone`'s try/except).

**`Tool` subclass** — use when the tool needs `__init__` state/config (an API client, a
cache, constructor args like `max_results`). Put it in its own file under `tools/`,
following [tools/web_search.py](../../../src/huggingface_agent/tools/web_search.py) or
[tools/visit_webpage.py](../../../src/huggingface_agent/tools/visit_webpage.py):

```python
from smolagents.tools import Tool

class MyNewTool(Tool):
    name = "my_new_tool"
    description = "What this tool does, written for the model to decide when to call it."
    inputs = {
        "arg1": {"type": "string", "description": "what this argument means"},
    }
    output_type = "string"

    def forward(self, arg1: str) -> str:
        ...
        return result
```

`name`, `description`, `inputs`, and `output_type` are all required class attributes —
the agent reads them to decide if/when/how to call the tool.

## Step 2 — register it with the agent

In `app.py`:
- `@tool` functions: just define them above `agent = CodeAgent(...)` and add the
  function to the `tools=[...]` list.
- `Tool` subclasses: import the class, instantiate it (e.g.
  `my_new_tool = MyNewTool()`), and add that instance to `tools=[...]`.

Never remove `final_answer` from the list — the agent needs it to terminate.

## Step 3 — sanity check

- If the tool needs a secret/API key, load it via `.env` + `load_dotenv()` the same way
  `HF_TOKEN` is loaded at the top of `app.py` — don't hardcode credentials.
- If the tool needs a new PyPI package, add it to `pyproject.toml` dependencies.
- Run `python -m huggingface_agent` (or launch `app.py` directly) and ask the agent a
  prompt that should trigger the new tool, to confirm it's actually being selected and
  called.
