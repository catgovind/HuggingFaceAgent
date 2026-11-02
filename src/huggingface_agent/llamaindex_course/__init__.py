"""Package structure mirroring HF Agents Course Unit 2 — LlamaIndex.

Source: https://huggingface.co/learn/agents-course/en/unit2/llama-index/introduction

LlamaIndex is a toolkit for building LLM-powered agents over your own data,
built around three core concepts: Components, Agents (+ Tools), and Workflows.
The course builds "Alfred," a gala-hosting agent, progressively across these
topics rather than as a separate capstone page.

Submodules:
    llama_hub   - the registry components/tools/integrations are installed from
    components  - basic building blocks (prompts, models, databases, indexes)
    tools       - capabilities an agent can call (search, calculate, query engines)
    agents      - autonomous components that use tools to pursue a goal
    workflows   - explicit, event-driven, async-first step sequences
"""
