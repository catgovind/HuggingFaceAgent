"""Package structure mirroring HF Agents Course Unit 2 — LangGraph.

Source: https://huggingface.co/learn/agents-course/en/unit2/langgraph/introduction

LangGraph gives explicit, graph-based control over an agent's flow --
nodes, edges, and shared state -- aimed at organized, production-ready
applications rather than a fully autonomous loop. The course's own
examples were run with GPT-4o; this project uses HF-hosted models instead,
the same substitution made throughout smolagents_course/ and
llamaindex_course/.

Submodules:
    when_to_use_langgraph    - when LangGraph is (and isn't) the right fit
    building_blocks           - core components: nodes, edges, state, graph
    first_graph                - "Alfred, the mail sorting butler" walkthrough
    document_analysis_agent    - "Alfred, the document analyst agent" walkthrough

Not mirrored: the unit's quiz (assessment, not code) and its Resources
section (external links to LangGraph's own examples and LangChain Academy).
"""
