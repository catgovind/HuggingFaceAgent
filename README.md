# HuggingFaceAgent

A lightweight Python project scaffold for building an agent that integrates with Hugging Face models and tools.

## Features

- Modern Python packaging via `pyproject.toml`
- Source layout under `src/`
- Simple agent class with a minimal interface
- Ready for testing and extension

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
python -m huggingface_agent "Hello from HuggingFaceAgent"
```

## Project structure

```text
HuggingFaceAgent/
├── README.md
├── pyproject.toml
├── .gitignore
├── src/
│   └── huggingface_agent/
│       ├── __init__.py
│       ├── __main__.py
│       └── agent.py
└── tests/
    └── test_agent.py
```
