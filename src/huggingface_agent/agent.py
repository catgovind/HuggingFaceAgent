class HuggingFaceAgent:
    """Minimal agent scaffold for Hugging Face integrations."""

    def __init__(self, name: str = "HuggingFaceAgent") -> None:
        self.name = name

    def run(self, prompt: str) -> str:
        """Return a simple response for a prompt."""
        return f"{self.name} received: {prompt}"

    def chat(self, prompt: str) -> str:
        """Backward-compatible alias for `run`."""
        return self.run(prompt)
