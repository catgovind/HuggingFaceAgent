"""GAIA-specific tool: fetching a task's attached file so the agent's own
generated code (pandas/pypdf/PIL, authorized in app.py) can read it.

https://huggingface.co/learn/agents-course/en/unit4/hands-on
"""

from pathlib import Path

from smolagents import Tool

from huggingface_agent.unit4_final_benchmark.gaia_client import GaiaClient

DOWNLOAD_DIR = Path(__file__).resolve().parent / "downloads"


class DownloadTaskFileTool(Tool):
    """Downloads the file attached to a GAIA task_id and returns its local
    path, so the agent can then open it with ordinary Python (pandas for
    .xlsx/.csv, pypdf for .pdf, PIL for images, open() for .py/.txt)."""

    name = "download_task_file"
    description = (
        "Downloads the file attached to a GAIA task (by task_id and its "
        "file_name, e.g. 'abc123.xlsx') and returns the local file path. "
        "Only call this when the question mentions an attached file."
    )
    inputs = {
        "task_id": {"type": "string", "description": "The GAIA task_id for this question."},
        "file_name": {
            "type": "string",
            "description": "The attachment's file_name, exactly as given with the question.",
        },
    }
    output_type = "string"

    def __init__(self, client: GaiaClient | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.client = client or GaiaClient()

    def forward(self, task_id: str, file_name: str) -> str:
        try:
            path = self.client.download_file(task_id, file_name, DOWNLOAD_DIR)
            return str(path)
        except Exception as e:
            return f"Error downloading {file_name} for task {task_id}: {e!s}"
