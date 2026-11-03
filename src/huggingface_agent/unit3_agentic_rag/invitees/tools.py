"""smolagents Tool wrapping the BM25 guest retriever.

https://huggingface.co/learn/agents-course/en/unit3/agentic-rag/invitees
"""

from langchain_community.retrievers import BM25Retriever
from smolagents import Tool


class GuestInfoRetrieverTool(Tool):
    """Retrieves gala guest info (relation, background, email) by name or relation."""

    name = "guest_info_retriever"
    description = (
        "Retrieves detailed information about gala guests, including relation, "
        "background, and email, based on a guest's name or relation to the host."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The name or relation of the guest to look up.",
        }
    }
    output_type = "string"

    def __init__(self, retriever: BM25Retriever, **kwargs) -> None:
        super().__init__(**kwargs)
        self.retriever = retriever

    def forward(self, query: str) -> str:
        results = self.retriever.invoke(query)
        if not results:
            return "No matching guest information found."
        return "\n\n".join(doc.page_content for doc in results)
