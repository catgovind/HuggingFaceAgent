"""Agentic RAG: a CodeAgent backed by a BM25 retriever tool over a small
local knowledge base, instead of (or alongside) live web search.

https://huggingface.co/learn/agents-course/en/unit2/smolagents/retrieval_agents
"""

from langchain_community.docstore.document import Document
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from smolagents import CodeAgent, InferenceClientModel, Tool

# A tiny local knowledge base. In a real use case this would come from files,
# a database, or a larger document store.
PARTY_IDEAS = [
    {"text": "Villain masquerade theme: gold accents, dark lighting, a Gotham Rogues' Ball vibe.", "source": "themes"},
    {"text": "Hire a DJ who can mix orchestral villain scores with modern synth tracks.", "source": "entertainment"},
    {"text": "Serve superhero-named dishes: Avenger Bites, Batman Brownies, Kryptonite Punch.", "source": "catering"},
    {"text": "Decorate with Gotham City skyline silhouettes and bat-signal projections.", "source": "decor"},
    {"text": "For kids' parties, keep the menu casual: pizza, snacks, and colorful drinks.", "source": "catering"},
]


class PartyPlanningRetrieverTool(Tool):
    """Retrieves relevant party-planning ideas from a local knowledge base via BM25."""

    name = "party_planning_retriever"
    description = (
        "Uses BM25 keyword search to retrieve party-planning ideas (themes, catering, "
        "decor, entertainment) relevant to a query from a local knowledge base."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The party-planning question or topic to search for.",
        }
    }
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        self.retriever = BM25Retriever.from_documents(docs, k=3)

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Query must be a string."
        docs = self.retriever.invoke(query)
        return "\nRetrieved ideas:\n" + "".join(
            f"\n\n===== Idea {i} ({doc.metadata.get('source', 'unknown')}) =====\n{doc.page_content}"
            for i, doc in enumerate(docs)
        )


def build_knowledge_base() -> list[Document]:
    """Turns PARTY_IDEAS into chunked LangChain Documents ready for retrieval."""
    source_docs = [Document(page_content=item["text"], metadata={"source": item["source"]}) for item in PARTY_IDEAS]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, add_start_index=True)
    return splitter.split_documents(source_docs)


class PartyPlannerRetrievalAgent:
    """CodeAgent whose only tool is local-knowledge-base retrieval (agentic RAG)."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
        max_steps: int = 6,
        verbosity_level: int = 2,
    ) -> None:
        self.model = InferenceClientModel(model_id=model_id)
        self.retriever_tool = PartyPlanningRetrieverTool(build_knowledge_base())
        self.agent = CodeAgent(
            tools=[self.retriever_tool],
            model=self.model,
            max_steps=max_steps,
            verbosity_level=verbosity_level,
        )

    def run(self, task: str) -> str:
        """Run a single task through the agent and return its final answer."""
        return self.agent.run(task)


if __name__ == "__main__":
    planner = PartyPlannerRetrievalAgent()
    print(planner.run("Find superhero party ideas, including a theme and a menu."))
