"""Loads the guest list and builds a BM25 retriever over it.

Reads from the local invitees.json by default (a saved copy of the real
agents-course/unit3-invitees dataset, downloaded and verified once -- see
its header comment) instead of hitting the Hub on every run. Pass
from_hub=True to re-fetch from the Hub directly instead.

https://huggingface.co/learn/agents-course/en/unit3/agentic-rag/invitees
"""

import json
from pathlib import Path

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

DATASET_NAME = "agents-course/unit3-invitees"
LOCAL_DATA_PATH = Path(__file__).resolve().parent / "invitees.json"


def _rows_from_local() -> list[dict]:
    return json.loads(LOCAL_DATA_PATH.read_text(encoding="utf-8"))


def _rows_from_hub(dataset_name: str, split: str) -> list[dict]:
    import datasets

    ds = datasets.load_dataset(dataset_name, split=split)
    return [dict(row) for row in ds]


def load_guest_docs(
    dataset_name: str = DATASET_NAME,
    split: str = "train",
    from_hub: bool = False,
) -> list[Document]:
    """Converts each guest row into one retrievable Document."""
    rows = _rows_from_hub(dataset_name, split) if from_hub else _rows_from_local()
    docs = []
    for row in rows:
        content = "\n".join(
            [
                f"Name: {row['name']}",
                f"Relation: {row['relation']}",
                f"Description: {row['description']}",
                f"Email: {row['email']}",
            ]
        )
        docs.append(Document(page_content=content, metadata={"name": row["name"]}))
    return docs


def build_guest_retriever(k: int = 3, from_hub: bool = False) -> BM25Retriever:
    """Builds a BM25 keyword retriever over the guest list."""
    docs = load_guest_docs(from_hub=from_hub)
    return BM25Retriever.from_documents(docs, k=k)
