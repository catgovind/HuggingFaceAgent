"""Loads the real guest-list dataset and builds a BM25 retriever over it.

Dataset: agents-course/unit3-invitees (3 guests, fields: name, relation,
description, email) -- confirmed real, not a placeholder, by loading it
directly before writing anything against it.

https://huggingface.co/learn/agents-course/en/unit3/agentic-rag/invitees
"""

import datasets
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

DATASET_NAME = "agents-course/unit3-invitees"


def load_guest_docs(dataset_name: str = DATASET_NAME, split: str = "train") -> list[Document]:
    """Converts each guest row into one retrievable Document."""
    ds = datasets.load_dataset(dataset_name, split=split)
    docs = []
    for row in ds:
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


def build_guest_retriever(k: int = 3) -> BM25Retriever:
    """Builds a BM25 keyword retriever over the guest list."""
    docs = load_guest_docs()
    return BM25Retriever.from_documents(docs, k=k)
