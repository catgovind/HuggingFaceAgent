"""Components: the five-stage RAG pipeline (Loading, Indexing, Storing,
Querying, Evaluation) behind LlamaIndex's QueryEngine, built over a small
local knowledge base about Wayne Manor parties.

https://huggingface.co/learn/agents-course/en/unit2/llama-index/components
"""

import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent / "data"
DB_DIR = Path(__file__).resolve().parent / "chroma_db"


class PartyQueryEngine:
    """Wraps the course's five RAG stages into one reusable object.

    1. Loading   -- SimpleDirectoryReader reads data/*.txt
    2. Indexing  -- SentenceSplitter chunks text, HuggingFaceEmbedding embeds it
    3. Storing   -- ChromaVectorStore persists the vectors to disk
    4. Querying  -- VectorStoreIndex.as_query_engine() answers questions
    5. Evaluation -- FaithfulnessEvaluator checks an answer against its sources

    Note: re-running this against an existing chroma_db/ re-ingests the same
    documents into the same collection -- fine for learning, but would need
    deduplication/versioning for a real persistent index.
    """

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        db_dir: Path = DB_DIR,
        embed_model_name: str = "BAAI/bge-small-en-v1.5",
        llm_model_name: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
    ) -> None:
        # 1. Loading
        documents = SimpleDirectoryReader(input_dir=str(data_dir)).load_data()

        # 3. Storing (set up the vector store first, indexing writes into it)
        db = chromadb.PersistentClient(path=str(db_dir))
        collection = db.get_or_create_collection("party_planning")
        vector_store = ChromaVectorStore(chroma_collection=collection)

        # 2. Indexing: chunk + embed, writing into the vector store above
        embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=512, chunk_overlap=20),
                embed_model,
            ],
            vector_store=vector_store,
        )
        pipeline.run(documents=documents)

        self.index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
        self.llm = HuggingFaceInferenceAPI(model_name=llm_model_name, token=os.getenv("HF_TOKEN"))

    def query(self, question: str, response_mode: str = "tree_summarize") -> str:
        """4. Querying -- answer a question using the indexed documents."""
        query_engine = self.index.as_query_engine(llm=self.llm, response_mode=response_mode)
        return str(query_engine.query(question))

    def query_and_evaluate(self, question: str, response_mode: str = "tree_summarize") -> tuple[str, bool]:
        """4 + 5 -- answer a question, then check the answer is faithful to its sources."""
        query_engine = self.index.as_query_engine(llm=self.llm, response_mode=response_mode)
        response = query_engine.query(question)
        evaluator = FaithfulnessEvaluator(llm=self.llm)
        result = evaluator.evaluate_response(response=response)
        return str(response), result.passing


if __name__ == "__main__":
    engine = PartyQueryEngine()

    answer = engine.query("What should I serve at the villain masquerade party, given past feedback?")
    print("Answer:", answer)

    answer, passing = engine.query_and_evaluate("What wine pairs with the villain masquerade theme?")
    print("\nAnswer:", answer)
    print("Faithful to sources:", passing)
