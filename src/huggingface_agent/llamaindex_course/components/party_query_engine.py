"""Components: the five-stage RAG pipeline (Loading, Indexing, Storing,
Querying, Evaluation) behind LlamaIndex's QueryEngine.

Originally built over a small local knowledge base about Wayne Manor
parties (see data/); now points at data_tamil/, real Tamil-language chapter
text from Ponniyin Selvan (sourced from Project Madurai's Unicode etext,
since the PDF in docs/ uses a broken legacy Tamil font encoding that
produces garbled text on extraction). Uses a multilingual embedding model
instead of the original English-only one for this reason.

https://huggingface.co/learn/agents-course/en/unit2/llama-index/components
"""

import os
import sys
from pathlib import Path

# Tamil text will be printed; Windows' default console codepage can't
# encode it (same UnicodeEncodeError seen in multi_agent_systems).
sys.stdout.reconfigure(encoding="utf-8")

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

DATA_DIR = Path(__file__).resolve().parent / "data_tamil"
DB_DIR = Path(__file__).resolve().parent / "chroma_db"


class PartyQueryEngine:
    """Wraps the course's five RAG stages into one reusable object.

    1. Loading   -- SimpleDirectoryReader reads data_dir/*.txt
    2. Indexing  -- SentenceSplitter chunks text, HuggingFaceEmbedding embeds it
    3. Storing   -- ChromaVectorStore persists the vectors to disk
    4. Querying  -- VectorStoreIndex.as_query_engine() answers questions
    5. Evaluation -- FaithfulnessEvaluator checks an answer against its sources

    Ingestion (1+2) only runs once per collection -- __init__ checks
    collection.count() first and skips re-ingesting if it's already
    populated. Earlier versions of this class re-ingested on every
    construction, which silently duplicated every chunk on repeated runs
    (confirmed for real: 3,657 vectors accumulated from what should have
    been ~700-900) and corrupted retrieval quality as a result.
    """

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        db_dir: Path = DB_DIR,
        # BAAI/bge-small-en-v1.5 (the original default) is English-only and
        # produces poor embeddings for Tamil text -- intfloat/multilingual-e5-small
        # is trained across languages, confirmed by testing: a Tamil sentence and
        # its English translation scored 0.91 cosine similarity vs. 0.78 for an
        # unrelated English sentence.
        embed_model_name: str = "intfloat/multilingual-e5-small",
        llm_model_name: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
        collection_name: str = "story_of_chola",
    ) -> None:
        # 3. Storing (set up the vector store first, indexing writes into it)
        db = chromadb.PersistentClient(path=str(db_dir))
        collection = db.get_or_create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=collection)

        embed_model = HuggingFaceEmbedding(model_name=embed_model_name)

        # Only ingest once per collection. Without this check, every
        # PartyQueryEngine() construction re-ran 1. Loading + 2. Indexing
        # against the same collection, silently duplicating every chunk --
        # confirmed for real: 3 files that should produce ~700-900 chunks
        # had accumulated 3,657 vectors after repeated runs, which also
        # corrupted retrieval (duplicate chunks crowding out real matches).
        if collection.count() == 0:
            # 1. Loading
            documents = SimpleDirectoryReader(input_dir=str(data_dir)).load_data()
            # 2. Indexing: chunk + embed, writing into the vector store above
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

    # English question against Tamil source text -- exercises the
    # multilingual embedding model's cross-lingual retrieval directly.
    answer = engine.query("Who is Vandiyathevan and what is he doing at the lake?")
    print("Answer:", answer)
