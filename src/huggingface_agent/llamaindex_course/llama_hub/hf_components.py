"""LlamaHub: installing and using individual LlamaIndex integrations.

LlamaHub components follow one naming pattern for both the pip package and
the import path:

    pip install llama-index-{component-type}-{framework-name}
    from llama_index.{component_type}.{framework_name} import ...

e.g. `llama-index-llms-huggingface-api` -> `llama_index.llms.huggingface_api`.

https://huggingface.co/learn/agents-course/en/unit2/llama-index/llama-hub
"""

import os

from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

load_dotenv()


class HuggingFaceInferenceAPIExample:
    """The course page's own LLM example, wrapped in a class.

    __init__ builds the llm exactly as the page shows it; complete() is
    just the page's `response = llm.complete(prompt)` line as a method.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-32B-Instruct") -> None:
        hf_token = os.getenv("HF_TOKEN")
        self.llm = HuggingFaceInferenceAPI(
            model_name=model_name,
            temperature=0.7,
            max_tokens=100,
            token=hf_token,
            provider="auto",
        )

    def complete(self, prompt: str) -> str:
        response = self.llm.complete(prompt)
        return str(response)


class HuggingFaceEmbeddingExample:
    """Not from the page -- it installs this package but never shows usage code.

    Included anyway, verified working, since the page leaves it untested.
    Runs locally (model downloaded once, cached) -- no HF_TOKEN, no API call.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.embed_model = HuggingFaceEmbedding(model_name=model_name)

    def embed(self, text: str) -> list[float]:
        return self.embed_model.get_text_embedding(text)


if __name__ == "__main__":
    print("LLM completion:")
    llm_example = HuggingFaceInferenceAPIExample()
    print(llm_example.complete("Hello, how are you?"))

    print("\nEmbedding vector:")
    embedding_example = HuggingFaceEmbeddingExample()
    vector = embedding_example.embed("Hello, world!")
    print(f"length={len(vector)}, first 5 values={vector[:5]}")
