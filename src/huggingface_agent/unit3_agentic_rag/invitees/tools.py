"""Alfred's tools: the BM25 guest retriever (invitees page), plus weather
info and Hub stats (tools page). Kept in one file, same as the course's
own project -- these pages build up the same tools.py, not separate ones.

https://huggingface.co/learn/agents-course/en/unit3/agentic-rag/invitees
https://huggingface.co/learn/agents-course/en/unit3/agentic-rag/tools
"""

import random

from huggingface_hub import list_models
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


class WeatherInfoTool(Tool):
    """Dummy weather lookup -- the course uses hard-coded data here for
    simplicity, deferring real weather (e.g. OpenWeatherMap, as in Unit 1)
    as an exercise.
    """

    name = "weather_info"
    description = "Fetches dummy weather information for a given location."
    inputs = {
        "location": {
            "type": "string",
            "description": "The location to get weather information for.",
        }
    }
    output_type = "string"

    def forward(self, location: str) -> str:
        weather_conditions = [
            {"condition": "Rainy", "temp_c": 15},
            {"condition": "Clear", "temp_c": 25},
            {"condition": "Windy", "temp_c": 20},
        ]
        data = random.choice(weather_conditions)
        return f"The weather in {location} is {data['condition']} with a temperature of {data['temp_c']}°C."


class HubStatsTool(Tool):
    """Looks up a Hugging Face author's most-downloaded model -- a real
    API call, not dummy data.
    """

    name = "hub_stats"
    description = "Fetches the most downloaded model from a specific author on the Hugging Face Hub."
    inputs = {
        "author": {
            "type": "string",
            "description": "The username of the model author/organization to find models from.",
        }
    }
    output_type = "string"

    def forward(self, author: str) -> str:
        try:
            # 'sort' alone already returns descending order in this
            # huggingface_hub version -- the course's direction=-1 kwarg
            # has been removed and raises TypeError if passed.
            models = list(list_models(author=author, sort="downloads", limit=1))
            if models:
                model = models[0]
                return f"The most downloaded model by {author} is {model.id} with {model.downloads:,} downloads."
            return f"No models found for author {author}."
        except Exception as e:
            return f"Error fetching models for {author}: {e!s}"
