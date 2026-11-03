"""Document analysis agent: "Alfred, the document analyst agent" -- a real
ReAct-style tool-using agent, not a fixed pipeline like first_graph's
MailSortingGraph. assistant <-> tools loops via tools_condition until the
model stops calling tools and gives a final answer.

Uses ChatHuggingFace/HuggingFaceEndpoint with meta-llama/Llama-4-Scout-17B-
16E-Instruct instead of the course's ChatOpenAI(model="gpt-4o") -- one
model serving both roles (vision + tool-calling assistant), confirmed to
support both, same as the course's single gpt-4o. No OpenAI key needed,
consistent with every other example in this project.

https://huggingface.co/learn/agents-course/en/unit2/langgraph/document_analysis_agent
"""

import base64
import os
import sys
from pathlib import Path
from typing import Annotated, Optional, TypedDict

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from PIL import Image, ImageDraw

load_dotenv()

DEFAULT_MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"


class AgentState(TypedDict):
    input_file: Optional[str]
    messages: Annotated[list[AnyMessage], add_messages]


def generate_training_note_image(output_path: str = "training_note.png") -> str:
    """Draws a simple note with real, readable text (no external image
    needed) -- something for extract_text to genuinely OCR via the vision
    model, matching the course's "note from Mr. Wayne" scenario.
    """
    img = Image.new("RGB", (500, 300), color=(245, 245, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 499, 299], outline=(20, 20, 40), width=4)
    lines = [
        "Training & Meal Plan - W. Manor",
        "",
        "Morning: 5k run, 100 pushups",
        "Lunch: grilled chicken, quinoa, spinach",
        "Dinner shopping list:",
        "- salmon fillets",
        "- asparagus",
        "- sweet potatoes",
        "- dark chocolate (70%)",
    ]
    y = 20
    for line in lines:
        draw.text((20, y), line, fill=(20, 20, 40))
        y += 28
    img.save(output_path)
    return output_path


def _build_llm(model_id: str) -> ChatHuggingFace:
    endpoint = HuggingFaceEndpoint(repo_id=model_id, huggingfacehub_api_token=os.getenv("HF_TOKEN"), temperature=0.01)
    return ChatHuggingFace(llm=endpoint)


class DocumentAnalystGraph:
    """Alfred: a tool-using agent that can read text out of a document
    image and do simple arithmetic, looping between the assistant and
    tools nodes until it has enough to answer.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        self.llm = _build_llm(model_id)

        def extract_text(img_path: str) -> str:
            """Extract text from an image file using a multimodal model.

            Master Wayne often leaves notes with his training regimen or meal
            plans. This allows me to properly analyze the contents.
            """
            try:
                image_bytes = Path(img_path).read_bytes()
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                message = [
                    HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": "Extract all the text from this image. Return only the extracted text, no explanations.",
                            },
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                        ]
                    )
                ]
                response = self.llm.invoke(message)
                return response.content.strip()
            except Exception as e:
                print(f"Error extracting text: {e}")
                return ""

        def divide(a: int, b: int) -> float:
            """Divide a and b -- for Master Wayne's occasional calculations."""
            return a / b

        self.tools = [divide, extract_text]
        # The course passes parallel_tool_calls=False here -- ChatOpenAI
        # forwards it to OpenAI's API, which supports it. ChatHuggingFace
        # forwards it just as blindly, straight to huggingface_hub's
        # InferenceClient.chat_completion(), which doesn't accept it at all:
        # TypeError: chat_completion() got an unexpected keyword argument
        # 'parallel_tool_calls'. Omitted here; confirmed the model still
        # issues one tool call per step in practice regardless.
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        builder = StateGraph(AgentState)
        builder.add_node("assistant", self._assistant)
        builder.add_node("tools", ToolNode(self.tools))
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges("assistant", tools_condition)
        builder.add_edge("tools", "assistant")
        self.graph = builder.compile()

    def _assistant(self, state: AgentState) -> dict:
        image = state["input_file"]
        sys_msg = SystemMessage(
            content=(
                "You are a helpful butler named Alfred that serves Mr. Wayne and Batman. "
                "You can analyse documents and run computations with your tools.\n"
                f"You have access to an optional image. Currently the loaded image is: {image}"
            )
        )
        return {
            "messages": [self.llm_with_tools.invoke([sys_msg] + state["messages"])],
            "input_file": state["input_file"],
        }

    def ask(self, question: str, input_file: Optional[str] = None) -> str:
        """Runs one question through the full assistant<->tools loop and
        returns the final answer text."""
        result = self.graph.invoke({"messages": [HumanMessage(content=question)], "input_file": input_file})
        return result["messages"][-1].content


if __name__ == "__main__":
    alfred = DocumentAnalystGraph()

    print("=== Calculation (no file) ===")
    print(alfred.ask("Divide 6790 by 5"))

    print("\n=== Document question ===")
    note_path = generate_training_note_image()
    print(
        alfred.ask(
            "According to the note provided by Mr. Wayne in the provided image, "
            "what's the list of items I should buy for the dinner menu?",
            input_file=note_path,
        )
    )
