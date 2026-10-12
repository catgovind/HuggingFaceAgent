"""Vision agent: a CodeAgent that reasons over an image, not just text.

Scoped to the VLM half of this course page only -- no live browser
automation (no Chrome/selenium/helium, no OpenAI key available). The agent
is given a generated party-invitation image and asked to describe its theme
and suggest a matching menu, combining visual understanding with the
existing suggest_menu tool.

https://huggingface.co/learn/agents-course/en/unit2/smolagents/vision_agents
"""

from pathlib import Path

from PIL import Image, ImageDraw
from smolagents import CodeAgent, InferenceClientModel

from huggingface_agent.smolagents_course.code_agents.party_planner import suggest_menu

OUTPUT_DIR = Path(__file__).resolve().parents[4] / "outputs"

# meta-llama/Llama-4-Scout-17B-16E-Instruct is one of the few models confirmed
# live across multiple HF Inference Providers (novita, nscale, deepinfra) that
# accepts image inputs -- several other vision model IDs returned
# "not supported by any provider you have enabled" or were at capacity.
VISION_MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"


def generate_invite_image() -> Image.Image:
    """Draws a simple masquerade-themed invitation card (no external image needed)."""
    img = Image.new("RGB", (400, 300), color=(20, 20, 40))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 399, 299], outline=(212, 175, 55), width=8)
    draw.ellipse([60, 100, 180, 180], fill=(80, 0, 0))
    draw.ellipse([220, 100, 340, 180], fill=(80, 0, 0))
    draw.polygon([(200, 90), (170, 140), (230, 140)], fill=(212, 175, 55))
    draw.text((100, 220), "GOTHAM ROGUES' BALL", fill=(212, 175, 55))
    return img


class PartyInviteVisionAgent:
    """CodeAgent that reads a party-invitation image and acts on what it sees."""

    def __init__(
        self,
        model_id: str = VISION_MODEL_ID,
        max_steps: int = 6,
        verbosity_level: int = 2,
    ) -> None:
        self.model = InferenceClientModel(model_id=model_id)
        self.agent = CodeAgent(
            tools=[suggest_menu],
            model=self.model,
            max_steps=max_steps,
            verbosity_level=verbosity_level,
        )

    def run(self, task: str, image: Image.Image) -> str:
        """Run a task through the agent, with one image passed in as visual context."""
        return self.agent.run(task, images=[image])


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    invite = generate_invite_image()
    invite_path = OUTPUT_DIR / "party-invite.png"
    invite.save(invite_path)
    print(f"Invitation image saved to: {invite_path}")

    agent = PartyInviteVisionAgent()
    print(
        agent.run(
            "This image is a party invitation. Describe its visual theme, then use "
            "your suggest_menu tool to pick a matching menu for that theme.",
            invite,
        )
    )
