from pathlib import Path

from dotenv import load_dotenv
from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel, load_tool, tool
import datetime
import requests
import pytz
from tools.final_answer import FinalAnswerTool

from Gradio_UI import GradioUI

load_dotenv()  # reads HF_TOKEN from the project's .env file

# Below is an example of a tool that does nothing. Amaze us with your creativity!
@tool
def my_custom_tool(arg1:str, arg2:int)-> str: # it's important to specify the return type
    # Keep this format for the tool description / args description but feel free to modify the tool
    """A tool that does nothing yet 
    Args:
        arg1: the first argument
        arg2: the second argument
    """
    return "What magic will you build ?"

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


final_answer = FinalAnswerTool()
model = InferenceClientModel(
    max_tokens=2096,
    temperature=0.5,
    model_id='Qwen/Qwen2.5-Coder-32B-Instruct',
    custom_role_conversions=None,
)


# Import tool from Hub
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

# Save every generated image to <project>/outputs and report the real path in the tool logs.
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
_generate_image = image_generation_tool.forward

def _generate_and_save(*args, **kwargs):
    image = _generate_image(*args, **kwargs)
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / f"image-{datetime.datetime.now():%Y%m%d-%H%M%S}.png"
    image.save(path)
    print(f"Image saved to: {path}")
    return image

image_generation_tool.forward = _generate_and_save

# No custom prompt_templates: the course's prompts.yaml targets smolagents 1.13, so we use the
# installed version's built-in prompts, which match its code parser.
agent = CodeAgent(
    model=model,
    tools=[final_answer, get_current_time_in_timezone, my_custom_tool, image_generation_tool, DuckDuckGoSearchTool()], # add your tools here (don't remove final_answer)
    max_steps=6,
    verbosity_level=1,
    planning_interval=None,
    name=None,
    description=None,
    instructions=(
        "When asked to draw or generate an image, call image_generator(prompt=...) and pass the "
        "returned image object directly to final_answer(image). Never invent file paths or URLs."
    ),
)


GradioUI(agent).launch()