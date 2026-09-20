import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient



load_dotenv()  # reads HF_TOKEN from the project's .env file

## You need a token from https://hf.co/settings/tokens, ensure that you select 'read' as the token type.
## If you run this on Google Colab, add it in the "Secrets" tab (key icon on the left sidebar) and call it "HF_TOKEN".




try:
    from google.colab import userdata
    HF_TOKEN = userdata.get("HF_TOKEN")
except ImportError:
    HF_TOKEN = os.environ.get("HF_TOKEN")

client = InferenceClient(model="moonshotai/Kimi-K2.5", token=HF_TOKEN)

# This system prompt is a bit more complex and actually contains the function description already appended.
# Here we suppose that the textual description of the tools has already been appended.

SYSTEM_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

get_weather: Get the current weather in a given location

The way you use the tools is by specifying a json blob.
Specifically, this json should have an `action` key (with the name of the tool to use) and an `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are:
get_weather: Get the current weather in a given location, args: {"location": {"type": "string"}}
example use :

{{
  "action": "get_weather",
  "action_input": {"location": "New York"}
}}


ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about one action to take. Only one action at a time in this format:
Action:

$JSON_BLOB (inside markdown cell)

Observation: the result of the action. This Observation is unique, complete, and the source of truth.
(this Thought/Action/Observation can repeat N times, you should take several steps when needed. The $JSON_BLOB must be formatted as markdown and only use a SINGLE action at a time.)

You must always end your output with the following format:

Thought: I now know the final answer
Final Answer: the final answer to the original input question

Now begin! Reminder to ALWAYS use the exact characters `Final Answer:` when you provide a definitive answer. """

output = client.chat.completions.create(
    messages=[
         {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What's the weather in London?"},
    ],
    stream=False,
    max_tokens=1024,
    extra_body={'thinking': {'type': 'disabled'}},
)
print(output.choices[0].message.content)