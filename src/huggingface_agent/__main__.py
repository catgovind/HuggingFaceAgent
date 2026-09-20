import sys

from dotenv import load_dotenv

from .agent import HuggingFaceAgent


def main() -> None:
    load_dotenv()  # makes HF_TOKEN from .env available to huggingface_hub / smolagents

    prompt = " ".join(sys.argv[1:]) or "Hello from HuggingFaceAgent"
    agent = HuggingFaceAgent()
    print(agent.run(prompt))


if __name__ == "__main__":
    main()
