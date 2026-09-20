from huggingface_agent.agent import HuggingFaceAgent


def test_agent_run_returns_prompt():
    agent = HuggingFaceAgent(name="DemoAgent")
    assert agent.run("Hello") == "DemoAgent received: Hello"


def test_agent_chat_alias():
    agent = HuggingFaceAgent(name="DemoAgent")
    assert agent.chat("Hi") == "DemoAgent received: Hi"
