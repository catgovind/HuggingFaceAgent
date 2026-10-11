"""Local tracing for CodeAgent runs via OpenTelemetry (no external account needed).

The course's own example sends traces to Langfuse's cloud, which needs a
signup and API keys. This instead prints each span to stdout, so you can see
the same per-step/per-tool-call structure without creating any account.

https://huggingface.co/learn/agents-course/en/unit2/smolagents/code_agents
"""

from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor


def enable_console_tracing() -> None:
    """Instrument smolagents so every agent run emits spans to stdout.

    Call this once, before constructing any CodeAgent or ToolCallingAgent.
    """
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    SmolagentsInstrumentor().instrument(tracer_provider=provider)
