"""Workflows: the raw Workflow/@step/Event framework that AgentWorkflow
(agents/party_agents.py) is actually built on top of.

Scoped to this lower-level API specifically -- the AgentWorkflow multi-agent
pattern from the course's step 6/7 is already covered in
agents/party_agents.py's MultiAgentPartyPlanner, so it isn't rebuilt here.

https://huggingface.co/learn/agents-course/en/unit2/llama-index/workflows
"""

import asyncio
import random
import sys

sys.stdout.reconfigure(encoding="utf-8")

from llama_index.core.workflow import Context, Event, StartEvent, StopEvent, Workflow, step


class HelloWorkflow(Workflow):
    """1. The simplest possible Workflow: one step, StartEvent straight to StopEvent."""

    @step
    async def say_hello(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result="Hello from a one-step Workflow!")


class ThemeChosenEvent(Event):
    """Carries data from choose_theme to check_rsvps."""

    theme: str


class RetryRSVPEvent(Event):
    """Triggers check_rsvps again -- this is what creates the loop."""

    message: str


class RSVPCheckEvent(Event):
    """Carries data from check_rsvps to plan_menu, once enough guests confirmed."""

    confirmed_guests: int


class MenuPlannedEvent(Event):
    """Carries data from plan_menu to finalize."""

    menu: str


class PartyPlanningWorkflow(Workflow):
    """2+3+5. Multiple steps, a loop, and Context state, modeling party planning:

        choose_theme -> check_rsvps (loops on itself via RetryRSVPEvent
        until enough guests are confirmed, or 3 attempts) -> plan_menu -> finalize

    check_rsvps accepting `ThemeChosenEvent | RetryRSVPEvent` as input, and
    returning `RSVPCheckEvent | RetryRSVPEvent`, is exactly the course's
    loop/branch mechanism: a step that can trigger itself again.

    Context (ctx.store) holds `theme`, since RetryRSVPEvent doesn't carry it
    -- the loop would lose the theme on every retry without it. This is the
    actual reason Context exists: state an Event chain alone can't carry
    across a loop.
    """

    @step
    async def choose_theme(self, ctx: Context, ev: StartEvent) -> ThemeChosenEvent:
        theme = ev.get("theme", "villain masquerade")
        await ctx.store.set("theme", theme)
        await ctx.store.set("rsvp_attempts", 0)
        return ThemeChosenEvent(theme=theme)

    @step
    async def check_rsvps(
        self, ctx: Context, ev: ThemeChosenEvent | RetryRSVPEvent
    ) -> RSVPCheckEvent | RetryRSVPEvent:
        attempts = await ctx.store.get("rsvp_attempts") + 1
        await ctx.store.set("rsvp_attempts", attempts)

        confirmed = random.randint(2, 10)
        if confirmed < 5 and attempts < 3:
            return RetryRSVPEvent(message=f"Only {confirmed} confirmed, checking again (attempt {attempts}).")
        return RSVPCheckEvent(confirmed_guests=confirmed)

    @step
    async def plan_menu(self, ctx: Context, ev: RSVPCheckEvent) -> MenuPlannedEvent:
        theme = await ctx.store.get("theme")
        menu = (
            "Buffet with high-energy and healthy food."
            if theme == "superhero"
            else "3-course dinner with wine and dessert."
        )
        await ctx.store.set("confirmed_guests", ev.confirmed_guests)
        return MenuPlannedEvent(menu=menu)

    @step
    async def finalize(self, ctx: Context, ev: MenuPlannedEvent) -> StopEvent:
        theme = await ctx.store.get("theme")
        guests = await ctx.store.get("confirmed_guests")
        attempts = await ctx.store.get("rsvp_attempts")
        result = (
            f"Party plan: theme='{theme}', confirmed_guests={guests}, "
            f"menu='{ev.menu}' (took {attempts} RSVP check(s))"
        )
        return StopEvent(result=result)


def visualize(output_path: str = "party_workflow.html") -> None:
    """4. Visualize every possible path through PartyPlanningWorkflow as an HTML file."""
    from llama_index.utils.workflow import draw_all_possible_flows

    draw_all_possible_flows(PartyPlanningWorkflow, output_path)
    print(f"Workflow diagram written to: {output_path}")


async def main() -> None:
    print("=== 1. HelloWorkflow ===")
    hello = HelloWorkflow(timeout=10, verbose=False)
    print(await hello.run())

    print("\n=== 2+3+5. PartyPlanningWorkflow (with loop + Context state) ===")
    planner = PartyPlanningWorkflow(timeout=10, verbose=True)
    result = await planner.run(theme="superhero")
    print("\nFINAL:", result)

    print("\n=== 4. Visualize ===")
    visualize()


if __name__ == "__main__":
    asyncio.run(main())
