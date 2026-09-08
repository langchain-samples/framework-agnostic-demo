"""Grocery shopping agent built with the OpenAI Agents SDK.

Routes model calls through LangChain's LLM gateway (OpenAI-compatible endpoint)
and traces to LangSmith via OpenAIAgentsTracingProcessor.
"""

import os

from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    Runner,
    function_tool,
    set_default_openai_client,
    set_default_openai_api,
    set_tracing_disabled,
    set_trace_processors,
)
from openai import AsyncOpenAI

import grocery_tools
from grocery_tools import SYSTEM_PROMPT

GATEWAY_OPENAI_BASE_URL = "https://gateway.smith.langchain.com/openai/v1"
GATEWAY_MODEL = "gpt-5.5"

_tracing_ready = False


# Wrap the shared plain functions as OpenAI Agents SDK tools.
@function_tool
def find_item_aisle(item: str) -> dict:
    """Find which aisle a grocery item is located in."""
    return grocery_tools.find_item_aisle(item)


@function_tool
def check_price(item: str) -> dict:
    """Check the current price of a grocery item."""
    return grocery_tools.check_price(item)


@function_tool
def check_stock(item: str) -> dict:
    """Check how many units of an item are currently in stock."""
    return grocery_tools.check_stock(item)


@function_tool
def suggest_recipe(meal: str) -> dict:
    """Suggest a shopping list of ingredients for a meal idea."""
    return grocery_tools.suggest_recipe(meal)


def _configure() -> None:
    global _tracing_ready
    if _tracing_ready:
        return

    from langsmith.integrations.openai_agents_sdk import OpenAIAgentsTracingProcessor

    api_key = os.environ["LANGSMITH_API_KEY"]
    client = AsyncOpenAI(base_url=GATEWAY_OPENAI_BASE_URL, api_key=api_key)
    set_default_openai_client(client, use_for_tracing=False)
    set_default_openai_api("chat_completions")
    # The gateway (not OpenAI) is our LLM backend; disable the SDK's own
    # OpenAI-hosted tracing and route spans to LangSmith instead.
    set_tracing_disabled(False)
    set_trace_processors([OpenAIAgentsTracingProcessor()])
    _tracing_ready = True


def _make_agent() -> Agent:
    model = OpenAIChatCompletionsModel(
        model=GATEWAY_MODEL,
        openai_client=AsyncOpenAI(
            base_url=GATEWAY_OPENAI_BASE_URL,
            api_key=os.environ["LANGSMITH_API_KEY"],
        ),
    )
    return Agent(
        name="grocery_agent",
        instructions=SYSTEM_PROMPT,
        model=model,
        tools=[find_item_aisle, check_price, check_stock, suggest_recipe],
    )


async def ask(message: str) -> str:
    """Send one message to the OpenAI Agents SDK agent and return its response."""
    _configure()
    result = await Runner.run(_make_agent(), message)
    return result.final_output
