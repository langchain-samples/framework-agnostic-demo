"""Grocery shopping agent built with bare code — no agent framework.

A hand-written tool-calling loop over raw OpenAI chat completions. Model calls
route through LangChain's LLM gateway (OpenAI-compatible endpoint) and are traced
to LangSmith via wrap_openai + @traceable.
"""

import json
import os

from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai import OpenAI

import grocery_tools
from grocery_tools import SYSTEM_PROMPT

GATEWAY_OPENAI_BASE_URL = "https://gateway.smith.langchain.com/openai/v1"
GATEWAY_MODEL = "gpt-5.5"

# JSON schema descriptions of the shared tools for the chat completions API.
_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "find_item_aisle",
            "description": "Find which aisle a grocery item is located in.",
            "parameters": {
                "type": "object",
                "properties": {"item": {"type": "string"}},
                "required": ["item"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_price",
            "description": "Check the current price of a grocery item.",
            "parameters": {
                "type": "object",
                "properties": {"item": {"type": "string"}},
                "required": ["item"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "Check how many units of an item are currently in stock.",
            "parameters": {
                "type": "object",
                "properties": {"item": {"type": "string"}},
                "required": ["item"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_recipe",
            "description": "Suggest a shopping list of ingredients for a meal idea.",
            "parameters": {
                "type": "object",
                "properties": {"meal": {"type": "string"}},
                "required": ["meal"],
            },
        },
    },
]

_DISPATCH = {
    "find_item_aisle": grocery_tools.find_item_aisle,
    "check_price": grocery_tools.check_price,
    "check_stock": grocery_tools.check_stock,
    "suggest_recipe": grocery_tools.suggest_recipe,
}


def _client() -> OpenAI:
    return wrap_openai(
        OpenAI(
            base_url=GATEWAY_OPENAI_BASE_URL,
            api_key=os.environ["LANGSMITH_API_KEY"],
        )
    )


@traceable(run_type="tool", name="grocery_tool")
def _run_tool(name: str, args: dict) -> dict:
    fn = _DISPATCH[name]
    return fn(**args)


@traceable(name="Grocery Assistant (bare code)")
def ask(message: str) -> str:
    """Send one message to the bare-code agent and return its final response."""
    client = _client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": message},
    ]

    # Manual tool-calling loop, capped to avoid runaway iterations.
    for _ in range(6):
        response = client.chat.completions.create(
            model=GATEWAY_MODEL,
            messages=messages,
            tools=_TOOL_SCHEMAS,
        )
        choice = response.choices[0].message

        if not choice.tool_calls:
            return choice.content or ""

        messages.append(choice)
        for tool_call in choice.tool_calls:
            args = json.loads(tool_call.function.arguments or "{}")
            result = _run_tool(tool_call.function.name, args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

    return "Sorry, I couldn't complete that request within the step limit."
