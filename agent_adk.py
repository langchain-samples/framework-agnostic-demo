"""Grocery shopping agent built with Google ADK.

Routes model calls through LangChain's LLM gateway (OpenAI-compatible endpoint)
via LiteLlm, and traces to LangSmith using the native ADK integration.
"""

import os
import uuid

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from grocery_tools import ALL_TOOLS, SYSTEM_PROMPT

GATEWAY_OPENAI_BASE_URL = "https://gateway.smith.langchain.com/openai/v1"
GATEWAY_MODEL = "gpt-5.5"

_APP_NAME = "grocery_adk"
_USER_ID = "shopper"


def _make_agent() -> Agent:
    api_key = os.environ["LANGSMITH_API_KEY"]
    # LiteLlm talks to the OpenAI-compatible gateway endpoint using the
    # LangSmith API key as the bearer token; the gateway supplies the real
    # OPENAI_API_KEY from workspace secrets.
    model = LiteLlm(
        model=f"openai/{GATEWAY_MODEL}",
        api_base=GATEWAY_OPENAI_BASE_URL,
        api_key=api_key,
    )
    return Agent(
        name="grocery_agent",
        model=model,
        description="In-store grocery shopping assistant.",
        instruction=SYSTEM_PROMPT,
        tools=ALL_TOOLS,
    )


def _configure_tracing() -> None:
    from langsmith.integrations.google_adk import configure_google_adk

    configure_google_adk(project_name=os.environ.get("LANGSMITH_PROJECT", "grocery-adk"))


async def ask(message: str) -> str:
    """Send one message to the ADK agent and return its final text response."""
    _configure_tracing()

    session_service = InMemorySessionService()
    runner = Runner(
        agent=_make_agent(),
        app_name=_APP_NAME,
        session_service=session_service,
    )

    session_id = str(uuid.uuid4())
    await session_service.create_session(
        app_name=_APP_NAME, user_id=_USER_ID, session_id=session_id
    )

    final_text = ""
    async for event in runner.run_async(
        user_id=_USER_ID,
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text=message)]),
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_text = event.content.parts[0].text or ""
    return final_text
