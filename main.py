import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv(dotenv_path=".env", override=True)

if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
    cred_path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    if not os.path.isabs(cred_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(Path(__file__).resolve().parent / cred_path)


def get_weather(city: str) -> dict:
    """Get weather for a city."""
    return {"city": city, "temperature": "72°F", "conditions": "Sunny"}


def make_agent():
    return Agent(
        name="weather_agent",
        model="gemini-2.0-flash",
        description="Provides weather information.",
        instruction="Use the get_weather tool to answer weather questions.",
        tools=[get_weather],
    )


async def run_agent(runner, session_service, session_id):
    await session_service.create_session(
        app_name="weather_app",
        user_id="user_123",
        session_id=session_id,
    )

    async for event in runner.run_async(
        user_id="user_123",
        session_id=session_id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text="What's the weather in San Francisco?")],
        ),
    ):
        if event.is_final_response():
            print(event.content.parts[0].text)


# ---------------------------------------------------------------------------
# Test 1: Native LangSmith integration (configure_google_adk)
# ---------------------------------------------------------------------------
async def test_native():
    from langsmith.integrations.google_adk import configure_google_adk

    configure_google_adk(project_name="adk-native")

    session_service = InMemorySessionService()
    runner = Runner(agent=make_agent(), app_name="weather_app", session_service=session_service)
    await run_agent(runner, session_service, session_id="native_session")


# ---------------------------------------------------------------------------
# Test 2: OpenTelemetry integration (langsmith.integrations.otel)
# ---------------------------------------------------------------------------
async def test_otel():
    from langsmith.integrations.otel import configure

    configure(project_name="adk-otel")

    session_service = InMemorySessionService()
    runner = Runner(agent=make_agent(), app_name="weather_app", session_service=session_service)
    await run_agent(runner, session_service, session_id="otel_session")


async def main():
    print("=== Test 1: Native LangSmith integration ===")
    await test_native()

    print("\n=== Test 2: OTel → LangSmith integration ===")
    await test_otel()


asyncio.run(main())
