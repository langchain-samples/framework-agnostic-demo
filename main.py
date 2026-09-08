"""CLI runner: exercise all three grocery agent implementations once each.

Demonstrates framework-agnostic LangSmith tracing — Google ADK, the OpenAI
Agents SDK, and bare code all route through LangChain's LLM gateway and trace
to LangSmith. For an interactive experience, run the Streamlit UI instead:

    uv run streamlit run app.py
"""

import asyncio
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)

QUESTION = "I want to make pasta tonight. What do I need, which aisles, and how much?"


async def main() -> None:
    import agent_adk
    import agent_bare
    import agent_openai_sdk

    print("=== Google ADK ===")
    print(await agent_adk.ask(QUESTION))

    print("\n=== OpenAI Agents SDK ===")
    print(await agent_openai_sdk.ask(QUESTION))

    print("\n=== Bare code (no framework) ===")
    print(agent_bare.ask(QUESTION))


if __name__ == "__main__":
    asyncio.run(main())
