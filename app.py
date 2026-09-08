"""Streamlit UI for the framework-agnostic LangSmith tracing demo.

Pick one of three grocery shopping agent implementations — Google ADK,
OpenAI Agents SDK, or bare code (no framework) — and chat with it. Every
implementation routes model calls through LangChain's LLM gateway and traces
to LangSmith, demonstrating that tracing is framework-agnostic.
"""

import asyncio
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)

st.set_page_config(page_title="Grocery Shopping Assistant", page_icon="🛒")

AGENTS = {
    "Google ADK": {
        "module": "agent_adk",
        "blurb": "Built with Google ADK · LiteLlm → Gemini gateway · native LangSmith ADK integration",
        "async": True,
    },
    "OpenAI Agents SDK": {
        "module": "agent_openai_sdk",
        "blurb": "Built with the OpenAI Agents SDK · OpenAI gateway · OpenAIAgentsTracingProcessor",
        "async": True,
    },
    "Bare code (no framework)": {
        "module": "agent_bare",
        "blurb": "Hand-written tool loop · OpenAI gateway · wrap_openai + @traceable",
        "async": False,
    },
}


def _load_ask(module_name: str):
    import importlib

    module = importlib.import_module(module_name)
    return module.ask


def run_agent(choice: str, message: str) -> str:
    cfg = AGENTS[choice]
    ask = _load_ask(cfg["module"])
    if cfg["async"]:
        return asyncio.run(ask(message))
    return ask(message)


with st.sidebar:
    st.title("🛒 Grocery Assistant")
    st.caption("Framework-agnostic LangSmith tracing")
    choice = st.radio("Agent implementation", list(AGENTS.keys()))
    st.info(AGENTS[choice]["blurb"])
    st.caption(
        "All three route through LangChain's LLM gateway and trace to LangSmith. "
        "Open your LangSmith project to see the traces."
    )
    if st.button("Clear conversation"):
        st.session_state.pop("history", None)
        st.rerun()

st.header(f"Chatting via: {choice}")

history = st.session_state.setdefault("history", [])
for role, text in history:
    with st.chat_message(role):
        st.markdown(text)

prompt = st.chat_input("Ask about aisles, prices, stock, or meal ideas…")
if prompt:
    history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"Thinking with {choice}…"):
            try:
                answer = run_agent(choice, prompt)
            except Exception as exc:  # surface config/gateway errors in the UI
                answer = f"⚠️ Error: {exc}"
        st.markdown(answer)

    history.append(("assistant", answer))
