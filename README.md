# Framework-agnostic LangSmith tracing demo

A grocery **in-store shopping assistant** built **three different ways** — to show
that LangSmith tracing is framework-agnostic. All three implementations route
their model calls through **LangChain's LLM gateway** and trace to LangSmith:

| Implementation | Framework | Model (via gateway) | Tracing |
| --- | --- | --- | --- |
| `agent_adk.py` | Google ADK | `gemini-3.5-flash-lite` (LiteLlm) | native `configure_google_adk` |
| `agent_openai_sdk.py` | OpenAI Agents SDK | `gpt-5.6-luna` | `OpenAIAgentsTracingProcessor` |
| `agent_bare.py` | none (hand-written tool loop) | `gpt-5.6-luna` | `wrap_openai` + `@traceable` |

All three share the same in-store shopping tools (`grocery_tools.py`): find an
item's aisle, check price, check stock, and suggest a recipe shopping list.

The **LangChain LLM gateway** is the single point every implementation calls:

- Gemini endpoint: `https://gateway.smith.langchain.com/gemini`
- OpenAI-compatible endpoint: `https://gateway.smith.langchain.com/openai/v1`

Your `LANGSMITH_API_KEY` is used both as the gateway API key **and** to send
traces to LangSmith.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Copy `.env.example` to `.env` and fill in your LangSmith API key:

```
LANGSMITH_API_KEY=<your key>
LANGSMITH_PROJECT="grocery-agents"
LANGSMITH_TRACING="true"
```

## Run the UI (recommended)

A Streamlit chat UI lets you pick which implementation to talk to:

```bash
uv run streamlit run app.py
```

Use the sidebar to switch between **Google ADK**, **OpenAI Agents SDK**, and
**Bare code**, then chat. Open your LangSmith project to watch traces land from
all three.

## Run from the CLI

Fire one question at each of the three implementations in sequence:

```bash
uv run main.py
```

Traces from every implementation show up in the LangSmith project set by
`LANGSMITH_PROJECT`.
