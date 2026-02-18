# Google ADK + LangSmith tracing demo

Traces a Google ADK agent to LangSmith using two approaches:

1. **Native integration** — `langsmith.integrations.google_adk.configure_google_adk`
2. **OpenTelemetry** — `langsmith.integrations.otel.configure` (uses ADK's built-in OTel instrumentation)

Both send traces to LangSmith; the difference is the plumbing underneath.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Copy `.env.example` to `.env` and fill in:

```
LANGSMITH_API_KEY=<your key>
LANGSMITH_TRACING=true

GOOGLE_APPLICATION_CREDENTIALS=./vertexCred.json
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_LOCATION=us-central1
```

`GOOGLE_APPLICATION_CREDENTIALS` should point to a Vertex AI service account JSON key.
If you'd rather use the Gemini API directly, drop the `GOOGLE_*` vars and set `GOOGLE_API_KEY` instead.

## Run

```bash
uv run main.py
```

Traces show up in two LangSmith projects: `adk-native` and `adk-otel`.
