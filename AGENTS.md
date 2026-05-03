# AGENTS.md

Short guide for agents working on Gallama.

## What This Project Is

Gallama is a Python 3.11+ local LLM inference API server for agentic workflows.
It serves local models through OpenAI-compatible, Anthropic-compatible,
Responses API, realtime, embedding, audio, and model-management endpoints.

The most-tested backend is ExLlamaV3. Other backends exist, but treat them as
less-covered unless the task is specifically about them.

## Where To Look

- `src/gallama/cli.py` - `gallama` CLI entry point.
- `src/gallama/server.py` - multi-model server and top-level routing.
- `src/gallama/app.py` - per-model FastAPI app wiring.
- `src/gallama/routes/` - model API routes such as chat, embeddings, audio,
  and websocket endpoints.
- `src/gallama/server_routes/` - server-level routes, realtime, and Responses
  websocket bridge.
- `src/gallama/backend/llm/engine/` - backend implementations.
- `src/gallama/backend/llm/prompt_engine/` - prompt formatting and tool-call
  parsing.
- `src/gallama/backend/llm/prompt_engine/by_model/` - model-family native
  tool-call parsers.
- `src/gallama/data_classes/` - request and response schemas.
- `tests/unit/` - tests that do not need a running model server.
- `tests/live/` - API tests that expect a running Gallama server.
- `.pi/skills/test/` - testing instructions for Pi/simple agents.

## Local Environment

- Use the user's conda env named `exllama` when running commands, so results
  match the normal development environment.
- Runtime config is usually under `~/gallama/model_config.yaml`.
- Set `GALLAMA_HOME_PATH` only when you need a different config directory.

## Common Commands

Run unit tests:

```shell
conda run -n exllama ./scripts/run_unit_tests.sh
```

Run one unit test directly:

```shell
conda run -n exllama env PYTHONPATH="$PWD/src:$PWD/tests" pytest tests/unit/test_file.py -k name
```

Run live API tests against an already running server:

```shell
conda run -n exllama env LOCAL_BASE_URL=http://127.0.0.1:8000 ./scripts/run_live_api_tests.sh
```

Start Gallama:

```shell
conda run -n exllama gallama run
```

Start a fully specified model:

```shell
conda run -n exllama gallama run -id "model_name=minimax model_id=/path/to/model backend=exllamav3"
```

## Rules For Agents

- Keep changes small and scoped to the request.
- Do not revert unrelated user changes.
- Do not touch generated files, caches, local logs, `dist/`, `artifacts/`,
  `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `.coverage`, or
  `coverage.xml` unless asked.
- Preserve OpenAI, Anthropic, and Responses API compatibility unless the user
  explicitly asks for a breaking change.
- Prefer existing Pydantic/dataclass models in `src/gallama/data_classes/` over
  ad hoc dictionaries for structured API data.
- Add or update focused unit tests for behavior changes when practical.
- Avoid live GPU/model-loading tests unless the task requires them or the user
  asks for them.
- Update `README.md`, `tests/README.md`, or this file when behavior or workflow
  instructions change.

## Tool Calling Notes

Native tool calling is model-family specific. If a model emits a new native
tool-call format, add or update a parser in
`src/gallama/backend/llm/prompt_engine/by_model/`.

