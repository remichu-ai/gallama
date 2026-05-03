---
description: Run, add, fix, and report Gallama unit or live API tests.
---

# Test Skill

Use this skill when the task asks you to run, add, fix, or explain tests for
Gallama.

## Default Test Environment

Use the user's conda environment:

```shell
conda run -n exllama <command>
```

Run commands from the repository root.

## Unit Tests

Prefer unit tests for normal code changes. They do not require a running model
server.

Run all unit tests:

```shell
conda run -n exllama ./scripts/run_unit_tests.sh
```

Run a single unit test file:

```shell
conda run -n exllama env PYTHONPATH="$PWD/src:$PWD/tests" pytest tests/unit/test_file.py
```

Run a specific test or pattern:

```shell
conda run -n exllama env PYTHONPATH="$PWD/src:$PWD/tests" pytest tests/unit/test_file.py -k test_name
```

## Live API Tests

Only run live tests when a Gallama server is already running or the user asks
for live validation. These tests call the real HTTP API.

Run live API tests:

```shell
conda run -n exllama env LOCAL_BASE_URL=http://127.0.0.1:8000 ./scripts/run_live_api_tests.sh
```

Run only live reasoning tests:

```shell
conda run -n exllama ./scripts/run_live_api_tests.sh --only-reasoning
```

## When Adding Tests

- Put fast tests in `tests/unit/`.
- Keep live server tests in `tests/live/`.
- Match the existing test style before adding helpers.
- For API schema or compatibility changes, test the serialized response shape.
- For tool-calling changes, include realistic model output examples when
  possible.
- For streaming changes, check event order and final/error events.

## Reporting Results

When finished, report:

- The exact command run.
- Whether it passed or failed.
- The failing test name and first useful error if it failed.
- Any tests skipped because they require a running server, GPU, model, or
  missing dependency.
