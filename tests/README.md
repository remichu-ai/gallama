Tests are split into two categories.

Live API tests:
- `tests/live/test_openai.py`
- `tests/live/test_anthropic.py`
- `tests/live/test_responses.py`
- These expect a running Gallama server and hit the real HTTP API.
- Run them with `./scripts/run_live_api_tests.sh`.

Unit tests:
- Everything under `tests/unit/test_*.py`
- These run with `pytest` and do not require a running Gallama model.
- Run them with `./scripts/run_unit_tests.sh`.

Model notes for the live suites:
- Tool-calling and vision tests have been run with `Qwen3-VL-32B`.
- Thinking tests have been run with `minimax m2.5`.
