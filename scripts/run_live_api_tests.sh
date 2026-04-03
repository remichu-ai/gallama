#!/bin/sh
set -eu

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
PYTHON_BIN="${PYTHON:-python}"

cd "$ROOT_DIR"

echo "Running live API tests against Gallama at ${LOCAL_BASE_URL:-"(using each script default)"}"

for test_file in \
  tests/live/test_openai.py \
  tests/live/test_anthropic.py \
  tests/live/test_responses.py
do
  echo
  echo "==> ${test_file}"
  PYTHONPATH="${ROOT_DIR}/tests${PYTHONPATH:+:${PYTHONPATH}}" "$PYTHON_BIN" "$test_file"
done
