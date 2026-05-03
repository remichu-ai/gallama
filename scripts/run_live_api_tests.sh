#!/bin/sh
set -eu

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
PYTHON_BIN="${PYTHON:-}"
ENABLE_REASONING_TESTS="${ENABLE_REASONING_TESTS:-1}"
ONLY_REASONING_TESTS="${ONLY_REASONING_TESTS:-0}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --reasoning)
      if [ "$#" -lt 2 ]; then
        echo "Missing value for --reasoning (expected on/off)" >&2
        exit 1
      fi
      case "$2" in
        on|true|1)
          ENABLE_REASONING_TESTS=1
          ;;
        off|false|0)
          ENABLE_REASONING_TESTS=0
          ;;
        *)
          echo "Invalid value for --reasoning: $2 (expected on/off)" >&2
          exit 1
          ;;
      esac
      shift 2
      ;;
    --only-reasoning)
      ONLY_REASONING_TESTS=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

cd "$ROOT_DIR"

if [ -z "$PYTHON_BIN" ]; then
  for candidate in python python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c "import openai, anthropic" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

if ! "$PYTHON_BIN" -c "import openai, anthropic" >/dev/null 2>&1; then
  echo "Selected interpreter '$PYTHON_BIN' is missing required packages: openai, anthropic" >&2
  exit 1
fi

echo "Running live API tests against Gallama at ${LOCAL_BASE_URL:-"(using each script default)"}"
echo "Reasoning tests: ${ENABLE_REASONING_TESTS}"
echo "Only reasoning tests: ${ONLY_REASONING_TESTS}"

OPENAI_BASE_URL=""
ANTHROPIC_BASE_URL=""
RESPONSES_BASE_URL=""

if [ -n "${LOCAL_BASE_URL:-}" ]; then
  case "$LOCAL_BASE_URL" in
    */v1)
      OPENAI_BASE_URL="$LOCAL_BASE_URL"
      RESPONSES_BASE_URL="$LOCAL_BASE_URL"
      ANTHROPIC_BASE_URL=${LOCAL_BASE_URL%/v1}
      ;;
    *)
      OPENAI_BASE_URL="${LOCAL_BASE_URL%/}/v1"
      RESPONSES_BASE_URL="${LOCAL_BASE_URL%/}/v1"
      ANTHROPIC_BASE_URL="${LOCAL_BASE_URL%/}"
      ;;
  esac
fi

if [ "$ONLY_REASONING_TESTS" = "1" ]; then
  TEST_FILES="
tests/live/test_anthropic.py
tests/live/test_responses.py
"
else
  TEST_FILES="
tests/live/test_openai.py
tests/live/test_anthropic.py
tests/live/test_responses.py
"
fi

for test_file in $TEST_FILES; do
  echo
  echo "==> ${test_file}"
  test_base_url=""
  case "$test_file" in
    tests/live/test_openai.py)
      test_base_url="$OPENAI_BASE_URL"
      ;;
    tests/live/test_anthropic.py)
      test_base_url="$ANTHROPIC_BASE_URL"
      ;;
    tests/live/test_responses.py)
      test_base_url="$RESPONSES_BASE_URL"
      ;;
  esac

  if [ -n "$test_base_url" ]; then
    PYTHONPATH="${ROOT_DIR}/tests${PYTHONPATH:+:${PYTHONPATH}}" LOCAL_BASE_URL="$test_base_url" ENABLE_REASONING_TESTS="$ENABLE_REASONING_TESTS" ONLY_REASONING_TESTS="$ONLY_REASONING_TESTS" "$PYTHON_BIN" "$test_file"
  else
    PYTHONPATH="${ROOT_DIR}/tests${PYTHONPATH:+:${PYTHONPATH}}" ENABLE_REASONING_TESTS="$ENABLE_REASONING_TESTS" ONLY_REASONING_TESTS="$ONLY_REASONING_TESTS" "$PYTHON_BIN" "$test_file"
  fi
done
