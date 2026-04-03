#!/bin/sh
set -eu

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
PYTEST_BIN="${PYTEST:-pytest}"

cd "$ROOT_DIR"

UNIT_TESTS=$(
  find tests/unit -maxdepth 1 -type f -name 'test_*.py' | sort
)

if [ -z "$UNIT_TESTS" ]; then
  echo "No unit test files found."
  exit 1
fi

test_count=$(printf '%s\n' "$UNIT_TESTS" | wc -l | tr -d ' ')
echo "Running ${test_count} unit test files with pytest"

failures=0
PYTHONPATH_VALUE="${ROOT_DIR}/src:${ROOT_DIR}/tests${PYTHONPATH:+:${PYTHONPATH}}"

for test_file in $UNIT_TESTS; do
  echo
  echo "==> ${test_file}"
  if ! PYTHONPATH="$PYTHONPATH_VALUE" "$PYTEST_BIN" "$@" "$test_file"; then
    failures=$((failures + 1))
  fi
done

if [ "$failures" -gt 0 ]; then
  echo
  echo "${failures} unit test file(s) failed."
  exit 1
fi
