#!/usr/bin/env bash
# Clean ExLlamaV3 JIT extension state for a conda env before upgrading/reinstalling ExLlamaV3.
#
# Default behavior is conservative:
#   - stop Gallama/ExLlamaV3-related background processes
#   - remove PyTorch JIT build dirs for exllamav3_ext
#   - keep the installed exllamav3 Python package
#
# Use --uninstall to also pip-uninstall exllamav3 and remove package leftovers.
#
# Examples:
#   scripts/clean_exllamav3_env.sh
#   scripts/clean_exllamav3_env.sh --env exllama --uninstall
#   scripts/clean_exllamav3_env.sh --dry-run

set -euo pipefail

ENV_NAME="exllama"
UNINSTALL=0
DRY_RUN=0
KILL_PROCS=1

usage() {
  cat <<'EOF'
Usage: scripts/clean_exllamav3_env.sh [options]

Options:
  --env NAME       Conda env name to clean (default: exllama)
  --uninstall     Also uninstall exllamav3 from the env and remove package leftovers
  --no-kill       Do not stop Gallama/ninja/nvcc processes first
  --dry-run       Print actions without deleting/uninstalling
  -h, --help      Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      ENV_NAME="${2:?--env requires a value}"
      shift 2
      ;;
    --uninstall)
      UNINSTALL=1
      shift
      ;;
    --no-kill)
      KILL_PROCS=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[dry-run] '
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

run_shell() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] $*"
  else
    bash -lc "$*"
  fi
}

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH" >&2
  exit 1
fi

PYTHON_BIN="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)')"
PYTHON_BIN="$(echo "$PYTHON_BIN" | tail -n 1)"

if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "Could not resolve python for conda env '$ENV_NAME'" >&2
  exit 1
fi

echo "Cleaning ExLlamaV3 state for conda env: $ENV_NAME"
echo "Python: $PYTHON_BIN"

if [[ "$KILL_PROCS" == "1" ]]; then
  echo "Stopping Gallama/ExLlamaV3 build processes if present..."
  run_shell 'pkill -f "gallama run" 2>/dev/null || true'
  run_shell 'pkill -f "src/gallama/app.py" 2>/dev/null || true'
  run_shell 'pkill -f "ninja.*exllamav3_ext" 2>/dev/null || true'
  run_shell 'pkill -f "nvcc.*exllamav3" 2>/dev/null || true'
fi

BUILD_DIR="$($PYTHON_BIN - <<'PY'
from torch.utils.cpp_extension import _get_build_directory
print(_get_build_directory('exllamav3_ext', verbose=False))
PY
)"
BUILD_DIR="$(echo "$BUILD_DIR" | tail -n 1)"

echo "Removing ExLlamaV3 PyTorch JIT extension build/cache dirs..."
if [[ -n "$BUILD_DIR" ]]; then
  run rm -rf "$BUILD_DIR"
fi
# Also remove stale builds from other CUDA/Python variants, which can leave locks around after upgrades.
run_shell 'rm -rf "$HOME"/.cache/torch_extensions/*/exllamav3_ext "$HOME"/.cache/torch_extensions/exllamav3_ext 2>/dev/null || true'

if [[ "$UNINSTALL" == "1" ]]; then
  echo "Uninstalling exllamav3 package from env..."
  run conda run -n "$ENV_NAME" python -m pip uninstall -y exllamav3

  SITE_DIR="$($PYTHON_BIN - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
  SITE_DIR="$(echo "$SITE_DIR" | tail -n 1)"
  if [[ -n "$SITE_DIR" && -d "$SITE_DIR" ]]; then
    echo "Removing exllamav3 leftovers from: $SITE_DIR"
    run_shell 'rm -rf '"$(printf '%q' "$SITE_DIR")"'/exllamav3 '"$(printf '%q' "$SITE_DIR")"'/exllamav3-'"'"'*.dist-info'"'"' '"$(printf '%q' "$SITE_DIR")"'/exllamav3_'"'"'*.dist-info'"'"' '"$(printf '%q' "$SITE_DIR")"'/exllamav3_ext* 2>/dev/null || true'
  fi
fi

echo "Remaining exllamav3_ext cache dirs:"
find "$HOME/.cache/torch_extensions" -maxdepth 3 -type d -name 'exllamav3_ext' -print 2>/dev/null || true

echo "Done. Next install example:"
echo "  conda run -n $ENV_NAME python -m pip install --no-cache-dir --force-reinstall exllamav3"
echo "Then force a clean JIT rebuild with:"
echo "  conda run -n $ENV_NAME python -c 'from exllamav3 import Config, Model, Cache, Tokenizer; print(\"exllamav3 OK\")'"
