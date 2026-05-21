#!/usr/bin/env bash
# Clean stale ExLlamaV3 JIT-compiled CUDA extension caches.
# Run this BEFORE upgrading/reinstalling exllamav3 to avoid stale .so bugs.
#
# Usage:
#   scripts/clean_exllamav3_ext.sh
#   scripts/clean_exllamav3_ext.sh --dry-run

set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[dry-run] %s\n' "$*"
  else
    "$@"
  fi
}

echo "Removing stale ExLlamaV3 JIT extension caches..."

# Torch JIT extension caches (all Python/CUDA variants)
run find "$HOME/.cache/torch_extensions" -maxdepth 3 -type d -name 'exllamav3_ext' -exec rm -rf {} + 2>/dev/null || true
run rm -rf "$HOME/.cache/torch_extensions/exllamav3_ext" 2>/dev/null || true

# Any other exllamav3 caches
run rm -rf "$HOME/.cache/exllamav3" 2>/dev/null || true

echo "Done."
echo ""
echo "Remaining exllamav3 ext dirs in cache:"
find "$HOME/.cache" -maxdepth 4 -type d -name 'exllamav3*' 2>/dev/null || true
echo ""
echo "Now install/upgrade exllamav3. The first import will trigger a fresh JIT compile."
