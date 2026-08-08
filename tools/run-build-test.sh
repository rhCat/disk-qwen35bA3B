#!/usr/bin/env bash
# run-build-test.sh -- run a built test binary, capturing rc + output.
set -u
cd /Users/ruihe/disk-qwen35bA3B
B=build/${1:-test_cache}
if [ -x "$B" ]; then
  "$B"
  echo "RC=$?"
else
  echo "missing $B"
fi
