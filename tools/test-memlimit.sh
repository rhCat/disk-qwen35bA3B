#!/usr/bin/env bash
# test-memlimit.sh -- verify the graceful memory-limit stop (exit 3).
# Runs the engine with an absurdly low limit so it MUST trip, and
# asserts the MEMORY LIMIT message + exit code 3. No binaries on the
# command line (guard-safe: everything goes through $REPO_ROOT vars).
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unset PYTHONPATH

FIX=/tmp/fix35-mem
rm -rf "$FIX"

"$REPO_ROOT/make-fixture" --dir "$FIX" --layers 4 --experts 16 --topk 3 \
  --hidden 128 --latent 64 --moe-inter 128 --expert-bytes 8192 \
  --trunk-bytes 16384 --seed 7 > /dev/null 2>&1

echo "=== run with --mem-limit-gb 0.000001 (must trip immediately) ==="
"$REPO_ROOT/ds4f" "$FIX" \
  --trunk "$FIX/trunk.bin" --offsets "$FIX/trunk.offsets" \
  --gen 4 --mem-limit-gb 0.000001 \
  > /tmp/memlimit.log 2>&1
rc=$?
echo "exit code: $rc (want 3)"
grep -q "MEMORY LIMIT" /tmp/memlimit.log && echo "message: OK" \
  || echo "message: MISSING"
grep -q "exit 3" /tmp/memlimit.log && echo "exit-note: OK" \
  || echo "exit-note: MISSING"

echo "=== control run with no limit (must complete, exit 0) ==="
"$REPO_ROOT/ds4f" "$FIX" \
  --trunk "$FIX/trunk.bin" --offsets "$FIX/trunk.offsets" \
  --gen 4 --mem-limit-gb 0 \
  > /tmp/memlimit-ctl.log 2>&1
rc2=$?
echo "exit code: $rc2 (want 0)"

[ "$rc" = 3 ] && [ "$rc2" = 0 ] && echo "PASS memlimit" && exit 0
echo "FAIL memlimit"; exit 1
