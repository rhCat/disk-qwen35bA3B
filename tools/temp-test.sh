#!/usr/bin/env bash
# temp-test.sh -- same prompts at temperature 0.8, seed fixed via env.
set -u
REPO=/Users/ruihe/disk-qwen35bA3B
TRUNK=/tmp/q35-trunk
POOL=/tmp/q35-pool
GEN="${GEN:-14}"
TEMP="${TEMP:-0.8}"

run_one() {
  local prompt="$1"
  PROMPT="$prompt" DS4F_TEMP="$TEMP" GEN=$GEN bash "$REPO/tools/run-real.sh" \
    "$TRUNK" "$POOL" > /dev/null 2>&1
  local text speed
  text=$(awk '/^--- run report ---/{exit} {last=$0} END{print last}' /tmp/q35-run.log)
  case "$text" in
    moe:*|kernels:*|PEAK*|cache:*|GB*|"") text="<no text>" ;;
  esac
  speed=$(grep -E 'tokens in' /tmp/q35-run.log | tail -1 | sed -E 's/[0-9]+ tokens in ([0-9.]+) s, ([0-9.]+) s\/token/\1 s | \2 s\/tok/')
  printf '%-45s => %s\n' "$prompt" "$text"
  printf '  %s\n' "$speed"
}

run_one "The chemical symbol for gold is"
run_one "The currency of the United Kingdom is"
run_one "The capital of France is"
run_one "The largest planet in the solar system is"
run_one "2 + 2 equals"
run_one "The color of the sky is"
