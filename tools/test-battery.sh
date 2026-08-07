#!/usr/bin/env bash
# test-battery.sh -- run real prompts, extract the generated text line.
set -u
REPO=/Users/ruihe/disk-qwen35bA3B
TRUNK=/tmp/q35-trunk
POOL=/tmp/q35-pool
GEN="${GEN:-14}"

run_one() {
  local prompt="$1"
  PROMPT="$prompt" DS4F_GREEDY=1 GEN=$GEN bash "$REPO/tools/run-real.sh" \
    "$TRUNK" "$POOL" > /dev/null 2>&1
  # generated text: bare text line right before "--- run report ---"
  local text
  text=$(awk '/^--- run report ---/{exit} {last=$0} END{print last}' /tmp/q35-run.log)
  # strip if it's not text
  case "$text" in
    moe:*|kernels:*|PEAK*|cache:*|GB*|"") text="<no text>" ;;
  esac
  local speed
  speed=$(grep -E 'tokens in' /tmp/q35-run.log | tail -1)
  printf '%-45s => %s\n' "$prompt" "$text"
  printf '  %s\n' "$speed"
}

run_one "The capital of France is"
run_one "The capital of Japan is"
run_one "The largest planet in the solar system is"
run_one "The first president of the United States was"
run_one "Water boils at a temperature of"
run_one "The chemical symbol for gold is"
run_one "The tallest mountain on Earth is"
run_one "The currency of the United Kingdom is"
