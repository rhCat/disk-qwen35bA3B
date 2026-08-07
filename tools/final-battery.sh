#!/usr/bin/env bash
# final-battery.sh -- faithful engine, greedy, longer gen, raw text.
set -u
REPO=/Users/ruihe/disk-qwen35bA3B
TRUNK=/tmp/q35-trunk
POOL=/tmp/q35-pool
GEN="${GEN:-24}"

run_one() {
  local prompt="$1"
  PROMPT="$prompt" DS4F_GREEDY=1 GEN=$GEN bash "$REPO/tools/run-real.sh" \
    "$TRUNK" "$POOL" > /dev/null 2>&1
  local text
  text=$(awk '/^--- run report ---/{exit} {last=$0} END{print last}' /tmp/q35-run.log)
  case "$text" in
    moe:*|kernels:*|PEAK*|cache:*|GB*|"") text="<no text>" ;;
  esac
  printf '%-60s => %s\n' "${prompt:0:58}" "$text"
}

run_one "The capital of France is"
run_one "The capital of Japan is"
run_one "The largest planet in the solar system is"
run_one "The first president of the United States was"
run_one "Water boils at a temperature of"
run_one "The chemical symbol for gold is"
run_one "The tallest mountain on Earth is"
run_one "The currency of the United Kingdom is"
run_one "2 + 2 ="
run_one "The color of the sky is"
