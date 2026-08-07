#!/usr/bin/env bash
# perf-battery.sh -- run prompts, capture output + perf metrics, print table.
set -u
REPO=/Users/ruihe/disk-qwen35bA3B
TRUNK=/tmp/q35-trunk
POOL=/tmp/q35-pool
GEN="${GEN:-14}"
TEMP="${TEMP:-}"

run_one() {
  local prompt="$1"
  local label="$2"
  local text speed rss gb
  if [ -n "$TEMP" ]; then
    PROMPT="$prompt" DS4F_TEMP="$TEMP" GEN=$GEN bash "$REPO/tools/run-real.sh" \
      "$TRUNK" "$POOL" > /dev/null 2>&1
  else
    PROMPT="$prompt" DS4F_GREEDY=1 GEN=$GEN bash "$REPO/tools/run-real.sh" \
      "$TRUNK" "$POOL" > /dev/null 2>&1
  fi
  text=$(awk '/^--- run report ---/{exit} {last=$0} END{print last}' /tmp/q35-run.log)
  case "$text" in
    moe:*|kernels:*|PEAK*|cache:*|GB*|"") text="<no text>" ;;
  esac
  speed=$(grep -E 'tokens in' /tmp/q35-run.log | tail -1 | sed -E 's/[0-9]+ tokens in ([0-9.]+) s, ([0-9.]+) s\/token/\1 s | \2 s\/tok/')
  rss=$(grep -E 'PEAK RSS' /tmp/q35-run.log | tail -1 | grep -oE '[0-9.]+ GB')
  gb=$(grep -E 'GB read per token' /tmp/q35-run.log | tail -1 | grep -oE 'per token: [0-9.]+' | grep -oE '[0-9.]+')
  printf '%s|%s|%s|%s|%s\n' "$label" "$text" "$speed" "$rss" "$gb"
}

{
  echo "PROMPT|OUTPUT|TIME|RSS|GB/tok"
  run_one "The capital of France is" "France"
  run_one "The capital of Japan is" "Japan"
  run_one "The largest planet in the solar system is" "Planet"
  run_one "The first president of the United States was" "President"
  run_one "Water boils at a temperature of" "Boiling"
  run_one "The chemical symbol for gold is" "Gold"
  run_one "The tallest mountain on Earth is" "Mountain"
  run_one "The currency of the United Kingdom is" "UK currency"
} > /tmp/perf-battery.txt
echo DONE >> /tmp/perf-battery.txt
