#!/usr/bin/env bash
# chat-test.sh -- chat-formatted (exact mlx-lm template incl <think>), greedy.
set -u
REPO=/Users/ruihe/disk-qwen35bA3B
TRUNK=/tmp/q35-trunk
POOL=/tmp/q35-pool
GEN="${GEN:-30}"

run_one() {
  local prompt="$1"
  PROMPT="$prompt" DS4F_GREEDY=1 GEN=$GEN bash "$REPO/tools/run-real.sh" \
    "$TRUNK" "$POOL" > /dev/null 2>&1
  local text
  text=$(awk '/^--- run report ---/{exit} {last=$0} END{print last}' /tmp/q35-run.log)
  case "$text" in
    moe:*|kernels:*|PEAK*|cache:*|GB*|"") text="<no text>" ;;
  esac
  printf '%s\n  => %s\n' "$prompt" "$text"
}

run_one "<|im_start|>user
The capital of France is<|im_end|>
<|im_start|>assistant
<think>
"
run_one "<|im_start|>user
The largest planet in the solar system is<|im_end|>
<|im_start|>assistant
<think>
"
run_one "<|im_start|>user
What is the chemical symbol for gold?<|im_end|>
<|im_start|>assistant
<think>
"
run_one "<|im_start|>user
2 + 2 = ?<|im_end|>
<|im_start|>assistant
<think>
"
