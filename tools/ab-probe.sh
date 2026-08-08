#!/usr/bin/env bash
# ab-probe.sh -- tiny 12-token prompt, serial vs chunk, NAN_PROBE L0/L1 dumps.
set -u
cd /Users/ruihe/disk-qwen35bA3B
unset PYTHONPATH
export DS4F_NAN_PROBE=1 DS4F_GREEDY=1
unset DS4F_PREFILL_CHUNK DS4F_PREFILL_B
ARGS="--trunk /tmp/q35-trunk/trunk.bin --offsets /tmp/q35-trunk/trunk.offsets --layout-trunk /tmp/q35-trunk/trunk.json --pool /tmp/q35-pool/pool.bin --layout-pool /tmp/q35-pool/manifest.json --head /tmp/q35-trunk/head.json --embed /tmp/q35-trunk/embed.json --tokenizer $HOME/.cache/huggingface/mlx-qwen35-a3b-4bit/tokenizer.json --pids-file /tmp/bench-gpu-ids.txt --gen 2 --cache-gb 5 --pin-layers 4 --mem-limit-gb 20"
bash tools/run-clean.sh ./ds4f /tmp/q35-trunk $ARGS > /tmp/probe-serial.log 2>&1
echo "serial rc=$?"
export DS4F_PREFILL_CHUNK=1 DS4F_PREFILL_B=8
bash tools/run-clean.sh ./ds4f /tmp/q35-trunk $ARGS > /tmp/probe-chunk.log 2>&1
echo "chunk rc=$?"
echo "=== serial L0/L1 traces ==="
grep -E '\[lin\] L[01] |\[linout\] L[01] |\[postconv\]|\[moeacc\] L[01]' /tmp/probe-serial.log | head -12
echo "=== chunk L0/L1 traces ==="
grep -E '\[lin\] L[01] |\[linout\] L[01] |\[postconv\]|\[moeacc\] L[01]' /tmp/probe-chunk.log | head -12
