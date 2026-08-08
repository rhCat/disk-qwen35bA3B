#!/usr/bin/env bash
# tier-ab.sh -- 200/500/1500 token tiers: serial vs chunked, top1
# identity + decoded response. Usage: tier-ab.sh [tiers...]
set -u
cd /Users/ruihe/disk-qwen35bA3B
unset PYTHONPATH
GEN=8
B=64
export DS4F_GREEDY=1 DS4F_DEBUG7=1
ARGS_BASE="--trunk /tmp/q35-trunk/trunk.bin --offsets /tmp/q35-trunk/trunk.offsets --layout-trunk /tmp/q35-trunk/trunk.json --pool /tmp/q35-pool/pool.bin --layout-pool /tmp/q35-pool/manifest.json --head /tmp/q35-trunk/head.json --embed /tmp/q35-trunk/embed.json --tokenizer $HOME/.cache/huggingface/mlx-qwen35-a3b-4bit/tokenizer.json --cache-gb 5 --pin-layers 4 --mem-limit-gb 20"
TIERS="${*:-200 500 1500}"
for T in $TIERS; do
    IDS="/tmp/q35-${T}-ids.txt"
    ARGS="$ARGS_BASE --pids-file $IDS --gen $GEN"
    unset DS4F_PREFILL_CHUNK DS4F_PREFILL_B DS4F_CHUNK_MS
    echo "=========== TIER $T (${GEN} gen) ==========="
    bash tools/run-clean.sh ./ds4f /tmp/q35-trunk $ARGS > /tmp/tier-${T}-serial.log 2>&1
    echo "serial rc=$?"
    export DS4F_PREFILL_CHUNK=1 DS4F_PREFILL_B=$B
    bash tools/run-clean.sh ./ds4f /tmp/q35-trunk $ARGS > /tmp/tier-${T}-chunk.log 2>&1
    echo "chunk rc=$?"
    unset DS4F_PREFILL_CHUNK DS4F_PREFILL_B
    echo "--- timing ---"
    grep 'tokens in' /tmp/tier-${T}-serial.log /tmp/tier-${T}-chunk.log
    echo "--- gen top1 identity ---"
    python3 tools/ab-compare.py /tmp/tier-${T}-serial.log /tmp/tier-${T}-chunk.log "$IDS"
    echo "--- ACTUAL RESPONSE ---"
    echo "[serial]"
    python3 tools/acc-decode.py /tmp/tier-${T}-serial.log "$IDS" 400
    echo "[chunked]"
    python3 tools/acc-decode.py /tmp/tier-${T}-chunk.log "$IDS" 400
    echo ""
done
