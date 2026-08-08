#!/usr/bin/env bash
# layerdiff.sh -- 200-token serial vs chunked, diff the per-layer
# state dumps (t8) to find the exact first-divergence layer.
set -u
cd /Users/ruihe/disk-qwen35bA3B
unset PYTHONPATH
ARGS="--trunk /tmp/q35-trunk/trunk.bin --offsets /tmp/q35-trunk/trunk.offsets --layout-trunk /tmp/q35-trunk/trunk.json --pool /tmp/q35-pool/pool.bin --layout-pool /tmp/q35-pool/manifest.json --head /tmp/q35-trunk/head.json --embed /tmp/q35-trunk/embed.json --tokenizer $HOME/.cache/huggingface/mlx-qwen35-a3b-4bit/tokenizer.json --pids-file /tmp/q35-200-ids.txt --gen 8 --cache-gb 5 --pin-layers 4 --mem-limit-gb 20"
export DS4F_GREEDY=1 DS4F_NAN_PROBE=1
mkdir -p /tmp/ld-serial /tmp/ld-chunk
rm -f /tmp/q35-eng-L*-t8.bin
echo "=== serial (200-tok) ==="
bash tools/run-clean.sh ./ds4f /tmp/q35-trunk $ARGS > /tmp/ld-serial.log 2>&1
echo "serial rc=$?"
mv /tmp/q35-eng-L*-t8.bin /tmp/ld-serial/ 2>/dev/null
rm -f /tmp/q35-eng-L*-t8.bin
export DS4F_PREFILL_CHUNK=1 DS4F_PREFILL_B=64
echo "=== chunked ==="
bash tools/run-clean.sh ./ds4f /tmp/q35-trunk $ARGS > /tmp/ld-chunk.log 2>&1
echo "chunk rc=$?"
unset DS4F_PREFILL_CHUNK DS4F_PREFILL_B DS4F_NAN_PROBE
mv /tmp/q35-eng-L*-t8.bin /tmp/ld-chunk/ 2>/dev/null
echo "=== per-layer diff (first divergence) ==="
python3 tools/ld-diff.py /tmp/ld-serial /tmp/ld-chunk
echo "=== timing ==="
grep 'tokens in' /tmp/ld-serial.log /tmp/ld-chunk.log
