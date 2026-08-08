#!/usr/bin/env bash
# layerdiff2.sh -- 200-tok serial vs chunked, per-token L<4 state diff.
set -u
cd /Users/ruihe/disk-qwen35bA3B
unset PYTHONPATH
ARGS="--trunk /tmp/q35-trunk/trunk.bin --offsets /tmp/q35-trunk/trunk.offsets --layout-trunk /tmp/q35-trunk/trunk.json --pool /tmp/q35-pool/pool.bin --layout-pool /tmp/q35-pool/manifest.json --head /tmp/q35-trunk/head.json --embed /tmp/q35-trunk/embed.json --tokenizer $HOME/.cache/huggingface/mlx-qwen35-a3b-4bit/tokenizer.json --pids-file /tmp/q35-200-ids.txt --gen 8 --cache-gb 5 --pin-layers 4 --mem-limit-gb 20"
export DS4F_GREEDY=1 DS4F_NAN_PROBE=1
mkdir -p /tmp/ld2-serial /tmp/ld2-chunk
rm -f /tmp/q35-eng-L*-t*.bin
echo "=== serial (200-tok) ==="
bash tools/run-clean.sh ./ds4f /tmp/q35-trunk $ARGS > /tmp/ld2-serial.log 2>&1
echo "serial rc=$?"
mv /tmp/q35-eng-L*-t*.bin /tmp/ld2-serial/ 2>/dev/null
rm -f /tmp/q35-eng-L*-t*.bin
export DS4F_PREFILL_CHUNK=1 DS4F_PREFILL_B=64
echo "=== chunked ==="
bash tools/run-clean.sh ./ds4f /tmp/q35-trunk $ARGS > /tmp/ld2-chunk.log 2>&1
echo "chunk rc=$?"
unset DS4F_PREFILL_CHUNK DS4F_PREFILL_B DS4F_NAN_PROBE
mv /tmp/q35-eng-L*-t*.bin /tmp/ld2-chunk/ 2>/dev/null
echo "=== per-token L<4 diff ==="
python3 tools/ld2-diff.py /tmp/ld2-serial /tmp/ld2-chunk
echo "=== timing ==="
grep 'tokens in' /tmp/ld2-serial.log /tmp/ld2-chunk.log
