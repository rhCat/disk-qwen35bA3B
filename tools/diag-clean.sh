#!/usr/bin/env bash
# diag-clean.sh -- run with a scrubbed env, capture the failing path.
set -u
cd /Users/ruihe/disk-qwen35bA3B
unset PYTHONPATH
for v in DS4F_NAN_PROBE DS4F_GREEDY DS4F_DEBUG7 DS4F_PROJ_MS DS4F_WATERFALL \
         DS4F_PREFILL_CHUNK DS4F_PREFILL_B DS4F_STRIP_THINK DS4F_SKIP_ATTN \
         DS4F_TIME_LAYERS DS4F_DEBUG2 DS4F_DEBUG3 DS4F_DEBUG4 DS4F_DEBUG5 \
         DS4F_DEBUG6 DS4F_DEBUG10 DS4F_NO_NORMS DS4F_HEAD_RAW; do
  unset "$v"
done
bash tools/run-clean.sh ./ds4f /tmp/q35-trunk \
  --trunk /tmp/q35-trunk/trunk.bin --offsets /tmp/q35-trunk/trunk.offsets \
  --layout-trunk /tmp/q35-trunk/trunk.json \
  --pool /tmp/q35-pool/pool.bin --layout-pool /tmp/q35-pool/manifest.json \
  --head /tmp/q35-trunk/head.json --embed /tmp/q35-trunk/embed.json \
  --tokenizer "$HOME/.cache/huggingface/mlx-qwen35-a3b-4bit/tokenizer.json" \
  --pids-file /tmp/bench-gpu-ids.txt --gen 4 \
  --cache-gb 5 --pin-layers 4 --mem-limit-gb 20 \
  > /tmp/diag-clean.log 2>&1
echo "rc=$?"
grep -E 'attn step|moe step|trunk bind|\[lin\]|linbody|matvec2|FAIL' /tmp/diag-clean.log | head -10
