#!/usr/bin/env bash
# bench-4k.sh -- 4K-token QA long-context validation, GEN=8.
set -u
REPO=/Users/ruihe/disk-qwen35bA3B
cd "$REPO"
unset PYTHONPATH
export PATH="/Volumes/prod/miniforge3/envs/ca_lpp/bin:$PATH"
export DS4F_GREEDY=1 DS4F_REP_PENALTY=1.3 DS4F_DEBUG7=1
PY="$REPO/.venv/bin/python3"
t0=$("$PY" -c 'import time; print(time.time())')
bash tools/run-clean.sh ./ds4f /tmp/q35-trunk \
  --trunk /tmp/q35-trunk/trunk.bin --offsets /tmp/q35-trunk/trunk.offsets \
  --layout-trunk /tmp/q35-trunk/trunk.json \
  --pool /tmp/q35-pool/pool.bin --layout-pool /tmp/q35-pool/manifest.json \
  --head /tmp/q35-trunk/head.json --embed /tmp/q35-trunk/embed.json \
  --tokenizer /Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit/tokenizer.json \
  --pids-file /tmp/q35-4k-ids.txt --gen 8 \
  --cache-gb 5 --pin-layers 4 --mem-limit-gb 20 \
  > /tmp/bench-4k.log 2>&1
rc=$?
t1=$("$PY" -c 'import time; print(time.time())')
echo "EXIT $rc | wall $("$PY" -c "import time;print(f'{$t1-$t0:.1f}s')")"
grep -E 'tokens in|GB read|PEAK|cache:' /tmp/bench-4k.log | head -4
echo "--- generated tokens (DEBUG7 top1) ---"
"$PY" - <<'EOF' 2>/dev/null
import re
log = open("/tmp/bench-4k.log", encoding="utf-8", errors="replace").read()
npids = len(open("/tmp/q35-4k-ids.txt").read().split(","))
text = ""
for l in log.splitlines():
    m = re.search(r"logits: t(\d+) state_rms.*?top5\s+\[\s*\d+ [\d.]+ ([^\]]*)\]", l)
    if m and int(m.group(1)) >= npids:
        text += "\n" if m.group(2).strip() in ("Ċ", "<0x0A>") else m.group(2).strip()
print(repr(text[:200]))
EOF
