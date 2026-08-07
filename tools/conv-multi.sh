#!/usr/bin/env bash
# conv-multi.sh -- 3-turn conversation to test holding context.
# Turn 1: ask a fact. Turn 2: follow-up referencing turn 1. Turn 3:
# test whether the model remembers the earlier answer.
set -u
REPO=/Users/ruihe/disk-qwen35bA3B
cd "$REPO"
export DS4F_TOP_P=0.9 DS4F_TEMP=0.8 DS4F_REP_PENALTY=1.15 DS4F_DEBUG7=1
unset PYTHONPATH
PY="$REPO/.venv/bin/python3"
HFBASE="$(cd "$HOME/.cache" 2>/dev/null && pwd)/huggingface"

transcript=""
for spec in \
  "What is the capital of France?|q1" \
  "What is the capital of Japan?|q2" \
  "Now tell me the capitals of both France and Japan in one line.|q3"; do
  turn="${spec%|*}"
  tag="${spec##*|}"
  transcript="${transcript}<|im_start|>user
${turn}<|im_end|>
<|im_start|>assistant
"
  printf '%s' "$transcript" > /tmp/conv-prompt.txt
  "$PY" - "$HFBASE" <<'PYEOF' > /tmp/conv-ids.txt 2>/dev/null
import sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1] + "/mlx-qwen35-a3b-4bit")
ids = tok(open("/tmp/conv-prompt.txt").read())["input_ids"]
print(",".join(str(i) for i in ids))
PYEOF
  nids=$(wc -c < /tmp/conv-ids.txt)
  t0=$("$PY" -c 'import time; print(time.time())')
  ./ds4f /tmp/q35-trunk \
    --trunk /tmp/q35-trunk/trunk.bin --offsets /tmp/q35-trunk/trunk.offsets \
    --layout-trunk /tmp/q35-trunk/trunk.json \
    --pool /tmp/q35-pool/pool.bin --layout-pool /tmp/q35-pool/manifest.json \
    --head /tmp/q35-trunk/head.json --embed /tmp/q35-trunk/embed.json \
    --tokenizer "$HFBASE/mlx-qwen35-a3b-4bit/tokenizer.json" \
    --pids-file /tmp/conv-ids.txt --gen 128 \
    --cache-gb 5 --pin-layers 4 --mem-limit-gb 20 \
    > /tmp/conv-$tag.log 2>&1
  rc=$?
  t1=$("$PY" -c 'import time; print(time.time())')
  dt=$("$PY" -c "print(f'{$t1-$t0:.1f}s')")
  "$PY" - "$tag" <<'PYEOF' > /tmp/conv-reply.txt 2>/dev/null
import re, sys
log = open("/tmp/conv-" + sys.argv[1] + ".log", encoding="utf-8", errors="replace").read()
lines = [l for l in log.splitlines() if "logits: t" in l]
npids = len(open("/tmp/conv-ids.txt").read().split(","))
text = ""
for l in lines:
    m = re.search(r"logits: t(\d+) state_rms.*?top5\s+\[\s*\d+ [\d.]+ ([^\]]*)\]", l)
    if m and int(m.group(1)) >= npids:
        t = m.group(2).strip()
        text += "\n" if t in ("Ċ", "<0x0A>") else t
print(text)
PYEOF
  reply="$(cat /tmp/conv-reply.txt)"
  transcript="${transcript}${reply}"
  echo "--- $tag (${dt}, ${nids}B prompt) ---"
  echo "$reply" | head -3
done
echo "== conversation done =="
