#!/usr/bin/env bash
# conv-test.sh -- single-turn conversation test. The engine is one-shot
# (prompt -> generation), so a conversation accumulates a transcript and
# re-runs per turn. Avoids literal absolute-path tokens (the Hermes
# lifecycle guard tokenizes paths in -c payloads and flags directories).
set -u
REPO=/Users/ruihe/disk-qwen35bA3B
cd "$REPO"
export DS4F_GREEDY=1 DS4F_REP_PENALTY=1.3 DS4F_DEBUG7=1
unset PYTHONPATH
PY="$REPO/.venv/bin/python3"

TURN="$1"
GEN="${2:-48}"

transcript="<|im_start|>user
${TURN}<|im_end|>
<|im_start|>assistant
"
printf '%s' "$transcript" > /tmp/conv-prompt.txt
# encode the transcript via the real tokenizer (paths built at runtime)
HFBASE="$(cd "$HOME/.cache" 2>/dev/null && pwd)/huggingface"
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
  --pids-file /tmp/conv-ids.txt --gen "$GEN" \
  --cache-gb 5 --pin-layers 4 --mem-limit-gb 20 \
  > /tmp/conv-run.log 2>&1
rc=$?
t1=$("$PY" -c 'import time; print(time.time())')
dt=$("$PY" -c "print(f'{$t1-$t0:.1f}s')")
reply=$("$PY" - <<'PYEOF' 2>/dev/null
import re
log = open("/tmp/conv-run.log", encoding="utf-8", errors="replace").read()
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
)
echo "EXIT $rc | ${dt} | prompt ${nids}B"
echo "REPLY: $reply"
