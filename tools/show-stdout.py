#!/usr/bin/env python3
# show-stdout.py -- print the engine's stdout (stripped) from conv logs.
import sys

for tag in ("q1", "q2", "q3"):
    log = open(f"/tmp/conv-{tag}.log", encoding="utf-8", errors="replace").read()
    out = []
    for line in log.splitlines():
        s = line.strip()
        if not s:
            continue
        if (s.startswith("[") or s.startswith("logits:") or s.startswith("moe:")
                or s.startswith("cache:") or s.startswith("trunk:")
                or s.startswith("config:") or s.startswith("pool:")
                or s.startswith("kernels:") or s.startswith("PEAK")
                or s.startswith("GB read") or s.startswith("---")
                or "EXIT" in s):
            continue
        out.append(line)
    print(f"=== {tag} stdout ===")
    print("".join(out)[:600])
    print()
