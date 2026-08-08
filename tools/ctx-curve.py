#!/usr/bin/env python3
# ctx-curve.py -- build the context-scaling table from bench logs.
import re
import sys

runs = [
    # (label, log, prompt_tokens, gen, notes)
    ("5-tok",  "/tmp/wf8.log",           5,   80,  "post-attn-matrix"),
    ("500",    "/tmp/bench-ctx-q35-500-ids.log",   517, 8,   "post-attn-matrix"),
    ("1500",   "/tmp/bench-ctx-q35-1500-ids.log", 1532, 8,   "post-attn-matrix"),
    ("3000",   "/tmp/bench-ctx-q35-3000-ids.log", 3002, 8,   "post-attn-matrix"),
]
print(f"{'run':8s} {'tokens':>7s} {'s/token':>9s} {'RSS':>8s} {'attn':>6s} {'moe':>6s} {'fetch':>7s} {'head':>6s}")
for label, log, ptok, gen, note in runs:
    try:
        text = open(log, encoding="utf-8", errors="replace").read()
    except FileNotFoundError:
        print(f"{label:8s} (log missing)")
        continue
    m = re.search(r"(\d+) tokens in ([\d.]+) s", text)
    if not m:
        print(f"{label:8s} (no timing)")
        continue
    ntok, dt = int(m.group(1)), float(m.group(2))
    total = ptok + gen
    rate = dt / total
    mr = re.search(r"PEAK RSS: ([\d.]+) GB", text)
    rss = mr.group(1) if mr else "-"
    def ph(name):
        mm = re.search(rf"{name}:\s+([\d.]+) ms/token", text)
        return f"{float(mm.group(1)) / total:6.1f}" if mm else "  -  "
    print(f"{label:8s} {total:7d} {rate:9.3f} {rss:>8s} {ph('attn')} {ph('moe')} {ph('fetch')} {ph('head')}")
