#!/usr/bin/env python3
# bench-grade.py -- grade the bench: extract the generated text from each
# log, check the expected fact, print a table.
import json, re

meta = json.load(open("/tmp/q35-bench-meta.json"))

def gen_text(log):
    """Extract the streamed generation (after the last newline-prefixed
    marker) from the log: the decoded text follows '--- run report ---'
    lines are stderr; stdout text appears before the report."""
    lines = open(log, encoding="utf-8", errors="replace").read()
    # stdout text = everything after the last 'QUESTION:' is not in log;
    # instead: text is printed with printf("%s") before the report block.
    # The report starts with '--- run report ---'. Take text between the
    # startup banner and the report, stripping stderr lines.
    m = re.search(r"router: real matvec.*?\n(.*?)\n--- run report ---", lines, re.S)
    if m:
        return m.group(1).strip()
    return ""

print(f"{'scenario':<16} {'ctx':>6} {'gen':>4} {'dec sp/t':>8} {'peakGB':>7}  answer-graded")
for name, mm in meta.items():
    log = f"/tmp/bench-{name}.log"
    try:
        txt = open(log, encoding="utf-8", errors="replace").read()
    except FileNotFoundError:
        print(f"{name:<16} NOT RUN")
        continue
    tm = re.search(r"(\d+) tokens in ([\d.]+) s, ([\d.]+) s/token", txt)
    pk = re.search(r"PEAK RSS: ([\d.]+) GB", txt)
    gen_n = int(tm.group(1)) if tm else -1
    dt = float(tm.group(2)) if tm else 0
    spt = float(tm.group(3)) if tm else 0
    peak = pk.group(1) if pk else "?"
    out = gen_text(log)
    ans = mm["answer"]
    if ans == "NOT IN DOCUMENT":
        # control: correct = refuses to fabricate
        low = out.lower()
        ok = ("not" in low and any(w in low for w in
              ("specified", "mentioned", "provided", "found", "state", "given")))
        grade = "PASS(refuse)" if ok else "FAIL(fabricated?)"
    else:
        ok = ans in out
        grade = "PASS" if ok else "FAIL"
    print(f"{name:<16} {mm['prompt_tokens']:>6} {gen_n:>4} {spt:>8.2f} {peak:>7}  {grade}")
    print(f"    Q: {mm['question'][:70]}")
    print(f"    A: {out[:120]!r}")
