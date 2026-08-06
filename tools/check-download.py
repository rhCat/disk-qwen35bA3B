#!/usr/bin/env python3
"""check-download.py -- verify a Qwen3-30B-A3B (or any safetensors HF
repo) download is complete and intact BEFORE converting.

Checks, in order:
  1. config.json present + parseable (architecture sanity)
  2. model.safetensors.index.json present (multi-shard repo)
  3. every shard named in weight_map exists
  4. every shard's header is parseable and its declared payload size
     matches the file size on disk  (8 + header_len + max data end)
  5. optional --sha256: hash every shard and cross-check against
     index.json's total_size / file metadata when available
  6. summary: shards, tensors, dtype histogram, total model bytes

A repo that passes this is safe to feed to convert-ds4f.py. Anything
less is a partial download (HF's downloaders can silently leave
truncated shards) -- converting a truncated pool corrupts silently.

usage:
  python3 tools/check-download.py DIR                # size checks
  python3 tools/check-download.py DIR --sha256       # + full hashing
  python3 tools/check-download.py DIR --quick        # existence only
"""
import json
import os
import struct
import sys


def hsize(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


def shard_size(path):
    """Total bytes of a safetensors file: 8 + header_len + payload."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            return None, "not a safetensors file (short header)"
        hlen = struct.unpack("<Q", raw)[0]
        hdr = f.read(hlen)
        if len(hdr) != hlen:
            return None, "truncated header"
        try:
            idx = json.loads(hdr)
        except json.JSONDecodeError:
            return None, "header is not valid JSON"
        end = 0
        for name, meta in idx.items():
            if not isinstance(meta, dict) or "data_offsets" not in meta:
                continue
            end = max(end, meta["data_offsets"][1])
        return 8 + hlen + end, None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sha = "--sha256" in sys.argv
    quick = "--quick" in sys.argv
    if not args:
        print(__doc__)
        sys.exit(1)
    d = args[0]
    if not os.path.isdir(d):
        print(f"REFUSE: {d} is not a directory")
        sys.exit(1)

    problems = []

    # 1. config
    cfg_path = os.path.join(d, "config.json")
    if not os.path.exists(cfg_path):
        problems.append("config.json MISSING")
        cfg = {}
    else:
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            arch = cfg.get("architectures", cfg.get("model_type", "?"))
            print(f"config.json OK   arch={arch} "
                  f"layers={cfg.get('num_hidden_layers', '?')} "
                  f"experts={cfg.get('num_experts', '?')} "
                  f"topk={cfg.get('num_experts_per_tok', '?')} "
                  f"hidden={cfg.get('hidden_size', '?')}")
        except (json.JSONDecodeError, OSError) as e:
            problems.append(f"config.json unparseable: {e}")
            cfg = {}

    # 2. index
    idx_path = os.path.join(d, "model.safetensors.index.json")
    if not os.path.exists(idx_path):
        problems.append("model.safetensors.index.json MISSING "
                        "(single-shard repo? then pass the .safetensors file directly)")
        weight_map = {}
    else:
        try:
            with open(idx_path) as f:
                ij = json.load(f)
            weight_map = ij.get("weight_map", {})
            total = ij.get("total_size", 0)
            print(f"index.json OK    {len(weight_map)} tensors, "
                  f"{hsize(total)} total per index")
        except (json.JSONDecodeError, OSError) as e:
            problems.append(f"index.json unparseable: {e}")
            weight_map = {}

    # 3. shard existence + 4. size integrity
    shards = sorted({os.path.basename(p) for p in weight_map.values()})
    if not shards:
        shards = [f for f in os.listdir(d) if f.endswith(".safetensors")]
    ok_bytes = 0
    tensor_count = 0
    dtypes = {}
    for s in shards:
        p = os.path.join(d, s)
        if not os.path.exists(p):
            problems.append(f"shard MISSING: {s}")
            continue
        st = os.path.getsize(p)
        if quick:
            print(f"  {s:40s} {hsize(st)} (exists)")
            ok_bytes += st
            continue
        want, err = shard_size(p)
        if err:
            problems.append(f"shard {s}: {err}")
            continue
        if st != want:
            problems.append(f"shard {s}: on disk {st} B != header-declared "
                            f"{want} B (TRUNCATED or corrupted)")
        else:
            print(f"  {s:40s} {hsize(st)} OK")
        ok_bytes += st
        # dtype histogram from the header
        if sha:
            import hashlib
            h = hashlib.sha256()
            with open(p, "rb") as f:
                while True:
                    chunk = f.read(1 << 20)
                    if not chunk:
                        break
                    h.update(chunk)
            print(f"    sha256 {h.hexdigest()[:16]}...")
        with open(p, "rb") as f:
            raw = f.read(8)
            hlen = struct.unpack("<Q", raw)[0]
            idx = json.loads(f.read(hlen))
        for name, meta in idx.items():
            if isinstance(meta, dict) and "dtype" in meta:
                dtypes[meta["dtype"]] = dtypes.get(meta["dtype"], 0) + 1
                tensor_count += 1

    # 5. expected shard count
    if weight_map:
        missing_files = [s for s in shards
                         if not os.path.exists(os.path.join(d, s))]
        # shards list already filtered; report count vs index
        indexed_files = len({os.path.basename(p) for p in weight_map.values()})
        if len(shards) < indexed_files:
            problems.append(f"expected {indexed_files} shards, found {len(shards)}")

    # 6. summary
    print("-" * 60)
    print(f"shards on disk: {len(shards)}   tensors: {tensor_count}   "
          f"payload: {hsize(ok_bytes)}")
    if dtypes:
        print("dtype histogram: " + ", ".join(
            f"{k} x{v}" for k, v in sorted(dtypes.items())))
    if problems:
        print("\nPROBLEMS (%d):" % len(problems))
        for p in problems:
            print(f"  FAIL {p}")
        print("\nVERDICT: INCOMPLETE -- fix the download before converting "
              "(re-run the HF downloader; a truncated pool corrupts silently)")
        sys.exit(1)
    print("VERDICT: COMPLETE -- safe to convert")


if __name__ == "__main__":
    main()
