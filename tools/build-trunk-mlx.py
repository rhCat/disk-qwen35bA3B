#!/usr/bin/env python3
"""build-trunk-mlx.py -- build the engine's trunk (dense resident set)
from the MLX 4-bit Qwen3.5 repo, streaming, never loading the model.

The trunk is the ~3 GB resident set: attention (10 full-GQA layers +
30 linear-attn layers), router gate, shared expert, layernorms -- the
tensors active on EVERY token. The 256-expert switch_mlp pool stays on
disk (split-mlx-switchmlp.py). Vision tower + MTP are stripped.

Output (engine packed format):
    trunk.bin        dense per-layer bytes, contiguous
    trunk.offsets    [u64 n][u64 off x n][u64 size x n]
    trunk.json       per-layer tensor map (name, dtype, shape, off, nbytes)
    config.json      engine-format config (reuses pool config if present)
    embed.bin/json   embed_tokens (MLX triplet -> values only)
    head.bin/json    lm_head

usage:
  python3 tools/build-trunk-mlx.py DIR --out OUT [--pool-config DIR2]
"""
import json
import os
import re
import struct
import sys


def hsize(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


def load_header(path):
    with open(path, "rb") as f:
        raw = f.read(8)
        hlen = struct.unpack("<Q", raw)[0]
        hdr = json.loads(f.read(hlen))
    return hdr, 8 + hlen


def put_u64(f, v):
    f.write(struct.pack("<Q", v))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args or "--out" not in sys.argv:
        print(__doc__)
        sys.exit(1)
    d = args[0]
    out = sys.argv[sys.argv.index("--out") + 1]
    pool_cfg_dir = None
    if "--pool-config" in sys.argv:
        pool_cfg_dir = sys.argv[sys.argv.index("--pool-config") + 1]

    with open(os.path.join(d, "model.safetensors.index.json")) as f:
        wm = json.load(f)["weight_map"]

    # shard header cache
    shards = {}
    for fn in set(wm.values()):
        shards[fn] = load_header(os.path.join(d, fn))

    def span(name):
        fn = wm[name]
        h, pb = shards[fn]
        a, b = h[name]["data_offsets"]
        return fn, pb + a, b - a, h[name]["dtype"], list(h[name]["shape"])

    # ---- collect per-layer dense tensors (exclude pool + vision + MTP)
    SKIP = ("switch_mlp", "vision", "mtp", "shared_expert",
            "embed_tokens", "lm_head", "language_model.norm")
    layers = {}
    for name in wm:
        m = re.search(r"layers\.(\d+)\.", name)
        if not m:
            continue
        if any(s in name for s in SKIP):
            continue
        L = int(m.group(1))
        layers.setdefault(L, []).append(name)
    if not layers:
        print("REFUSE: no dense layer tensors found")
        sys.exit(1)
    layer_ids = sorted(layers)
    print(f"trunk: {len(layer_ids)} layers, "
          f"{sum(len(v) for v in layers.values())} tensors")

    os.makedirs(out, exist_ok=True)

    # ---- copy pool config if given (n_layers/n_experts/topk match)
    if pool_cfg_dir and os.path.exists(os.path.join(pool_cfg_dir, "config.json")):
        import shutil
        shutil.copy(os.path.join(pool_cfg_dir, "config.json"),
                    os.path.join(out, "config.json"))
        print("config.json: copied from pool dir")

    # ---- trunk.bin + trunk.offsets
    src_open = {}
    def reader(fn):
        if fn not in src_open:
            src_open[fn] = open(os.path.join(d, fn), "rb")
        return src_open[fn]

    tbin = open(os.path.join(out, "trunk.bin"), "wb")
    toff = open(os.path.join(out, "trunk.offsets"), "wb")
    put_u64(toff, len(layer_ids))
    buf = bytearray(1 << 20)
    at = 0
    trunk_json = {"n_layers": len(layer_ids), "layers": []}
    align = 8
    for L in layer_ids:
        put_u64(toff, at)
        lay_bytes = 0
        ltens = []
        for name in sorted(layers[L]):
            fn, off, nb, dt, shp = span(name)
            pad = (-lay_bytes) % align
            if pad:
                tbin.write(b"\0" * pad)
                lay_bytes += pad
            f = reader(fn)
            f.seek(off)
            left = nb
            while left:
                got = f.read(min(len(buf), left))
                if not got:
                    raise RuntimeError("short read")
                tbin.write(got)
                left -= len(got)
            ltens.append({"n": name, "dtype": dt, "shape": shp,
                          "off": lay_bytes, "nbytes": nb})
            lay_bytes += nb
        put_u64(toff, lay_bytes)
        trunk_json["layers"].append({"layer": L, "tensors": ltens})
        at += lay_bytes
        if L + 1 < len(layer_ids):
            pad = (-at) % align
            if pad:
                tbin.write(b"\0" * pad)
                at += pad
    tbin.close()
    toff.close()
    for f in src_open.values():
        f.close()
    with open(os.path.join(out, "trunk.json"), "w") as f:
        json.dump(trunk_json, f, indent=1)
    print(f"trunk.bin {hsize(at)} ({len(layer_ids)} layers) "
          f"-> {hsize(at / len(layer_ids))}/layer")

    # ---- embed + head (MLX triplet: weight + scales + biases)
    def write_pair(tag, names):
        cand = next((n for n in names if n in wm), None)
        if not cand:
            print(f"{tag}: NOT FOUND (names: {names[0]})")
            return
        fn, off, nb, dt, shp = span(cand)
        out_path = os.path.join(out, f"{tag}.bin")
        with open(out_path, "wb") as of:
            with open(os.path.join(d, fn), "rb") as sf:
                sf.seek(off)
                left = nb
                while left:
                    got = sf.read(min(len(buf), left))
                    if not got:
                        raise RuntimeError("short read")
                    of.write(got)
                    left -= len(got)
        meta = {"bin": f"{tag}.bin", "dtype": dt, "shape": shp,
                "nbytes": nb, "scale": None}
        # MLX triplet: append scales + biases (byte offsets recorded)
        for suffix, key in ((".scales", "scale"), (".biases", "bias")):
            sc = cand + suffix
            if sc in wm:
                fn2, off2, nb2, dt2, shp2 = span(sc)
                with open(out_path, "ab") as of:
                    with open(os.path.join(d, fn2), "rb") as sf:
                        sf.seek(off2)
                        left = nb2
                        while left:
                            got = sf.read(min(len(buf), left))
                            if not got:
                                raise RuntimeError("short read")
                            of.write(got)
                            left -= len(got)
                meta[key] = {"off": nb, "nbytes": nb2, "dtype": dt2,
                             "shape": shp2}
                nb += nb2
        with open(os.path.join(out, f"{tag}.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(f"{tag}.bin {hsize(nb)} {dt} {shp}")

    write_pair("embed", ["language_model.model.embed_tokens.weight"])
    write_pair("head", ["language_model.lm_head.weight"])

    print("run:")
    print(f"  ./ds4f {out} --trunk {out}/trunk.bin --offsets {out}/trunk.offsets"
          f" --pool {out}/../q35-pool/pool.bin --layout-trunk {out}/trunk.json"
          f" --layout-pool {out}/../q35-pool/manifest.json")


if __name__ == "__main__":
    main()
