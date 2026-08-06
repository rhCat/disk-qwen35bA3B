#!/usr/bin/env python3
"""split-mlx-switchmlp.py -- build the disk engine's pool from the MLX
4-bit Qwen3.5 repo, WITHOUT ever loading the model into RAM.

MLX 4-bit flattens the 256 routed experts into per-layer tensors:

    layers.N.mlp.switch_mlp.{gate,up,down}_proj.weight  U32 [256, R, C]
    layers.N.mlp.switch_mlp.{gate,up,down}_proj.scales  BF16 [256, R, S]
    layers.N.mlp.switch_mlp.{gate,up,down}_proj.biases  BF16 [256, R, S]

The disk engine's pool is per-expert fixed-rate: expert (L,E) at
base + (L*E_total + E)*nbytes, O(1). This tool streams each expert's
9 tensor slices (3 proj x weight/scales/biases) from the shard files
into that layout -- the machine never holds more than a few MB.

Output (the ds4f layout):
    config.json      engine-format config (n_layers, n_experts, topk,
                     n_shared, hidden, latent, moe_inter, expert_nbytes)
    pool.bin         [u64 expert_nbytes][u64 n_layers][u64 n_experts]
                     then expert (0,0), (0,1), ... (L,E) contiguous
    manifest.json    per-expert tensor map (name, off, nbytes)

usage:
  python3 tools/split-mlx-switchmlp.py DIR --out OUT
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


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args or "--out" not in sys.argv:
        print(__doc__)
        sys.exit(1)
    d = args[0]
    out = sys.argv[sys.argv.index("--out") + 1]

    idx_path = os.path.join(d, "model.safetensors.index.json")
    with open(idx_path) as f:
        wm = json.load(f)["weight_map"]

    # ---- inventory switch_mlp tensors per layer
    layers = {}
    for name, fn in wm.items():
        if "mlp.switch_mlp." not in name:
            continue
        head, proj = name.rsplit("mlp.switch_mlp.", 1)
        if "shared" in head:
            continue                      # shared_expert stays dense
        m = re.search(r"layers\.(\d+)\.$", head)
        if not m:
            continue
        L = int(m.group(1))
        layers.setdefault(L, {})[proj] = name
    if not layers:
        print("REFUSE: no switch_mlp tensors (is this the MLX 4-bit repo?)")
        sys.exit(1)
    n_layers = len(layers)
    layer_ids = sorted(layers)

    # ---- discover E, R, C, and per-expert payload size from one layer
    def tensor_meta(layer, proj, suffix):
        nm = layers[layer][f"{proj}.{suffix}"]
        fn = wm[nm]
        hdr, pb = load_header(os.path.join(d, fn))
        m = hdr[nm]
        a, b = m["data_offsets"]
        return nm, fn, pb + a, b - a, m["dtype"], list(m["shape"])

    proj_names = ["gate_proj", "up_proj", "down_proj"]
    L0 = layer_ids[0]
    _, _, _, _, dt0, shp0 = tensor_meta(L0, "gate_proj", "weight")
    E, R, C = shp0
    print(f"switch_mlp: {n_layers} layers x {E} experts, gate_proj [{R}x{C}] {dt0}")
    if dt0 != "U32":
        print(f"REFUSE: expected packed U32 4-bit, got {dt0}")
        sys.exit(1)

    # per-expert record = 3 proj x (weight + scales + biases), all
    # [E, :, :] sliceable except shared (excluded). weight is U32 with
    # 4-bit pairs packed 2/byte -> 4 bytes per 8 values; bytes per
    # expert = R*C*4 (the U32 element count is R*C, one per 8 values).
    weight_eb = (R * C) * 4
    scales_eb = None
    for suffix in ("scales", "biases"):
        nm, fn, off, nb, dt, shp = tensor_meta(L0, "gate_proj", suffix)
        assert shp[0] == E, f"{suffix} not per-expert: {shp}"
        if suffix == "scales":
            scales_eb = nb // E
    biases_eb = None
    for suffix in ("scales", "biases"):
        nm, fn, off, nb, dt, shp = tensor_meta(L0, "gate_proj", suffix)
        if suffix == "biases":
            biases_eb = nb // E
    expert_bytes = (weight_eb + scales_eb + biases_eb) * 3
    print(f"per-expert: weight {hsize(weight_eb)} + scales {hsize(scales_eb)}"
          f" + biases {hsize(biases_eb)} x3 proj = {hsize(expert_bytes)}")
    print(f"pool total: {hsize(expert_bytes * E * n_layers)}")

    # ---- stream slices into pool.bin
    os.makedirs(out, exist_ok=True)
    src_open = {}
    def reader(fn):
        if fn not in src_open:
            src_open[fn] = open(os.path.join(d, fn), "rb")
        return src_open[fn]

    pool_path = os.path.join(out, "pool.bin")
    with open(pool_path, "wb") as pf:
        pf.write(struct.pack("<QQQ", expert_bytes, n_layers, E))
        manifest = {"format": "mlx4-split-v1", "n_layers": n_layers,
                    "n_experts": E, "expert_nbytes": expert_bytes,
                    "tensors": []}
        buf = bytearray(1 << 20)
        pos = 24
        tcount = 0
        for L in layer_ids:
            for e in range(E):
                slot_off = 24 + (L * E + e) * expert_bytes
                for proj in proj_names:
                    # values (weight), scales, biases -> one tensor entry
                    # each; the engine's chain runs them in order so the
                    # proj triplet must be contiguous in slot order.
                    ent = {"layer": L, "expert": e,
                           "shape": None, "fmt": 1,
                           "v_off": None, "s_off": None, "b_off": None,
                           "v_nbytes": None, "s_nbytes": None,
                           "b_nbytes": None, "name": proj}
                    for suffix in ("weight", "scales", "biases"):
                        nm, fn, off, nb, dt, shp = tensor_meta(L, proj, suffix)
                        eb = nb // E
                        f = reader(fn)
                        f.seek(off + e * eb)
                        left = eb
                        while left:
                            got = f.read(min(len(buf), left))
                            if not got:
                                raise RuntimeError("short read")
                            pf.write(got)
                            left -= len(got)
                        if suffix == "weight":
                            # decoded dims: U32 packs 8 nibbles per word,
                            # so cols = packed_cols * 8 (the engine's
                            # matvec works on decoded [R, C]).
                            ent["shape"] = [shp[1], shp[2] * 8]
                            ent["v_off"] = pos
                            ent["v_nbytes"] = eb
                        elif suffix == "scales":
                            ent["s_off"] = pos
                            ent["s_nbytes"] = eb
                        else:
                            ent["b_off"] = pos
                            ent["b_nbytes"] = eb
                        pos += eb
                    manifest["tensors"].append(ent)
                    tcount += 1
            if int(L) % 8 == 0:
                print(f"  layer {L}/{n_layers-1} ...")
    for f in src_open.values():
        f.close()
    print(f"  {tcount} tensor entries ({E} experts x {len(proj_names)} proj x "
          f"{n_layers} layers)")

    # ---- config.json in engine format
    src_cfg = json.load(open(os.path.join(d, "config.json")))
    tc = src_cfg.get("text_config", src_cfg)
    n_shared = 1 if tc.get("shared_expert_intermediate_size") else 0
    cfg = {
        "n_layers": n_layers,
        "n_experts": E,
        "topk": int(tc.get("num_experts_per_tok", 8)),
        "n_shared": n_shared,
        "hidden": int(tc.get("hidden_size")),
        "latent": int(tc.get("hidden_size")),   # GQA: no latent
        "moe_inter": int(tc.get("moe_intermediate_size")),
        "expert_nbytes": expert_bytes,
        "num_key_value_heads": int(tc.get("num_key_value_heads", 2)),
        "head_dim": int(tc.get("head_dim", 256)),
        "rope_theta": tc.get("rope_parameters", {}).get("rope_theta", 1e7),
        "seed": 7,
    }
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    sz = os.path.getsize(pool_path)
    assert sz == 24 + expert_bytes * E * n_layers, f"size mismatch {sz}"
    print(f"pool.bin {hsize(sz)} OK (24 hdr + {E}x{n_layers} x {hsize(expert_bytes)})")
    print(f"config.json written (topk {cfg['topk']}, n_shared {n_shared})")
    print(f"run: ./ds4f {out} --trunk ... --offsets ... --pool {pool_path}")


if __name__ == "__main__":
    main()
