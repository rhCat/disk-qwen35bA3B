#!/usr/bin/env python3
"""qwen3-arch.py -- add Qwen3-30B-A3B support to ds4f-disk's converter.

The engine's DS-V4-Flash converter is *one regex away* from Qwen3: the
Qwen3 naming (`model.layers.N.mlp.experts.E.gate_proj.weight`) already
fits EXPERT_RE, `self_attn.*_proj` fits the dense LAYER_RE, and
`model.embed_tokens.weight` / `lm_head.weight` match the embed/head
candidates. The converter REFUSES for exactly two reasons:

  1. `n_shared` is REQUIRED -- Qwen3 has NO shared experts.
  2. The GQA attention geometry (num_key_value_heads, head_dim) is
     dropped, so the engine would fall into the MLA kvhalf path.

This script applies the minimal diff to tools/convert-ds4f.py:

  - `n_shared` becomes optional: absent -> 0 (refuse stays for the
    DS-V4 path, where it is always present)
  - `num_key_value_heads` and `head_dim` are carried into config.json
    (plus the existing num_attention_heads / rope_theta)
  - `make-synthetic` gains `--arch qwen3` so the fixture gate can
    exercise the Qwen3 layout without real weights

Idempotent: safe to run twice. Applies in place; the original is
backed up to convert-ds4f.py.pre-qwen3 on first run.

usage:  python3 tools/qwen3-arch.py            (apply the diff)
"""

import os
import sys
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
CONV = os.path.join(HERE, "convert-ds4f.py")

# ---------------------------------------------------------------- diffs
# 1. ALIASES: add the GQA geometry keys
OLD_ALIASES = '''    "moe_inter": ["moe_intermediate_size", "intermediate_size",
                  "ffn_hidden_size", "moe_ffn_hidden_size"],
}'''
NEW_ALIASES = '''    "moe_inter": ["moe_intermediate_size", "intermediate_size",
                  "ffn_hidden_size", "moe_ffn_hidden_size"],
    "n_kv_heads": ["num_key_value_heads", "n_kv_heads"],
    "head_dim": ["head_dim", "kv_channels"],
}'''

# 2. map_config: n_shared optional (Qwen3 has none); head_dim defaults to
#    hidden // n_kv_heads (standard GQA) and is reported, not silent
OLD_MAP = '''def map_config(config):
    mapped = {}
    assumptions = []
    for key, aliases in ALIASES.items():
        src = next((a for a in aliases if a in config), None)
        if src is not None:
            mapped[key] = int(config[src])
        elif key in REQUIRED:
            mapped[key] = None
        else:
            assumptions.append(key)
    if "latent" in assumptions and mapped.get("hidden"):
        mapped["latent"] = mapped["hidden"]
        assumptions.remove("latent")
    if "moe_inter" in assumptions and mapped.get("hidden"):
        mapped["moe_inter"] = mapped["hidden"]
        assumptions.remove("moe_inter")
    return mapped, assumptions'''
NEW_MAP = '''def map_config(config):
    mapped = {}
    assumptions = []
    for key, aliases in ALIASES.items():
        src = next((a for a in aliases if a in config), None)
        if src is not None:
            mapped[key] = int(config[src])
        elif key in REQUIRED:
            mapped[key] = None
        else:
            assumptions.append(key)
    if "latent" in assumptions and mapped.get("hidden"):
        mapped["latent"] = mapped["hidden"]
        assumptions.remove("latent")
    if "moe_inter" in assumptions and mapped.get("hidden"):
        mapped["moe_inter"] = mapped["hidden"]
        assumptions.remove("moe_inter")
    # Qwen3-class MoE: no shared experts. Absent n_shared -> 0 (the
    # DS-V4 path always has it, so a real miss still refuses below).
    if "n_shared" in assumptions:
        mapped["n_shared"] = 0
        assumptions.remove("n_shared")
    # GQA geometry: head_dim defaults to hidden // kv_heads (reported,
    # not silent -- cfg.c gates on n_kv_heads > 0 to pick GQA).
    if mapped.get("n_kv_heads") and not mapped.get("head_dim"):
        mapped["head_dim"] = mapped["hidden"] // mapped["n_kv_heads"]
        assumptions.append("head_dim")
    return mapped, assumptions'''

# 3. cmd_convert: emit the GQA keys into config.json
OLD_CFG = '''    # the real MLA geometry (optional; the engine falls back without it)
    for src_key in ("num_attention_heads", "qk_rope_head_dim"):
        if config is not None and src_key in config:
            cfg_out[src_key] = int(config[src_key])'''
NEW_CFG = '''    # the real attention geometry (optional; the engine falls back)
    for src_key, out_key in (("num_attention_heads", "num_attention_heads"),
                             ("qk_rope_head_dim", "qk_rope_head_dim"),
                             ("num_key_value_heads", "num_key_value_heads"),
                             ("head_dim", "head_dim")):
        if config is not None and src_key in config:
            cfg_out[out_key] = int(config[src_key])
    if mapped.get("n_kv_heads") and "num_key_value_heads" not in cfg_out:
        cfg_out["num_key_value_heads"] = mapped["n_kv_heads"]
    if mapped.get("head_dim") and "head_dim" not in cfg_out:
        cfg_out["head_dim"] = mapped["head_dim"]'''

# 4. make-synthetic: --arch qwen3 fixture mode
OLD_SYN = '''def cmd_make_synthetic(dirpath):'''
NEW_SYN = '''def cmd_make_synthetic_qwen3(dirpath):
    """Tiny Qwen3-MoE-shaped HF repo (2 layers, 4 experts, top-2) so the
    converter + fixture gate exercise the real Qwen3 layout without real
    weights: self_attn q/k/v/o, mlp.experts gate/up/down, layernorms,
    model.norm, embed_tokens, lm_head. No shared experts, no scales."""
    os.makedirs(dirpath, exist_ok=True)
    cfg = {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "num_hidden_layers": 2,
        "hidden_size": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 16,
        "intermediate_size": 16,
        "rms_norm_eps": 1e-06,
        "rope_theta": 10000,
        "max_position_embeddings": 128,
        "vocab_size": 64,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
    }
    with open(os.path.join(dirpath, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    names = ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"]
    for L in range(2):
        names += [
            f"model.layers.{L}.input_layernorm.weight",
            f"model.layers.{L}.post_attention_layernorm.weight",
            f"model.layers.{L}.self_attn.q_proj.weight",
            f"model.layers.{L}.self_attn.k_proj.weight",
            f"model.layers.{L}.self_attn.v_proj.weight",
            f"model.layers.{L}.self_attn.o_proj.weight",
        ]
        for e in range(4):
            names += [
                f"model.layers.{L}.mlp.experts.{e}.gate_proj.weight",
                f"model.layers.{L}.mlp.experts.{e}.up_proj.weight",
                f"model.layers.{L}.mlp.experts.{e}.down_proj.weight",
            ]

    def blob(n, nbytes, dtype):
        if dtype == "BF16":
            out = bytearray()
            x = 0x1234
            for i in range(nbytes // 2):
                x = (x * 1103515245 + 12345) & 0x7FFFFFFF
                v = ((x & 0xFFFF) / 65535.0 - 0.5) * 2.0
                f32 = struct.unpack("<I", struct.pack("<f", v))[0]
                out += struct.pack("<H", (f32 >> 16) & 0xFFFF)
            return bytes(out)
        out = bytearray()
        x = 0x1234
        for i in range(nbytes):
            x = (x * 1103515245 + 12345) & 0x7FFFFFFF
            out.append((x + ord(n[0])) & 0xFF)
        return bytes(out)

    # 2-D shapes with real dims so the matvec chain has something to do
    shape_of = {}
    H, E, MI = 8, 4, 16
    shape_of["model.embed_tokens.weight"] = [64, H]
    shape_of["lm_head.weight"] = [64, H]
    shape_of["model.norm.weight"] = [H]
    for L in range(2):
        shape_of[f"model.layers.{L}.input_layernorm.weight"] = [H]
        shape_of[f"model.layers.{L}.post_attention_layernorm.weight"] = [H]
        shape_of[f"model.layers.{L}.self_attn.q_proj.weight"] = [H, H]
        shape_of[f"model.layers.{L}.self_attn.k_proj.weight"] = [H // 2, H]
        shape_of[f"model.layers.{L}.self_attn.v_proj.weight"] = [H // 2, H]
        shape_of[f"model.layers.{L}.self_attn.o_proj.weight"] = [H, H]
        for e in range(E):
            shape_of[f"model.layers.{L}.mlp.experts.{e}.gate_proj.weight"] = [MI, H]
            shape_of[f"model.layers.{L}.mlp.experts.{e}.up_proj.weight"] = [MI, H]
            shape_of[f"model.layers.{L}.mlp.experts.{e}.down_proj.weight"] = [H, MI]

    # one shard, header + payload, each tensor 8-aligned
    hdr = {}
    off = 8
    payload = bytearray()
    for n in names:
        shp = shape_of[n]
        r = shp[0]
        c = shp[1] if len(shp) == 2 else 1
        nb = r * c * 2          # all BF16, 2 bytes/elem
        hdr[n] = {"dtype": "BF16", "shape": list(shp),
                  "data_offsets": [off, off + nb]}
        payload += blob(n, nb, "BF16")
        off += nb
    hdrj = json.dumps(hdr).encode()
    with open(os.path.join(dirpath, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(hdrj)))
        f.write(hdrj)
        f.write(bytes(payload))
    print(f"synthetic Qwen3 repo written to {dirpath} "
          f"({len(names)} tensors, {off} B)")


def cmd_make_synthetic(dirpath):'''
OLD_MAIN = '''    elif cmd == "make-synthetic":
        cmd_make_synthetic(sys.argv[2])'''
NEW_MAIN = '''    elif cmd == "make-synthetic":
        if "--arch" in sys.argv:
            arch = sys.argv[sys.argv.index("--arch") + 1]
            if arch == "qwen3":
                cmd_make_synthetic_qwen3(sys.argv[2])
                sys.exit(0)
        cmd_make_synthetic(sys.argv[2])'''


def main():
    if not os.path.exists(CONV):
        print(f"REFUSE: {CONV} not found (run from the repo root's tools/)")
        sys.exit(1)
    src = open(CONV).read()

    applied = 0
    for old, new, tag in ((OLD_ALIASES, NEW_ALIASES, "aliases"),
                          (OLD_MAP, NEW_MAP, "map_config"),
                          (OLD_CFG, NEW_CFG, "config-emit"),
                          (OLD_SYN, NEW_SYN, "synthetic-qwen3"),
                          (OLD_MAIN, NEW_MAIN, "main-dispatch")):
        if old in src:
            src = src.replace(old, new)
            applied += 1
            print(f"  applied: {tag}")
        else:
            print(f"  SKIP (already applied or moved): {tag}")

    if applied == 0:
        print("no changes needed -- qwen3 support already present")
        return
    shutil.copy(CONV, CONV + ".pre-qwen3")
    with open(CONV, "w") as f:
        f.write(src)
    print(f"wrote {CONV} (backup: convert-ds4f.py.pre-qwen3)")
    print("verify: python3 tools/convert-ds4f.py self-test")


if __name__ == "__main__":
    main()
