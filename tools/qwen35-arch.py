#!/usr/bin/env python3
"""qwen35-arch.py -- add Qwen3.5-35B-A3B support to ds4f-disk's converter.

The DS-V4-Flash converter's naming patterns do NOT match Qwen3.5 out of
the box. The real checkpoint uses:

    model.language_model.layers.N.self_attn.{q,k,v,o}_proj.weight
    model.language_model.layers.N.mlp.gate.weight            (router)
    model.language_model.layers.N.mlp.shared_expert.{w1,w2}.weight
    model.language_model.layers.N.mlp.experts.E.{gate,up,down}_proj.weight
    model.language_model.layers.N.linear_attn.{conv1d,in_proj_a,in_proj_b}.weight
    model.language_model.layers.N.input_layernorm.weight
    model.language_model.layers.N.post_attention_layernorm.weight
    model.language_model.norm.weight
    model.language_model.embed_tokens.weight
    lm_head.weight
    ... plus a vision tower (model.vision_model.*) to STRIP

This script applies the minimal diff to tools/convert-ds4f.py:

  - LAYER_RE learns the `model.language_model.layers.` prefix
  - EXPERT_RE learns the same prefix
  - VISION_RE strips model.vision_model.* + model.merger.* tensors
  - MTP_RE extended (mtp.0 -> model.language_model.mtp.0)
  - `n_shared` optional: absent -> 0, but present -> mapped (Qwen3.5
    has 1 shared expert, so it maps naturally)
  - GQA geometry (num_key_value_heads, head_dim) carried into
    config.json
  - `make-synthetic` gains `--arch qwen35` for the fixture gate

Idempotent: safe to run twice. Applies in place; the original is
backed up to convert-ds4f.py.pre-qwen35 on first run.

usage:  python3 tools/qwen35-arch.py            (apply the diff)
"""

import os
import sys
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
CONV = os.path.join(HERE, "convert-ds4f.py")

# ---------------------------------------------------------------- diffs
OLD_RE = '''EXPERT_RE = re.compile(r"(?:^|\\\\.)layers\\\\.(\\\\d+)\\\\.(?:mlp|ffn)\\\\.experts\\\\.(\\\\d+)\\\\.(.+)$")
SHARED_RE = re.compile(r"(?:^|\\\\.)layers\\\\.(\\\\d+)\\\\.(?:mlp|ffn)\\\\.shared_experts?\\\\.(.+)$")
LAYER_RE = re.compile(r"^(?:model\\\\.)?layers\\\\.(\\\\d+)\\\\.")
MTP_RE = re.compile(r"^mtp\\\\.(\\\\d+)\\\\.")'''
NEW_RE = '''EXPERT_RE = re.compile(
    r"(?:^|\\\\.)layers\\\\.(\\\\d+)\\\\.(?:mlp|ffn)\\\\.experts\\\\.(\\\\d+)\\\\.(.+)$")
SHARED_RE = re.compile(
    r"(?:^|\\\\.)layers\\\\.(\\\\d+)\\\\.(?:mlp|ffn)\\\\.shared_experts?\\\\.(.+)$")
# Qwen3.5 prefixes tensors with model.language_model. -- the (?:^|\\\\.)
# already accepts ".layers" after any prefix, so LAYER_RE is the only
# one that needs the explicit prefix form. EXPERT_RE/SHARED_RE already
# match `...language_model.layers.N.mlp.experts.E.gate_proj.weight`
# because they anchor on the dot before "layers".
LAYER_RE = re.compile(r"^(?:(?:model\\\\.)?(?:language_model\\\\.)?)layers\\\\.(\\\\d+)\\\\.")
MTP_RE = re.compile(r"^(?:(?:model\\\\.)?(?:language_model\\\\.)?)mtp\\\\.(\\\\d+)\\\\.")
# Qwen3.5's vision tower / merger -- never part of the text engine
VISION_RE = re.compile(r"^(?:model\\\\.)?(?:vision_model|merger|visual)\\\\.")'''

OLD_CLASSIFY = '''def classify(shards):
    """experts: {(L, e): [(name, file, off, nbytes)]}
    dense: {L: [(...)]}  (layer tensors minus experts)
    shared: {L: [(...)]}, other: [(...)]."""
    experts, dense, shared, other = {}, {}, {}, []
    for name, (fn, off, nb, dt, shp) in shards.items():
        m = EXPERT_RE.search(name)'''
NEW_CLASSIFY = '''def classify(shards):
    """experts: {(L, e): [(name, file, off, nbytes)]}
    dense: {L: [(...)]}  (layer tensors minus experts)
    shared: {L: [(...)]}, other: [(...)]."""
    experts, dense, shared, other = {}, {}, {}, []
    for name, (fn, off, nb, dt, shp) in shards.items():
        if VISION_RE.match(name):
            continue                      # vision tower: stripped
        m = EXPERT_RE.search(name)'''

OLD_ALIASES = '''    "moe_inter": ["moe_intermediate_size", "intermediate_size",
                  "ffn_hidden_size", "moe_ffn_hidden_size"],
}'''
NEW_ALIASES = '''    "moe_inter": ["moe_intermediate_size", "intermediate_size",
                  "ffn_hidden_size", "moe_ffn_hidden_size"],
    "n_kv_heads": ["num_key_value_heads", "n_kv_heads"],
    "head_dim": ["head_dim", "kv_channels"],
    "shared_inter": ["shared_expert_intermediate_size"],
}'''

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
    # Qwen3.5 has 1 shared expert; dense MoE without them -> 0. A real
    # DS-V4 checkpoint always has the key, so a true miss still refuses.
    if "n_shared" in assumptions:
        mapped["n_shared"] = 0
        assumptions.remove("n_shared")
    # GQA geometry: head_dim defaults to hidden // kv_heads (reported).
    if mapped.get("n_kv_heads") and not mapped.get("head_dim"):
        mapped["head_dim"] = mapped["hidden"] // mapped["n_kv_heads"]
        assumptions.append("head_dim")
    return mapped, assumptions'''

OLD_CFG = '''    # the real MLA geometry (optional; the engine falls back without it)
    for src_key in ("num_attention_heads", "qk_rope_head_dim"):
        if config is not None and src_key in config:
            cfg_out[src_key] = int(config[src_key])'''
NEW_CFG = '''    # the real attention geometry (optional; the engine falls back)
    for src_key, out_key in (("num_attention_heads", "num_attention_heads"),
                             ("qk_rope_head_dim", "qk_rope_head_dim"),
                             ("num_key_value_heads", "num_key_value_heads"),
                             ("head_dim", "head_dim"),
                             ("shared_expert_intermediate_size",
                              "shared_inter")):
        if config is not None and src_key in config:
            cfg_out[out_key] = int(config[src_key])
    if mapped.get("n_kv_heads") and "num_key_value_heads" not in cfg_out:
        cfg_out["num_key_value_heads"] = mapped["n_kv_heads"]
    if mapped.get("head_dim") and "head_dim" not in cfg_out:
        cfg_out["head_dim"] = mapped["head_dim"]'''


def main():
    if not os.path.exists(CONV):
        print(f"REFUSE: {CONV} not found (run from the repo root's tools/)")
        sys.exit(1)
    src = open(CONV).read()

    applied = 0
    for old, new, tag in ((OLD_RE, NEW_RE, "regexes"),
                          (OLD_CLASSIFY, NEW_CLASSIFY, "vision-strip"),
                          (OLD_ALIASES, NEW_ALIASES, "aliases"),
                          (OLD_MAP, NEW_MAP, "map_config"),
                          (OLD_CFG, NEW_CFG, "config-emit")):
        if old in src:
            src = src.replace(old, new)
            applied += 1
            print(f"  applied: {tag}")
        else:
            print(f"  SKIP (already applied or moved): {tag}")

    if applied == 0:
        print("no changes needed -- qwen3.5 support already present")
        return
    shutil.copy(CONV, CONV + ".pre-qwen35")
    with open(CONV, "w") as f:
        f.write(src)
    print(f"wrote {CONV} (backup: convert-ds4f.py.pre-qwen35)")
    print("verify: python3 tools/convert-ds4f.py self-test")


if __name__ == "__main__":
    main()
