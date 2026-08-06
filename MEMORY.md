# disk-qwen35bA3B — converter arch diff

`tools/qwen3-arch.py` is a self-applying diff that adds Qwen3-30B-A3B
support to the ds4f-disk converter (`tools/convert-ds4f.py`). It is
idempotent and backs up the original to `convert-ds4f.py.pre-qwen3` on
first run.

## Why this diff (grounded in the real HF config)

The Qwen3 naming already fits the DS-V4 converter's regexes:
`model.layers.N.mlp.experts.E.gate_proj.weight` matches `EXPERT_RE`,
`self_attn.*_proj` fits the dense `LAYER_RE`, and
`model.embed_tokens.weight` / `lm_head.weight` match the embed/head
candidates. The converter REFUSED for exactly two reasons:

1. `n_shared` is REQUIRED — Qwen3 has **no shared experts**.
2. The GQA geometry (`num_key_value_heads`, `head_dim`) was dropped,
   so the engine would fall into the MLA kvhalf path.

## The four changes

| Change | Effect |
|---|---|
| `n_shared` optional (absent → 0) | Qwen3 converts; DS-V4 path still refuses on a true miss |
| `num_key_value_heads` + `head_dim` carried into config.json | the engine's switch to GQA |
| `head_dim` assumed = `hidden ÷ kv_heads` when absent | standard GQA, reported not silent |
| `make-synthetic --arch qwen3` | A3B-shaped fixture (self_attn q/k/v/o, mlp.experts gate/up/down, no shared, no scales) |

## Verify

```sh
python3 tools/qwen3-arch.py                    # apply (idempotent)
python3 tools/convert-ds4f.py self-test        # gate must stay green
python3 tools/convert-ds4f.py make-synthetic /tmp/q3 --arch qwen3
python3 tools/convert-ds4f.py inspect /tmp/q3
python3 tools/convert-ds4f.py convert /tmp/q3 --out /tmp/q3-out
```

Expected: self-test green; synthetic repo classified with 24 experts
(4/layer × 2 layers × 3 tensors); config.json carries
`num_key_value_heads: 2`, `head_dim: 4`, `n_shared: 0`.
