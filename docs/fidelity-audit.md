# Fidelity Audit: ds4f engine vs mlx-lm reference (Qwen3.5-35B-A3B)

Status: **OPEN** — one divergence remains, localized to the L0 embedding path.
The reference (mlx-lm `models/qwen3_5.py` + `qwen3_next.py` + `gated_delta.py`,
the implementation that actually serves this repo) is ground truth. Where they
differ, the engine is wrong.

## Harness

- `tools/dbg-reffwd.py` — faithful Python port of the reference forward pass
  (Qwen3NextLinearAttention + Qwen3Moe + GQA), decoding the exact same
  trunk/pool bytes the engine reads. Runs 6 prompt tokens, dumps per-layer
  state traces + final state (`--dump /tmp/q35-ref-state.bin`).
- `tools/run-reffwd.sh` — runs the port (correct venv python, stripped env).
- `tools/run-clean.sh` — runs any python with the repo venv, unset PYTHONPATH,
  conda stripped from PATH.
- `tools/dbg-cmpstates.py` — rms + cosine of engine vs ref final states.
- `tools/dbg-head.py` — lm_head logits from a dumped state, top-5 tokens.
- `tools/dbg-dtypes.py` — per-tensor dtype audit of the trunk.
- Engine side: `--dump-state` (final state), `DS4F_NAN_PROBE=1` (per-layer
  probes in `attn_qwen.c`), `DS4F_DEBUG_CHAIN` (MoE probes in `moe.c`).

## Confirmed correct (bit-matched or verified against mlx-lm source)

1. **4-bit decode is `q*s+b`** (raw nibble q in [0,15], NO 8-offset) — verified
   against MLX Metal affine_dequantize, qdot, and C++ VJP dequantize
   (commit 2603489). The old `(q-8)*s+b` was a ~100%-of-bias DC shift.
2. **Shared expert** is a dense every-token MLP: `y = routed + sigmoid(gate(x)) *
   shared_expert(x)` (commit f83a7d1). 480 tensors, previously mis-skipped.
3. **Conv1d tap order** — causal, w[0] = oldest; ring reads
   `tpos = token-(K-1-k)` (commit 67c711c).
4. **trunk.offsets layout** — `[0] count, then n x (off, nbytes)` pairs. Reads
   must stride by 2 (`offs_all[1::2]`), not flat.
5. **A_log is stored F32** (mlx-lm cast_predicate keeps it uncast); dt_bias,
   norms, and everything else BF16. dtype must be checked per tensor.
6. **RMSNormGated normalizes per-head** over Dv=128 with the learned weight
   (mx.fast.rms_norm on the last axis), not globally over 4096.
7. Embed rms, q/k/v norms, delta rule, router (softmax->topk->renorm), GQA
   kv-group mapping: engine matches reference numbers at L0 t0 for the
   components listed above.

## The open divergence (L0, t0, prompt token 5)

| quantity | engine | ref |
|---|---|---|
| embed rms | 0.0101 | 0.01036 |
| z[0..3] (in_proj_z out) | -3.33 -3.01 -2.77 **-0.27** | -2.50 -2.44 -3.17 **+0.99** |
| readout rms pre-norm (head 0) | 0.0088 | 0.00135 |
| linear output rms | 0.015 | 0.411 |
| final state rms (6 toks) | 0.43 | 2.59 |
| cosine(final state) | ~0 (uncorrelated) | — |

Reasoning: `z = in_proj_z(xin)` with identical weights and identical decode
path. If `xin` were the same vector, z would be identical; z[3] differing by
1.26 absolute is not a 2.6% scaling (the embed rms gap) — it is a **different
xin vector**. So the divergence originates in the **embedding gather
(src/head.c)**, and everything downstream (qkv -> conv -> v -> delta state ->
readout) inherits it.

### Prime suspect: embed gather assumption error

`src/head.c` embed gather predates Qwen3.5 (DS-V4 era) and was patched for the
flat-vs-nested JSON schema. The reference decode:

```
quantized_embedding: weight [vocab, in/8] U32 (8 nibbles/word),
scales/biases [vocab, in/group_size]   # group_size=64 -> [vocab, 32]
per row: 32 groups of 64 over 2048 dims; val = q*scale + bias
```

The engine's gather must match that group stride (row-major [vocab,32],
g = i/64, sibling .scales/.biases triplet offsets). Any deviation — wrong
group count, row-stride bug in the scales read, stale offset — yields exactly
this signature: statistically similar embed (rms 0.010) but elementwise
different, poisoning every layer.

### Verification (next step)

1. Dump engine embed row for token 5 (2048 floats) vs ref `embed_row(5)`.
2. First differing element + pattern: element 0 -> offset/triplet bug;
   every 64 elements -> group-index/stride bug; smooth noise everywhere ->
   decode formula.
3. Fix `head.c`, re-run state comparison — engine and ref states should
   converge, and the head should then predict a sensible continuation of
   "The capital of France is".

## Known good commit anchors

- 2603489 MLX4 dequant fix (q*s+b)
- 67c711c conv1d tap order
- f83a7d1 shared expert
- 2bb3d93 prompt pass
- defe6e5 fidelity audit scaffold
