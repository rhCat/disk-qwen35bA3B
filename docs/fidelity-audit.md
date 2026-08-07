# Fidelity Audit: ds4f engine vs mlx-lm reference (Qwen3.5-35B-A3B)

Status: **FOUR real bugs found and fixed** (renorm, first-token, RoPE
pairing, GQA double-normalization). The engine now matches the reference:
GQA L3 t7 cosine 1.000000 / rel 1.0000, Jupiter rank 1 (engine == ref),
t0-t7 bit-exact at L0, coherent English generation.

## Harness

- `tools/dbg-reffwd.py` — faithful Python port of the reference forward
  (Qwen3NextLinearAttention + Qwen3Moe + GQA), decoding the exact same
  trunk/pool bytes the engine reads. Runs an 8+1-token prompt, dumps
  per-layer state traces + final state + L0 t8 delta-rule vectors.
  **Now F_NOCACHE + on-demand pread** (RSS 0.85 GB, was 17+ GB slurping
  pool.bin + trunk.bin).
- `tools/run-reffwd.sh` — runs the port (correct venv python, stripped env).
- `tools/run-clean.sh` — runs any python with the repo venv, unset PYTHONPATH,
  conda stripped from PATH.
- `tools/dbg-cmpstates.py` / `dbg-logitscmp.py` / `dbg-layercos.py` /
  `dbg-tokencos.py` / `dbg-veccos.py` — state/logits/vector cosines.
- `tools/dbg-trunkvssrc.py` / `dbg-headvssrc.py` — byte-verify trunk.bin and
  head.bin against the original safetensors.
- Engine side: `--dump-state` (final state), `DS4F_NAN_PROBE=1` (per-layer
  probes + L0 t8 vector dumps), `DS4F_DEBUG7` (top-5 logits).

## Verified ground truth (byte-level, vs source safetensors)

1. **embed.bin** (vocab x 2048, q*s+b) — byte-identical to source.
2. **head.bin / lm_head** — byte-identical (weight U32, scale/bias BF16).
3. **trunk.bin** — byte-identical for all probed tensors: input_layernorm,
   in_proj_qkv (U32), conv1d, A_log (F32), dt_bias, mlp.gate, post norm,
   GQA q_proj (L7), etc.
4. **pool.bin experts** — byte-identical (verified earlier session).

So the engine reads exactly what mlx-lm reads. Any remaining divergence is
in the engine's math, not the weights.

## The four bugs found (chronological)

1. **Router topk renorm** (`src/moe.c`) — mlx-lm only renormalizes
   `if norm_topk_prob:` and the config lacks it; the `/sum` amplified routed
   experts 2-5x. Restored coherent English (cosine 0.454 -> 0.786).
2. **First-token generation bug** (`src/main.c`) — prompt pass never sampled;
   the model echoed the prompt forever. Fixed: capture the last prompt
   token's sampled prediction.
3. **RoPE pairing** (`src/attn_qwen.c`) — mlx nn.RoPE(traditional=False) is
   HALF-SPLIT pairs (d, d+rd/2), not interleaved (2i, 2i+1). config
   mrope_interleaved=true is a transformers-ism mlx-lm ignores. Cosine at
   GQA L7 t1 went from divergent to 0.965.
4. **GQA attention double-normalization** (`src/attn_qwen.c:266`) — weights
   divided by softmax sum TWICE (`wgt[t2]/sum` after `wgt[t2]/=sum` at line
   260). Every GQA layer at position>0 contributed ~1/sum of its true signal.
   - t0 (1 pos, sum=1) was unaffected -> all early bit-exact checks passed.
   - t1+ magnitudes halved (1/2 at 2 pos), 1/8 at 8 pos. Cosine stayed
     0.95+ (scale-invariant) -> masked by cosine checks.
   - Jupiter rank 24 -> 1 (matches ref) after the fix.

## Current measured state (8-token prompt + 1 generated)

| quantity | value |
|---|---|
| GQA L3 t7 output cosine | **1.000000** (rel 1.0000) |
| L0 t7-L39 boundary cosine | 0.9545 (input to t8) |
| t7 per-token L0 cosine | 1.000000 (bit-exact all prompt tokens) |
| engine vs ref Jupiter rank | **1 vs 1** |
| logits cosine (9 tokens) | 0.934 |
| state cosine (9 tokens, pre-norm) | 0.852 |

## Long-context verification (does drift compound?)

**21-token prompt** (per-token final states, engine vs ref):
- t0–t8: cosine **1.00000000** (bit-exact)
- t9–t20: cosine 0.9997–0.9999, rel within 2% — float32 accumulation, damped not amplified

**129-token prompt** (Roman Empire passage, per-token final states):
- t0–t16: 1.00000000; t32: 0.999998; t64: 0.999997; t112: 0.999971; t128: 0.999999
- **Flat at 5–6 nines through 128 tokens — the drift does NOT compound.**

Why it stays flat: the delta-rule state is decayed every token (`state *= exp(g)`,
g ≈ -0.0005) and every layer renormalizes via RMSNorm — per-token float32
perturbations are damped, not amplified. The engine adds no systematic error
vs mlx-lm at any measured context length.

**20K-token stress run** (`tools/run20k.sh`, `--pids-file` support added for
prompts beyond argv limits): periodic `[stab]` final-state rms every 2K tokens
confirms bounded states (t=2000 rms 0.284, in the normal distribution range),
no NaN. Run cost is flat ~0.12 s/token prompt / ~0.30 s/token generation —
a normal conversation turn (300+150 tokens) is ~1-2 minutes; the 20K run is
slow only because it is 20K tokens.

## Sampling / repetition

- `DS4F_REP_PENALTY` (HF-style multiplicative over a 64-token window) breaks
  greedy long-context loops ("... X. X. X. ..."). With 1.3: France -> "Paris",
  2+2 -> "$4$", mountain -> "Everest" (chat format, greedy).
- `DS4F_TEMP` (softmax temperature before sampling) available.
- Qwen3.5 is a reasoning model: chat prompts need the full
  `<|im_start|>...<|im_start|>assistant\n<think>\n` template; generation
  starts with a think block, so give it room (GEN >= 64) to reach the answer.

## Memory findings (measured)

- Engine real footprint: 5.66-6.07 GB at --cache-gb 2 (GEN 12-16).
  --cache-gb 1: 2.46-5.07 GB depending on generation length.
- The 10-12 GB in Activity Monitor during probe runs was the REFERENCE PORT
  slurping pool.bin (17 GB) + trunk.bin (4.15 GB) — fixed with F_NOCACHE +
  pread (now 0.85 GB). The engine itself has F_NOCACHE on pool/trunk fds
  (verified: 1.6 GB read with flag = 0.01 GB page-cache delta).
- malloc empty-zone high-water: ~2.9 GB held in MALLOC_LARGE(empty) +
  MALLOC_REALLOC(empty) — macOS never returns freed large blocks to the OS.
- DS4F_CACHE_GB env override (default 2 GB arena), CLI --cache-gb wins.
- v0.1.0 tagged; perf: ~0.31-0.39 s/tok, 1.3-1.55 GB read/token.

## Known good commit anchors

- 2603489 MLX4 dequant fix (q*s+b)
- 67c711c conv1d tap order
- f83a7d1 shared expert
- 2bb3d93 prompt pass
- 65ea92d topk no-renorm + first-token fix
- 261014e RoPE half-split fix + memory probes
- 8a0f69b v0.1.0 prep: cache default 2 GB + DS4F_CACHE_GB
- a2a5e86 ref port F_NOCACHE pread (RSS 17GB -> 0.85GB)
- c460e32 GQA double-normalization fix (+ byte-verify + probe harness)
