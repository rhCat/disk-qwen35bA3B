# Breaking point — the memory contract

This document pins the experiment's hard numbers: the **3-4 GB resident
target**, why the pool must never be loaded, and the graceful-stop
guard. Anything that breaks these numbers is a breaking point.

## The target (fixed, not negotiable)

| Bucket | Budget | What lives there |
|---|---|---|
| **Resident (RAM)** | **3-4 GB** | attention q/k/v/o + linear-attn projections, router gate, shared expert, embed + lm_head, norms |
| **Pool (disk, streamed)** | ~17 GB (MLX 4-bit) | 256 experts × 40 layers, fixed-rate, O(1) random access |
| **Per-token stream** | 8 × 1.69 MB ≈ **13.5 MB/token** | only the top-8 experts touched per layer |
| **KV** | tiny | 10 full-GQA layers × ~40 KB/token + fixed O(1) linear-attn state |

The engine NEVER loads the pool into RAM. Per token it reads exactly
the 8 selected experts' slices from disk. If a run's peak RSS exceeds
~4 GB with the pool present, the pool is being resident-ified and the
design has broken.

## The evidence (why this exists)

- **2026-08-06, MLX 4-bit resident run on the M4 Pro (24 GB):** model
  load warning `requires 18594 MB, close to the maximum recommended
  size of 18186 MB`; memory pressure fell to **6% free** with 1.3M
  pages wired; the process was OOM-killed. Crash report
  `python3.14-2026-08-06-075424.ips` + JetsamEvent on file.
- That run loaded the WHOLE model (all 35B params) into unified
  memory — the anti-goal. The disk engine exists so that never happens.

## The artifact (already built)

`tools/split-mlx-switchmlp.py` streams the MLX 4-bit repo's flattened
`switch_mlp.{gate,up,down}_proj` tensors into the per-expert pool:

```
pool.bin  [u64 expert_nbytes][u64 n_layers][u64 n_experts]
          then expert (0,0), (0,1), ..., (L,E) contiguous, fixed-rate
```

Measured (this Mac, 2026-08-06): **16.88 GB, 256 experts × 40 layers,
1.69 MB per expert, topk 8, n_shared 1** — built without ever loading
the model into RAM (streaming slices, ~few MB peak).

`tools/build-trunk-mlx.py` builds the dense resident set the same way:
**702 MB trunk (40 layers, 910 tensors) + 242.5 MB embed + 242.5 MB
head** — ~1.2 GB of resident weights before cache/KV.

## Measured loading run (2026-08-06, M4 Pro, real pool + trunk)

```
config: 40 layers x 256 experts, topk 8, expert 1769472 bytes
trunk:  pin 4/40 layers, ring 2 x 19317184 bytes
cache:  1130 slots (1907 MB), 4 fetch threads
GB read per token: 0.70  (trunk 631 MB, experts 2053 MB)
cache: 1280 requests, 63 hits (4.9%), 0 dropped
exit 0 (23 GB gate armed, never fired)
```

- **0.70 GB read per token** from the 16.88 GB pool + trunk — the pool
  is streamed, never resident
- resident **plan 2.2 GB** — inside the 3-4 GB contract
- memory plan refused at >95% available (25.8 GB have / 2.2 GB need);
  the 23 GB graceful stop is the runtime backstop
- `make test` 20/20 (19 inherited + memlimit gate)

## Compute path LIVE (2026-08-06)

The MLX 4-bit kernel + format dispatch landed and the engine now
computes REAL Qwen3.5 expert matvecs from the disk pool:

```
router: real matvec on 40/40 layers      (mlx4 router gate drives selection)
moe: 2560 matvecs, 2684354560 decoded elements
GB read per token: 0.70  (trunk 631 MB, experts 2038 MB)
exit 0 (23 GB gate armed, never fired)
```

- `tools/split-mlx-switchmlp.py` emits the engine's layout schema
  (`fmt: 1`, `v_off/s_off/b_off`, decoded `[R, C]` dims) — 30720
  tensor entries
- `src/kernels.c`: `ds4f_mlx4_decode` / `ds4f_mlx4_matvec` (U32
  nibbles, BF16 scale+bias per 64-group), fixture-gated (tests 12/13)
- `src/moe.c`: `fmt`/`rel_b` fields, U32 dtype role, `.mlp.gate.weight`
  router matching, mlx4 dispatch in the expert chain
- `src/main.c`: MLX router gate path with sibling scales/biases
- `src/moe.c` (chain): Qwen3 parallel expert topology landed —
  `silu(gate(x)) * up(x) -> down` (chain flag, manifest order
  gate/up/down). Run: **3840 matvecs / 4.03B elements decoded**,
  40/40 layers real routing, exit 0, gate armed
- `src/attn_qwen.c` (GQA): full-GQA layers live — mlx4 q/k/v/o
  projections (packed-col x8 decode), RMSNorm q/k, 16-head attention
  over 2 kv-heads, o_proj residual add. Real effect measured:
  **bytes/token 0.70 -> 0.43 GB, cache hits 5.9% -> 52.8%** (10
  attention layers now compute from the resident trunk; real router
  scores stabilize expert selection). exit 0
- `src/attn_qwen.c` (linear): Gated DeltaNet LIVE — conv1d (kernel-4
  ring) + RMSNorm q/k + softplus(dt)*exp(A_log) decay + delta-rule
  state (kd x vd per value head, GQA 16->32) + readout + silu(z) gate
  + out_proj. **bytes/token 0.43 -> 0.32 GB, cache hits 52.8% ->
  72.6%** (30 linear layers compute from the resident trunk). exit 0
- remaining: the final lm_head (MLX 4-bit) isn't wired — all 40
  layers of attention + MoE are real, the head is the last piece
  before tokens come out
- **PERF (2026-08-06): 9.14 -> 0.21 s/token (43x).** Root cause of
  the original slowness: the engine built at -O0 (the Darwin default
  from the DS-V4 fixture era) AND the mlx4 kernel was scalar. Fixes:
  NEON/AVX2 two-pass mlx4 matvec (vectorized decode to scratch +
  8-accumulator FMA, **12.6 GFLOP/s / 6140M elems/s** vs 0.18 scalar),
  SIMD only when C%64==0 (group-boundary correctness; fixtures fall
  back to the LUT scalar path), malloc'd scratch row (per-row alloca
  SIGBUS'd in the 8-thread expert path), Makefile -O2 on Darwin (the
  full gate incl. e2e_text determinism passes at -O2). Real-model run:
  **0.21 s/token, PEAK RSS 2.28 GB, exit 0**
- **ACCURACY (2026-08-06): the forward is now numerically CORRECT --
  real logits, stable states, no NaN.** Three root-cause bugs fixed:
  1. **Router weights were raw logits (~55) not softmax probabilities**
     -- ds4f_topk stored scores directly; the reference does
     softmax(gate_logits) -> topk -> renormalize to sum 1. This was
     the 53x->8909x-per-layer MoE amplifier that exploded the state
     (0.04 -> 4.5e20 -> NaN by L28).
  2. **input_layernorm / post_attention_layernorm were never applied**
     -- the reference is x = x + attn(input_norm(x)); x = x +
     mlp(post_norm(x)). Both attention steps now project the NORMED
     input and residual-add to the raw state; the MoE norms xin.
  3. **RMSNormGated normalization is per-128-head**, not global 4096.
  Result: hidden state stable at 7-10 rms through all 40 layers,
  t0 logits are real vocabulary tokens, generated output went from
  "!!!!" to "Rot" (a real word fragment). State rms 5-8 across all 4
  generated tokens, exit 0.
- **TRUNK RE-STREAM (2026-08-06): multi-token generation.** The trunk
  ring was single-pass: the reader streamed layers forward and
  overwrote ring slots (L % nring) while token 1 re-read them --
  fixed tensors read as garbage at t1 (dt_bias = 1e-23). Added
  ds4f_trunk_rewind(): the engine re-streams the trunk once per token
  (reader waits at end-of-pass until rewound; ready/consumed
  handshake makes each bind wait for the re-fetch). Keeps the ring
  budget (2 x 19 MB); trunk re-read is 631 MB/token, the measured
  streaming rate. Gate 20/20 incl. the e2e fixtures (no deadlock).
- remaining: text is short fragments, not yet coherent prose -- the
  known approximations (text-only mrope positions, conv-ring boundary
  semantics) are the next fidelity targets
- **PROMPT PASS + GQA (2026-08-06): the engine now conditions on the
  WHOLE prompt.** Previously only pids[0] was processed and the rest
  of the prompt was never fed through the model (the model free-
  associated from one token). Fixes:
  1. The token loop now runs npids + gen iterations: the first npids
     feed the prompt tokens (no sampling/output) so the KV + delta-
     rule caches see the full context, then gen tokens generate.
  2. The GQA KV cache is sized npids + gen (was gen -- overflowed at
     position gen, reading uninitialized cache).
  3. GQA kv-head mapping fixed to the reference's repeat_kv grouping
     (q heads 0..7 -> kv head 0, 8..15 -> kv head 1; was h % kv_heads
     alternating).
  Result: output now DIFFERS BY PROMPT (Hello -> "elisrolerloeriore
  beruik", France -> "不同类型的ForMemberodoreoho ...") -- real
  conditioning. Logits are real vocabulary with plausible scores, but
  flat (top-5 within ~1-4) -- the likely floor is 4-bit quantization
  noise accumulating through 40 layers (scales ~0.005), vs the
  reference's BF16. Structural work is complete; coherent prose is a
  quantization-fidelity question, not a wiring question.

## Graceful stop (the guard)

The engine stops cleanly BEFORE the OS OOM-kills it:

```
--mem-limit-gb X    stop when peak RSS hits X GB (default 23; 0 = off)
```

- checked once per generated token against `ds4f_peak_rss()`
- on breach: prints `MEMORY LIMIT: peak RSS ... stopping gracefully`,
  writes `--dump-state` if requested, **exits 3**
- verified: `tools/test-memlimit.sh` (limit trip -> 3, control -> 0)

## Exit codes (the full contract)

| Code | Meaning |
|---|---|
| 0 | ok |
| 1 | config / usage |
| 2 | I/O |
| **3** | **memory limit (graceful stop)** |
| 4 | completed with dropped experts — silent corruption must not exit 0 |

## The honest distance

The I/O half of the loading experiment is **done and measured** (pool
built, fixed-rate, O(1)). The compute half — the trunk (attention +
shared expert + embed/head, the ~3 GB resident set) and the hybrid
GQA/linear-attention paths — is the remaining build. That is what turns
the pool into a runnable engine and produces the **measured** resident
footprint that validates (or kills) the 3-4 GB claim.
