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
- remaining: the 30 linear-attention layers (Mamba2-class: conv1d +
  A_log decay + selective scan) are still gracefully skipped, and the
  final lm_head isn't wired — the GQA half of attention is real, the
  linear half + head are next

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
