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
