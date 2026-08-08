# Token-generation waterfall — measured breakdown

The per-token cost of the engine, decomposed by phase. Measured with
`DS4F_WATERFALL=1` (per-phase timers in the decode loop, printed after
the run report). All runs: CPU path (no DS4F_GPU), greedy, cache-gb 5,
pin-layers 4.

## 5-token prompt, GEN=80 (the short-context baseline)

| phase | before NEON delta rule | after (f466206) | Δ |
|---|---|---|---|
| attn  | 70.8 ms (44.4%) | 57.7-59.7 ms (40%) | **-18%** |
| moe   | 47.4 ms (29.8%) | 47.5-49.4 ms (33%) | ~0 |
| fetch | 24.8 ms (15.6%) | 22.4-25.0 ms (16%) | ~0 |
| head  | 10.8 ms (6.8%)  | 10.9 ms (7.5%) | ~0 |
| router| 3.2 ms (2.0%)  | 3.2 ms (2.1%) | ~0 |
| misc  | 2.2 ms (1.4%)  | 2.3 ms (1.6%) | ~0 |
| total | 159.3 ms       | 144-151 ms | **-9%** |
| s/token | 0.16          | 0.14-0.15 | **-12%** (6.3 -> 7.1 tok/s) |

## 2K-token QA (2017-token prompt, post-NEON binary)

GEN=1 prompt pass: 339.4 s / 2017 tokens = **0.168 s/token**.
GEN=8: 336.9 s / 2025 = **0.167 s/token** (0.14-0.15 at 5 tokens +
a modest depth premium). Cache: 77.2% hit, 0 dropped. PEAK RSS
5.89 GB (GEN=1) / 6.90 GB (GEN=8).

Per-token waterfall at 2K (GEN=1, prompt pass):

| phase | ms/token | % |
|---|---|---|
| attn  | 73.8 | 43.8% |
| moe   | 41.7 | 24.8% |
| fetch | 38.1 | 22.7% |
| head  | 10.7 | 6.3% |
| router| 3.0  | 1.8% |

The fetch phase grows the most with context (25 ms at 5 tokens ->
38 ms at 2K): longer prompts touch more unique expert sets, and the
2825-slot cache (cache-gb 5) hit rate falls. attn stays flat
(57.7 -> 73.8 ms/token is the GEN=1 prompt-pass measure, which
includes the first-token state fill; the delta rule is O(1) per
token).

## History (how the waterfall moved this session)

| commit | change | s/token (5-tok) | s/token (2K) | attn | moe | fetch |
|---|---|---|---|---|---|---|
| 02a1341 | row-split main | 0.17 | 0.192 | 70.8 | 47.4 | 24.8 |
| f466206 | NEON delta rule | 0.14-0.15 | **0.168** | 57.7 | 47.5 | 22.4 |
| 0274c92 | persistent fetch pool | 0.14 | — | 56.7 | 47.9 | 23.2 |

## The fetch investigation (0274c92) -- what was measured, what it proved

The fetch phase (16-23% of the waterfall) was 22-38 ms for a ~3.5 ms
raw disk read, so the first hypothesis was spawn/join churn. Built a
persistent fetch pool (4 workers, claim/completion latches, no
per-layer pthread_create) -- 19/19 tests -- and measured:

| config | hits | misses/tok | disk bytes | fetch phase |
|---|---|---|---|---|
| cache-gb 5 (baseline) | ~70% | ~46 | 78 MB/tok | 23.2 ms |
| cache-gb 8 (bigger cache) | 89.1% | ~37 | 62 MB/tok | 22.4 ms |

Verdict: **fetch is disk-throughput-bound** (~4 GB/s parallel pread,
near NVMe capability). Neither the pool nor a 60% bigger cache moves
it. The 8 P-cores are NOT idle during fetch in a fixable way: the
router for layer L+1 needs moe(L)'s output (strict serial
dependency), and speculative prefetch is dead on arrival -- measured
adjacent-layer expert overlap is 3.6%, token-to-token 18.3%, so a
guess would be ~96% wrong. The route->fetch->compute serialization
is irreducible without a router predictor.

## What each phase IS (from the code)

- **attn** (40-44%): the Gated DeltaNet linear attention -- 39/40
  layers run the recurrent delta rule (state decay, kv_mem, delta,
  outer update, readout: 32 heads x 128x128 state) + the
  linear-attention projection matvecs (qkv 8192x2048, z 4096x2048,
  o 2048x2048 per layer, row-split). Layer 39 is one MLA/GQA softmax
  layer (naive full softmax, kvhalf fallback). NOT flash attention;
  the delta rule is O(1) per token, not O(position).
- **moe** (25-33%): 8 routed experts x 40 layers x 3 matvecs each
  (gate/up 512x2048, down 2048x512) -- fused decode+FMA,
  context-aware row-split, 8 expert threads.
- **fetch** (16-23%): expert fetch -- 8 experts x 40 layers from the
  2825-slot LRU cache (cache-gb 5), 4 fetch threads; disk read on
  miss. Trunk reads are flat ~693 MB/token at every depth (never the
  bottleneck -- pin-test proved: trunk 304 GB -> 1 MB with flat time).
- **head** (6-7.5%): lm_head 248320x2048 MLX-4 matvec (row-split 8).
- **router** (2%): gate scores (40x2048) + topk.
- **misc** (1.6%): norms, residual combine, sampling, embed.

## Context scaling

| prompt | per-token (pre-NEON) | per-token (post-NEON) | cache hit | trunk read/token |
|---|---|---|---|---|
| 5 tok | 0.16-0.17 s | 0.14-0.15 s | ~70% | ~745 MB |
| 2017 tok | 0.192 s | **0.168 s** | 77.2% | 693 MB |
| 4017 tok | 0.231 s | (not re-run) | 68.4% | 693 MB |

The rate degrades with context NOT because attention grows (the delta
rule is O(1) per token) but because longer contexts touch more unique
expert sets -> cache hits drop (77 -> 68%) and the fetch phase grows.
Trunk I/O stays flat -- disk is never the bound.

## History (how the waterfall moved this session)

| commit | change | s/token (5-tok) | s/token (2K) | attn | moe |
|---|---|---|---|---|---|
| 02a1341 | row-split main | 0.17 | 0.192 | 70.8 | 47.4 |
| f466206 | NEON delta rule | 0.14-0.15 | **0.168** | 57.7 | 47.5 |

Earlier in the session: malloc kill + fusion (0.24 -> 0.23),
context-aware row-split (0.23 -> 0.17). The GPU batched expert
offload measured 0.50 s/token (3x worse) and was rejected --
per-dispatch Metal sync overhead at 80 dispatches/token swamps the
compute.

## How to reproduce

```
DS4F_WATERFALL=1 bash tools/run-clean.sh ./ds4f /tmp/q35-trunk \
  --trunk /tmp/q35-trunk/trunk.bin --offsets /tmp/q35-trunk/trunk.offsets \
  --layout-trunk /tmp/q35-trunk/trunk.json \
  --pool /tmp/q35-pool/pool.bin --layout-pool /tmp/q35-pool/manifest.json \
  --head /tmp/q35-trunk/head.json --embed /tmp/q35-trunk/embed.json \
  --tokenizer ~/.cache/huggingface/mlx-qwen35-a3b-4bit/tokenizer.json \
  --pids-file /tmp/q35-2k-ids.txt --gen 8 --cache-gb 5 --pin-layers 4
```
## Chunked prefill (DS4F_PREFILL_CHUNK) — bit-identical batched attention

The prompt pass batches the linear-attention projections (qkv+z, M1) and
the GQA projections (q/k/v + o_proj, M3-lite) over B=64 tokens while
preserving the EXACT sequential accumulator map of the serial SIMD
matvec (the `(c>>2)&7` 8-accumulator topology, same FMA chains, same
reduction order). The recurrent bodies (Gated DeltaNet state chain,
GQA softmax over the growing K cache) stay serial per token. Verified
bit-identical against the ENGINE matvec (`tests/verify_batch2.c`).

### A/B table (greedy, 8 gen tokens, single-flight)

| prompt | serial | chunked | Δ | gen top1 identical |
|---|---|---|---|---|
| 200 tok | 31.7s | 24.7s | −22% | 8/8 |
| 500 tok | 83.3s | 62.4s | −25% | 8/8 |
| 1500 tok | 255.2s | 199.8s | −22% | 8/8 |
| 3000 tok | 575.5s | 444.7s | −23% | 8/8 |
| **4096 tok (target)** | **864.6s** | **612.2s** | **−29%** | **8/8** |

The 4096-tier pressure test (the target context level) is the strongest
result: 14.4 min serial → 10.2 min chunked, −29%, bit-identical
generation (8/8 top1, byte-identical response). The speedup grows with
context — the batched projections amortize better over longer prompts.
Per-token: 0.21 s/tok serial → 0.15 s/tok chunked at 4096.

Bugs found & fixed during the work (all documented in commit bodies):
absolute MLX4 group indexing in the batch kernel; double prompt
processing after the chunked pass (153s→72s); `gqa_step` zeroing only
HALF of attn_out (orows vs ocols — stale heads 8-15, masked by the
serial path's full-arena memset, exposed by the chunk; the 1500-tok
A/B went 0/24 → 8/8 after the fix); `lin_body` missing `return 0;`;
`last_tok` not captured by the chunk pass.

Rejected with data: M2 batched-moe (grouped experts measured 2× slower
at B=64 — routing diversity gives 2-4 tokens/expert, killing the
dequant amortization). See `docs/breakdown.md` for the full candidate
analysis and the M3-lite decision.

### Transmutation close-out (f7222137, after M3-lite merge)

calcination: commit proven (f7222137259b), 240 blueprints / 374
functions / 471 evidence. citrinitas ground: **Contradictions 0**
(ErrorPropagation 406, GateInventory 65, 362 flows) — the symmetry
regression oracle stays green through the full perf arc
(strdup-era 1 → 0 at 8bab9995 → 0 at a31a1a48 → 0 at f7222137).
