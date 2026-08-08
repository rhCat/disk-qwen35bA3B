# Batched prefill (chunked prompt processing) — SCOPE

Status: scoped, not yet implemented. Branch target: `perf/prefill-batch`.

## The problem (measured)

Every token — prompt OR generated — runs the same full 2.75 G MAC
forward pass, one at a time. There is no prefill batching:

  prompt: 158.1 ms/token (500-tok run), 178.9 ms/token (3K run)
  gen:    136.5 ms/token (5-tok run)

A 3K prompt costs ~9 minutes of ingestion at the same speed as
generation. In a batched engine the prompt tokens are processed in
parallel through shared weights (matvec -> matmul, B tokens at once),
amortizing the weight fetch and the dequant across the batch.

## Why batching is fast (the mechanism)

The MLX4 matvec cost is dominated by the DEQUANT: reading the 4-bit
weights and applying q*s+b per element. In a matvec (B=1) the dequant
happens once per weight element per token. In a batched matmul
(B=64) the SAME dequantized weight is reused for 64 dot products.
So batching amortizes the expensive part ~B times:

  dequantized weight row (2048) x 64 token vectors
    = 64 FMAs per dequant, instead of 1

The delta-rule state update is the one serial dependence (token t+1
needs token t's state). It has a chunked formulation (FLA: intra-
chunk quadratic + inter-chunk recurrent) but that's the hard part.

## Milestones (each measurable, each lands independently)

### M1: batched projection matvecs (THE FIRST IMPLEMENTATION)
The attention projections qkv (8192x2048), z (4096x2048), o
(2048x2048) are 72% of the attn phase and are pure matrix ops --
they have NO cross-token dependence. Batch them over B prompt
tokens:

  ds4f_mlx4_matvec_batch(vals, scales, biases, R, C, B,
                         xs[B][C], ys[B][R])

Same kernel, B columns. The row-split workers each handle a row
range x B tokens; the dequant is done once per (row, worker) and
reused across the B x-vectors.

Expected: attn projections 39.6 ms/token -> ~4-8 ms/token-equivalent
at B=64. attn phase ~51 -> ~16-20 ms/token-equiv. Total short-ctx
~0.135 -> ~0.10-0.11 s/token-equiv. Prompt ingestion 10-20x faster.

Wiring: linear_step / gqa_step get a B-token variant. The delta rule
still runs serially per token in the chunk (its 15.4 ms/token-equiv
stays for now) -- M1 is the projection-only win, measured in
isolation.

### M2: batched expert matvecs
The routed experts (1.0 G MACs) are the same matvec -> matmul
change, plus the fetch amortizes: 8 experts fetched ONCE per chunk
instead of once per token (fetch 22 ms/token -> ~0.4 ms/token-equiv
at B=64). Router must group chunk tokens by expert (expert-batched
routing).

### M3: chunked delta rule (the hard part)
FLA chunked formulation: intra-chunk attention is a BxB quadratic
(q k^T with the chunk's own keys), inter-chunk uses the recurrent
state. Kills the last serial dependence in the 39 linear layers.
Realistically a separate workstream (new kernel, correctness vs
e2e).

### M4: batched layer-39 softmax
Standard prefill: Q K^T (B x npos), row softmax, V. The O(position)
scan becomes a batched matmul. This also fixes the context-growth
cost (78.7 ms/token at 3K -> flat with npos).

## Measured baselines (must beat)

| metric | now | M1 | M1+M2 | +M3+M4 |
|---|---|---|---|---|
| prompt 3K | 178.9 ms/tok | ~100? | ~50? | ~15-25? |
| gen | 136.5 ms/tok | ~110? | ~100? | ~100? |

The gen rate barely moves (it's serial); the PROMPT rate is the win.

## Risks / open questions

1. The delta rule serial loop (M1 keeps it) means the chunk still
   pays 15.4 ms x B for the state updates -- M1's win is only the
   projection share. Honest scoping: M1 is ~2x on the prompt pass,
   NOT 10x. The 10x needs M3.
2. Memory: B x 2048 x 4B x 3 projections = 1.5 MB per chunk at
   B=64 -- trivial.
3. Router batching (M2): the 8 experts per token differ across the
   chunk; grouping adds a sort. Fetch amortization is the prize.
4. Bit-fidelity: the batched kernel must produce bit-identical rows
   to the serial matvec (same per-row accumulation order). e2e gate.
5. The 4-bit dequant in the batched kernel: decode once, broadcast
   to B lanes -- NEON can do this with vdupq over the B dimension.

## Next action
Implement M1 (ds4f_mlx4_matvec_batch + linear_step wiring) on
perf/prefill-batch, measure the attn phase at B=64 on the 500-token
fixture, verify e2e bit-fidelity.
