# Remaining cost breakdown & next-optimization candidates

After the bit-identical chunked prefill merge (PR #2, `1e2b01e`), the
prompt pass on the 500-token scale is 63.4s chunked vs 80.3s serial
(-21%). The remaining per-token cost is dominated by three phases
(serial-path waterfall, 5-tok GEN=80 measurements):

| phase | cost | % of token | status |
|---|---|---|---|
| moe (8 experts/token) | ~47 ms | 35% | serial per token; M2 batch REJECTED (2x slower at B=64) |
| attn (qkv+z + delta rule + GQA) | ~51 ms | 38% | qkv+z batched (M1); delta rule serial; **GQA projections serial** |
| fetch (disk) | ~21 ms | 15% | **at the NVMe floor — irreducible** (verified: pool + cache-gb 8 didn't move it) |
| head | ~10 ms | 8% | skipped for prompt tokens already |

## Candidate 1: M3-lite — chunked GQA projections (NEXT)

**What:** L3 and L39 are full-GQA (softmax attention) layers. In the
chunked path they still run `gqa_step` serially per token: q, k, v
projections (3 matvecs of 8192/512/512 rows over 2048 cols) + softmax
+ o_proj (2048 x 4096). The projections are *independent per token*
and share the same weight tensors — they batch exactly like M1's
qkv+z: one row-decode, B tokens through the same accumulator map.
The softmax attention itself stays serial (it reads the past-token K
cache — inherently sequential), but the projection cost (the bulk of
GQA) amortizes over the chunk.

**Expected:** ~8s of the chunk pass (measured `[chunk-sum]` gqa
phase at 200/500-tok) drops to ~2-3s. Bit-fidelity gate: the batch
kernel is already verified against the engine matvec; the GQA body
after the projections is unchanged serial code.

**Risk:** low — same proven kernel, same bit-fidelity gate, no
algorithm change. The attention math (softmax, gate, RoPE) stays
bit-identical.

## Candidate 2: M3-full — chunked delta rule

**What:** batch the recurrent Gated DeltaNet body (conv1d ring +
state update + readout) across the chunk. The state chain is
sequential by definition (token t+1 reads token t's state), but a
chunked-delta-rule algorithm (e.g. the FLA chunked formulation) can
compute the within-chunk state contributions in parallel and fold
them at the boundaries.

**Expected:** would cut the delta-rule body (~15.4 ms of attn's
51 ms) and the conv ring. But the projections around it are ALREADY
batched (M1), so the serial remnant is only ~30% of attention.

**Risk:** HIGH. New algorithm, must be bit-identical or verified
within the recurrence tolerance; the delta rule is the exact place
where 1-ULP drift amplifies (the whole reason for the sequential-map
preservation). The GQA layers (L3/L39) aren't delta-rule at all.

## Candidate 3: head batch

**What:** the lm_head projection for the prompt pass. Already
skipped: the chunked pass runs the head only once (last_tok capture
+ gen tokens). Nothing left to batch — DONE.

**Alternative within this slot:** batch the moe despite M2's
rejection — M2 grouped experts lost because B=64 routing gives 2-4
tokens/expert. At B>=256 the groups grow and the dequant amortizes;
the kernel exists (`ds4f_moe_step_batch`) and the batch matvec is
bit-identical. But B=256 means 4x the pstates memory and the rfm
loop is already serial per token; the win is speculative.

## Decision

**Proceed: M3-lite (candidate 1).** Same kernel, same gate, targets
the last serial projection work in the chunk path (GQA layers), no
algorithm risk. The chunked GQA projections reuse the verified
bit-identical batch matvec; the softmax/gate/RoPE body stays serial
and bit-identical.
