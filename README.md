# disk-qwen35bA3B

Disk-streaming MoE inference **structure** in portable C99 for
**Qwen3-30B-A3B** (the "35B A3B" class: 30.5B total, ~3.3B active).
No BLAS, no framework, no GPU, no dependencies beyond libc + pthreads.

The systems architecture mirrors the family proven by
[kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c),
[Colibrì](https://github.com/JustVugg/colibri), and the author's
[ds4f-disk](https://github.com/rhCat/ds4f-disk): a tiny in-RAM pointer
map, a contiguous packed trunk, and policy-controlled expert streaming.
This repo is the same structure **pointed at the A3B** — and the A3B is
the model the structure was always right for.

## Why A3B (the argument in one table)

| Property | Qwen3-30B-A3B | DS-V4-Flash (ds4f-disk) |
|---|---|---|
| Total params | ~30.5 B | ~304 B |
| Active per token | ~3.3 B | ~22 B |
| Attention | **GQA** (4 kv-heads) | MLA + mHC + Sinkhorn |
| Shared experts | none | yes |
| Research wall (freeze/contraction) | **none** | the unsolved core |
| Pool on disk | ~58 GB bf16 / ~30 GB FP8 | ~68 GB mxfp4 |
| Per-token expert stream (mxfp4) | **~0.9 GB** | ~3.6 GB-equivalent |

The A3B streams **4× less per token** on a pool **5× smaller**, and its
forward pass is plain GQA + top-8 MoE — mechanical, fixture-gateable,
no numerics research. The structure's resident set is ~5% of the model.

## Memory profile (real config: hidden 2048, 48 layers, 128 experts
top-8, moe_inter 768, 4 kv-heads × 128, vocab 151936)

| Bucket | Size | Why |
|---|---|---|
| **Resident (RAM)** | **~2.5–3 GB** | attention q/k/v/o, router gate, embed+lm_head (no tie), norms |
| **Streamed (disk)** | ~58 GB bf16 · **~30 GB FP8** | 128-expert pool, 6.25% touched per token |
| **KV cache** | **196 KB/token** fp32 → 6.3 GB @ 32K, 50 GB @ 256K | GQA full KV, 48 layers |
| **Per-token disk traffic** | **~0.9 GB mxfp4** · ~3.6 GB bf16 | 48 × 8 experts × 3 proj (2048×768) |

Serving reality: 128 GB CPU box → whole model + fat KV resident, no
GPU needed; the per-token stream at NVMe bandwidth lands
~0.2–0.3 s/token. On 2× DGX Spark (128 GB each) the full bf16 model
fits with ~95 GB/machine left for KV.

## Layout (target)

```
include/ds4f/ds4f.h     public API + invariants (forked from ds4f-disk)
src/cfg.c               config reader, qwen3_moe keys
src/st.c                safetensors index -> pointer map
src/trunk.c             packed trunk: pin + ring + async prefetch
src/cache.c             routed-expert cache: 3-phase fetch
src/router.c            top-8 router (norm_topk_prob)
src/attn.c              GQA attention (q/k/v/o + RoPE + per-head KV)
src/moe.c               Qwen3 expert chain (gate/up/down, silu)
src/kernels.c           scalar mxfp4 / bf16 kernels
src/head.c              embed + lm_head (BF16)
src/tokenizer.c         Qwen3 BPE
tools/convert-qwen3.py  HF checkpoint -> ds4f layout
tools/qwen3-arch.py     converter arch diff (DS-V4 converter -> A3B)
tools/make-fixture.c    synthetic A3B fixture (no real weights)
tests/                  weightless gate suite + fixture e2e
```

## Status

- [x] Config grounded from HF (`Qwen/Qwen3-30B-A3B-Instruct-2507`)
- [x] Converter arch diff authored (`tools/qwen3-arch.py`; n_shared→0,
      GQA geometry carried, `make-synthetic --arch qwen3`)
- [ ] Converter diff applied + self-test green
- [ ] GQA attention path (attn.c) — the MLA→GQA rewrite
- [ ] Qwen3 expert chain (moe.c): gate/up/down triplet, top-8,
      norm_topk_prob
- [ ] BF16 head/embed + Qwen3 tokenizer
- [ ] Fixture gate e2e (A3B-shaped synthetic, deterministic)
- [ ] Real checkpoint on a 128 GB box

## Build and verify (inherited from ds4f-disk, adapted)

```sh
make            # ds4f, pack-trunk, make-fixture
make test       # gate suite (weightless) + fixture e2e
```

## License

MIT. Model weights are not included and are subject to their own
licenses (Qwen3-30B-A3B is Apache-2.0).
