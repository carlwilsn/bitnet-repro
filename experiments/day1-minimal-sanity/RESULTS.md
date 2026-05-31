# Day 1 — Minimal Sanity: BitNet b1.58 vs FP16 baseline

**Setup:** char-level Shakespeare, nanoGPT-style model (n_embd=256, 4 layers, 3.22M params),
identical harness, single variable = `BitLinear` (ternary, STE) vs `nn.Linear` (FP16).
Hardware: Lambda 1x A10. Each run ~15s (1000 steps) / ~45s (3000 steps).

## Results (final val loss)

| Run        | quant   | LR    | 1000 steps | 3000 steps | Shape                          |
|------------|---------|-------|-----------|-----------|--------------------------------|
| FP16       | none    | 1e-3  | 1.78      | **1.55**  | smooth, monotonic descent      |
| BitNet     | bitnet  | 1e-3  | 3.01      | —         | plateau at ~3.0 (STE thrash)   |
| BitNet     | bitnet  | 3e-4  | 2.79      | **2.82**  | descends then floors ~2.80     |

Random-init reference: ln(65) = 4.17. Step-0 loss ~3.7 (instantly grabs unigram stats).

## Findings

1. **The 1e-3 BitNet plateau (3.01) was an optimization artifact, not a wall.**
   The straight-through estimator (STE) cannot tolerate LR=1e-3 at this scale — latent
   weights thrash across the round() boundary every step and nothing settles. Dropping to
   LR=3e-4 unsticks it: 3.01 -> 2.79.

2. **Underneath the artifact is a real, paper-consistent gap.** Given 3x the budget,
   BitNet does NOT keep falling — it floors ~2.80 by step ~1400 and stays flat (even drifts
   to 2.82) while FP16 uses the same steps to glide 1.78 -> 1.55. At 3.22M params BitNet
   floors ~1.25 nats above FP16 and will not close it with more steps.

3. This reproduces the small-scale ternary gap (BitNet underperforms below ~3B params,
   Table 1 in the b1.58 paper) on a controlled, single-variable toy A/B.

## Repro

```bash
python train.py --quant none   --lr 1e-3 --steps 3000 --device cuda   # FP16 baseline
python train.py --quant bitnet --lr 1e-3 --steps 1000 --device cuda   # plateau
python train.py --quant bitnet --lr 3e-4 --steps 3000 --device cuda   # unstuck + floor
```
