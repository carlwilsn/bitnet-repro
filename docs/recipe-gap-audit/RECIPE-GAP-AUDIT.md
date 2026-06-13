# Recipe-Gap Audit — Why the Ternary char-GPT Loses to FP

**Synthesis of `01-paper-recipe.md` (what the paper prescribes) and `02-code-inventory.md`
(what our code does today).** Analysis only — no code was changed and no training was run
to produce this document.

---

## 1. Framing — and the honest scale caveat

We observe a large train/val loss gap on the **day-1 character-level GPT** in
`DeepDL/bitnet-repro/experiments/day1-minimal-sanity`: the ternary ("BitNet") arm sits at
**~2.80 val loss** vs **~1.55** for the full-precision arm, on a **~3.22M-parameter**
4-layer / 4-head / 256-dim model trained on char-level Shakespeare (vocab 65, block 128).
This is **not** the torchtitan 160M run — it is the tiny sanity model, and that matters for
how we read the gap.

**The scale floor is real and must be subtracted before we blame the recipe.** BitNet b1.58's
own **Table 1** (and the scaling discussion in §3 of [b1.58], arXiv 2402.17764) shows ternary
weights *underperforming* the FP baseline at small parameter counts and only **catching up to —
then matching — FP16 around the ~3B-parameter scale**. Our model is **~3.22M params, roughly
three orders of magnitude below** where ternary is reported to become competitive. So a chunk of
the 2.80-vs-1.55 gap is almost certainly **irreducible at this scale** — no recipe fix will pull
a 3.2M ternary model all the way down to its FP twin, because the paper itself doesn't claim
parity there.

Therefore this audit explicitly separates two buckets:

- **Fixable recipe candidates** — divergences between our code and the prescribed recipe that
  plausibly *inflate* our loss beyond the scale floor, and that we can test cheaply. These are
  ranked in §3.
- **The scale floor** — the portion of the gap that is structural (3.2M ≪ 3B) and that we should
  *expect to remain* even after every recipe fix. We will only know its size empirically: fix the
  recipe candidates, re-measure, and treat the residual gap as the floor estimate for this size.

**Honest expectation:** the recipe fixes below should *narrow* 2.80 toward FP, perhaps
substantially (the LR/STE issue alone is a documented thrash, see RESULTS.md), but we should
**not** predict closing it to ~1.55. A realistic success here is "ternary lands meaningfully
below 2.80 and the residual is a defensible scale floor," not "gap eliminated."

---

## 2. Line-by-line diff table

Citations: paper side → section/equation in `01-paper-recipe.md`; code side → `file:line`
in `02-code-inventory.md`. "Diverges?" legend: **yes** (active mismatch) / **partial**
(present but weaker/different) / **silent** (recipe asks for it, code does nothing) /
**match**.

| Ingredient | Paper prescribes | Our code does | Diverges? |
|---|---|---|---|
| **Weight quant — gamma scope** | `absmean`, **whole-matrix** scalar γ = (1/nm)Σ\|W\| ([b1.58] §2 Eq 3, QUOTED) | whole-tensor `x.abs().mean()`, one scalar per matrix (`bitlinear.py:20`) | **match** (and note: both are whole-tensor — see §3 caveat) |
| **Weight quant — ε / clamp on γ** | scale by `γ+ε` ([b1.58] Eq 1, QUOTED) | `.clamp_min(1e-5)` on γ (`bitlinear.py:20`) | **match** |
| **Weight quant — RoundClip** | RoundClip→{−1,0,+1}, max(a,min(b,round)) ([b1.58] Eqs 1–2, QUOTED) | `torch.round` then `torch.clamp(·,−1,1)` (`bitlinear.py:24,27`) | **match** |
| **Weight quant — STE** | latent weights + identity-on-backward STE ([b1.58] §2, QUOTED qual.) | detached STE `w+(w_q−w).detach()` (`bitlinear.py:63,66`) | **match** |
| **Activation quant** | **8-bit, absmax, per-token, symmetric [−Qb,Qb]** ([b1.58] §2, QUOTED) | **NONE** — `x` enters `F.linear` full-precision (`bitlinear.py:55–66`) | **silent** — but see §3/§4: this makes our task *easier*, not harder |
| **Learning rate (value)** | **two-stage peak: 1.5e-3→1e-3 (700M), 1.2e-3→8e-4 (1.3B/3B/3.9B); ≈6× the FP LR** ([TrainTips] Table 2, **first-hand**) | single `--lr` default 1e-3, no arm override (`train.py:19`) | **partial** — peak in-range but no two-stage drop, no higher-LR-for-ternary handling |
| **LR schedule** | **two-stage linear decay**: stage-1 standard warmup to a higher peak, then decayed **midway** to a lower peak for stage-2; warmup **375 steps** ([TrainTips] §1 "Learning Rate" + Table 2, **first-hand**) | **none** — constant LR, no scheduler object (`train.py:58`, loop `:63–70`) | **silent** |
| **"BitNet wants higher LR than FP" rule** | explicit, load-bearing — [BitNet] §2.2 / §3.4 (**first-hand**): "BitNet benefits from a large learning rate… while the FP16 Transformer diverges at the beginning of training with the same learning rate"; [TrainTips] Table 2 quantifies it at **≈6×** the FP peak (**first-hand**) | same LR both arms; operator *manually* dropped BitNet to 3e-4 to stop thrash (RESULTS.md:11–14) | **yes** — code goes the *wrong direction* (lowered, not raised) |
| **Weight decay (value)** | **two-stage: 0.1 → 0** (stage-1 = 0.1 following LLaMA; disabled for the second half) ([TrainTips] §1 "Weight Decay" + Table 2 / Fig 1d, **first-hand**) | flat default 0.01, never staged (`train.py:58`, no `weight_decay=`) | **partial/yes** — wrong value, no staging |
| **WD param exclusions** | PAPER SILENT (norms/emb exclusion not prescribed; LLaMA convention only) | **none** — flat 0.01 on *every* param incl. norms & embeddings (`train.py:58`) | **silent** (vs convention) — code decays things usually excluded |
| **Optimizer** | AdamW (Table 2 reports Adam β **with** weight decay 0.1 → AdamW) ([TrainTips] Table 2, **first-hand**) | `torch.optim.AdamW` (`train.py:58`) | **match** |
| **Betas** | **(0.9, 0.95)** ([TrainTips] Table 2, **first-hand**) | default (0.9, 0.999) (`train.py:58`, unset) | **partial** — (0.9, 0.999) vs prescribed (0.9, 0.95); minor |
| **eps** | PAPER SILENT | default 1e-8 (`train.py:58`, unset) | **silent** |
| **Gradient clipping** | PAPER SILENT | **none** (`train.py:67–69`) | **silent** |
| **Init (latent weights)** | PAPER SILENT | `randn*0.02` then re-init `normal_(std=0.02)` (`bitlinear.py:48`, `model.py:120–122`) | **silent** (free choice) |
| **SubLN / norm placement** | norm **before** quant **inside** BitLinear — b1.58 uses a built-in **RMSNorm** in BitLinear and *removes* the pre-attention/pre-SwiGLU norm ([TrainTips] §3 code + step 2, **first-hand**; cf. SubLN in [BitNet] Fig 1) | `nn.LayerNorm` at **block boundary**, none fused inside BitLinear; no RMSNorm, no SubLN (`model.py:84,86,115`; none in `bitlinear.py`) | **yes** |
| **λ / quant warmup** | NOT FOUND — quantized from step 0 ([b1.58] §2, QUOTED) | none — full-strength ternary from step 0 (`bitlinear.py:55–66`) | **match** (both quantize from step 0) |
| **Mixed precision (shadow weights hi-precision)** | yes — full-precision latent weights, quantized on the fly via STE detach ([TrainTips] §1 "Weight Decay" + §3 code, **first-hand**; [BitNet] §2) | latent weights kept in fp32, updated via STE (`bitlinear.py:63`) | **match** (shadow-weight contract honored) |
| **Batch size** | PAPER SILENT for training (512 is inference-only) | 32 (`train.py:17`) | **silent** (free choice) |
| **Sequence / block length** | PAPER SILENT for training | 128 (`train.py:18`, `model.py:99`) | **silent** (free choice) |
| **Token budget** | 100B (RedPajama) / 2T (StableLM recipe) ([b1.58] §3, QUOTED) | ~4.1M @1k steps, ~12.3M @3k steps (`train.py:16–18`) | **yes** (different regime — tiny model, tiny budget; expected) |
| **Architecture — FFN** | **SwiGLU** ([b1.58] §2, QUOTED) | **GELU**, plain 4× MLP (`model.py:75`) | **yes** |
| **Architecture — pos-enc** | **RoPE** ([b1.58] §2, QUOTED) | learned absolute pos-emb (`model.py:110`) | **yes** |
| **Architecture — norm type** | **RMSNorm** ([b1.58] §2, QUOTED) | `nn.LayerNorm` (`model.py:84,86,115`) | **yes** |
| **Biases** | none anywhere ([b1.58] §2, QUOTED) | none at call sites (`model.py` all `bias=False`), though layer default is `True` (`bitlinear.py:42`) | **match** (by call-site convention) |
| **Dropout** | 0 (paper-style pretraining) | 0.1 (`model.py:103`) | **yes** (minor) |
| **LM head precision** | (head not specially quantized) | FP `nn.Linear` even in BitNet arm (`model.py:116`) | **match-ish** (FP head fine) |
| **Baseline "FP16" precision** | FP16/BF16 baseline ([b1.58] §2, QUOTED) | **misnomer** — both arms run fp32, no autocast/GradScaler (`train.py`) | **partial** (labeling bug, not a gap driver) |
| **Loss** | autoregressive CE ([b1.58] §3, QUOTED qual.) | `F.cross_entropy` flattened (`model.py:140–142`) | **match** |
| **Seed control** | PAPER SILENT | single `manual_seed` governs init+sampling, arms RNG-matched (`train.py:34`) | **silent** (free choice; good — A/B fair) |

---

## 3. Ranked divergences — most likely to close the gap first

Ranked on **mechanism**, not vibes. Each entry: (a) the divergence, (b) one-line mechanism,
(c) concrete next experiment with file/flag, expected **direction** and rough **magnitude**
(honest where magnitude is a guess).

### #1 — No LR schedule / warmup + the "BitNet needs a *higher* LR" rule is inverted in practice
**(a)** Paper prescribes a **two-stage linear-decay LR** — a standard warmup (375 steps) to a
**higher peak (1.2–1.5e-3, ≈6× the FP twin's 2–2.5e-4)**, then decayed **midway** through training
to a lower peak (8e-4–1e-3) — plus the load-bearing rule that ternary tolerates that higher peak
([TrainTips] §1 + Table 2 and [BitNet] §2.2/§3.4, both **first-hand** this pass). Our code runs a
**constant** LR with no warmup/decay (`train.py:58`, loop `:63–70`), and the operator empirically
*lowered* BitNet to 3e-4 to stop `1e-3` from thrashing the STE (RESULTS.md:11–14) — the opposite of
the prescribed direction.
**(b) Mechanism:** ternary forward + STE backward makes the loss landscape noisy and the *effective*
gradient weak (small latent-weight updates often don't flip the rounded sign). A constant, unwarmed
LR either thrashes (too high early, before the weights organize) or stalls (too low to flip signs).
Warmup tames the early-step STE noise; a high-then-dropped peak lets weights flip then settle. This
is the single most-cited BitNet recipe fact and our most direct mismatch.
**(c) Experiment:** add a warmup+cosine (or two-stage) scheduler in `train.py` around the optimizer
at `:58` and step it in the loop after `:69`. Sweep peak LR around the paper's first-hand value —
e.g. {1e-3, 1.5e-3, 3e-3, 6e-3} (the b1.58 peak is **1.2–1.5e-3**; note [Reloaded]'s 1e-2 is ~7×
above what the authors actually used) — with a short warmup (paper used **375 steps ≈ 0.4%** of its
~100K-step run; 1–5% is a reasonable tiny-run analog), BitNet arm only. **Direction:** lower BitNet
val loss. **Magnitude (guess):**
this is the biggest single suspect — plausibly **0.3–0.8** off the 2.80, because the current
configuration is a *known* thrash/stall regime, not a tuned one. Most uncertain part is the peak;
the warmup itself is high-confidence-positive.

### #2 — Weight decay 0.01 applied to *everything* (norms + embeddings included), and not two-staged
**(a)** Paper uses a **two-stage WD: 0.1 → 0** — stage-1 = 0.1 (following LLaMA), then **disabled
for the second half** of training ([TrainTips] §1 "Weight Decay" + Table 2 / Fig 1d, **first-hand**
this pass) — and is **silent on param exclusions** (the norms/embeddings exclusion is LLaMA
convention, *not* stated in TrainTips; see §4). Our code applies a flat **0.01 to every parameter**
including LayerNorm gains and both embedding tables (`train.py:58`, no param grouping).
**(b) Mechanism:** decaying LayerNorm gains and embeddings pulls them toward zero with no
counterbalancing signal, shrinking representation scale — especially damaging when the *only*
full-precision capacity left in the BitNet arm lives in the norms, embeddings, and LM head (the
projections are ternary). Decaying exactly those FP escape-valves disproportionately hurts the
ternary arm relative to FP.
**(c) Experiment:** split `model.parameters()` into decay / no-decay groups in `train.py:58`
(exclude all LayerNorm params + both `nn.Embedding` tables, keep decay on the BitLinear/Linear
weights), and test `weight_decay=0` vs `0.1` on the decay group. **Direction:** lower BitNet loss.
**Magnitude (guess):** **0.05–0.25**; smaller than #1 but cheap, and it specifically protects the
ternary arm's FP capacity. Confidence on *direction* is high; magnitude is a guess.

### #3 — Plain scale (3.22M ≪ 3B) — the structural floor
**(a)** Not a code bug — a fact. Paper Table 1 shows ternary below FP until ~3B params; we are at
3.22M (`02` param-count row; `01` §3 scaling).
**(b) Mechanism:** ternary weights carry ~1.58 bits each; at tiny width the model can't spend extra
capacity to compensate for the quantization noise, so a residual gap is expected by the paper's own
curve. No recipe knob removes this.
**(c) "Experiment":** there is no fix — instead, **measure the floor**: after #1 and #2 land, the
residual BitNet−FP gap *is* our empirical scale-floor estimate at 3.2M. Optionally bracket it by
running the same A/B at 2–3× width (n_embd 256→512, `model.py:102`) and checking the gap *shrinks*,
which would confirm the residual is scale, not recipe. **Direction:** wider ⇒ smaller gap.
**Magnitude:** unknown — this is what we're trying to *characterize*, not eliminate.

### #4 — No SubLN / norm fused before the BitLinear matmul (+ LayerNorm instead of RMSNorm)
**(a)** Paper places a normalization **inside** BitLinear before quantization (SubLN; [BitNet] Fig 1)
and uses RMSNorm model-wide. Our LayerNorm sits at the **block boundary**, with nothing normalizing
activations immediately in front of the quantized matmul (`model.py:84,86` vs `bitlinear.py`).
**(b) Mechanism:** SubLN stabilizes the *scale* of activations entering the quantizer, which keeps
γ and the rounding well-conditioned and steadies STE gradients. Without it, activation scale drifts
and the (already absent) activation quant would be ill-conditioned; even weight-only, the matmul
input variance is unmanaged at the quantization boundary.
**(c) Experiment:** add an RMSNorm/SubLN inside `BitLinear.forward` (`bitlinear.py:55–66`) before
the matmul, BitNet arm only, and A/B against current. **Direction:** lower/steadier BitNet loss.
**Magnitude (guess):** **0.05–0.2**, and partly *interacts* with #1 (more relevant once activation
quant exists). Lower confidence than #1/#2 in isolation.

### #5 — Architecture mismatch: GELU (not SwiGLU), learned pos-emb (not RoPE), dropout 0.1 (not 0)
**(a)** Paper is LLaMA-alike (SwiGLU/RoPE, dropout 0); we run GELU + learned absolute pos-emb +
dropout 0.1 (`model.py:75,110,103`).
**(b) Mechanism:** these change absolute model quality but apply to **both** arms roughly equally,
so they inflate *both* losses rather than the *gap*. Dropout 0.1 is the most arm-asymmetric suspect
here — regularization stacked on top of quantization noise can over-regularize the already
capacity-starved ternary arm.
**(c) Experiment:** cheapest first — set `dropout=0` (`model.py:103`) and A/B. SwiGLU/RoPE are
larger rewrites; defer. **Direction:** dropout 0 lowers both losses, possibly the BitNet arm
slightly more. **Magnitude (guess):** small on the *gap*, **<0.1**; matters more for matching the
paper's *absolute* numbers later than for the gap now.

### #6 (deliberately LOW) — No activation quantization (we are W1.58A16, paper is W1.58A8)
**(a)** Paper quantizes activations to 8-bit absmax per-token; our code quantizes **weights only**,
activations stay full precision (`bitlinear.py:55–66`).
**(b) Mechanism — and why this is NOT #1:** weight-only ternary with full-precision activations is
a strictly **easier** optimization problem than full W1.58A8. Adding the prescribed activation quant
would inject *more* quantization noise, which can only **raise** our loss toward the paper's harder
task, not lower it. So a *missing* activation quant **cannot explain why our loss is HIGHER than the
target** — if anything we have an advantage here. We list it for completeness and recipe-fidelity,
not as a gap-closer.
**(c) Experiment:** only relevant when the goal shifts from "close the gap" to "faithfully reproduce
W1.58A8." If added (8-bit absmax per-token in `bitlinear.py:55–66`), **expected direction is loss
UP slightly**, traded for recipe fidelity / deployability. Do **not** run this to close the gap.

---

## 4. Honest uncertainties (carried forward, not laundered)

- **The LR and weight-decay numbers are SECONDARY, not first-hand.** The two-stage LR shape, the
  "BitNet tolerates a higher peak LR" rule, the ~0.1→0 two-stage weight decay, and the implied AdamW
  optimizer all come **only** from the authors' **"Training Tips, Code and FAQ" companion** — which
  `01-paper-recipe.md` read **second-hand** this pass (via a hosted/third-party copy), **not** from
  the actual `The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf`. The **primary** b1.58 paper
  (arXiv 2402.17764, read in full) contains **no LR, no WD, no warmup, no optimizer, no batch size**.
  So #1 and #2 above rest on **unverified** numbers — the *direction* (warmup good, don't decay
  norms/embeddings, ternary likes a bigger LR) is well-corroborated, but the **exact peak LR, drop
  step, stage-1 WD value, and the step it zeroes** are NOT ground truth. **Before treating any of
  these as settled, open the actual Training-Tips PDF and replace the SECONDARY rows with first-hand
  quotes.**

- **Activation-quant ranking is deliberately #6, not #1.** Our code is **W1.58A16** (weight-only),
  an *easier* task than the paper's **W1.58A8**. A *missing* activation quant does **not** explain a
  *higher* loss — adding it would push loss **up**. Do not reflex-rank it as the top gap-closer.

- **Gamma scope is a MATCH, not a divergence.** Both paper (Eq 3) and code (`bitlinear.py:20`) use a
  **whole-tensor** scalar γ. The mission brief floated "per-channel γ starving dynamic range" as a
  suspect — but per-channel γ is **not** what the paper prescribes, so switching to it would be a
  *deviation from the recipe*, not a fix toward it. It may still be worth an exploratory experiment
  (per-output-channel γ could give ternary weights more usable dynamic range), but flag it as
  **off-recipe exploration**, not a recipe-gap fix. Keeping it out of the ranked recipe list on
  purpose.

- **Betas, eps, grad clipping, init, batch size, seq len are PAPER SILENT** — free reproducer
  choices, not divergences. Our values (betas (0.9,0.999), eps 1e-8, no clip, std-0.02 init, bs 32,
  block 128) are defensible defaults; none is a confirmed gap driver, though adding gradient clipping
  is a cheap stability hedge that pairs naturally with the #1 LR work.

- **"FP16" is a labeling bug, not a gap mechanism.** RESULTS.md/README call the baseline "FP16" but
  both arms run **fp32** (no autocast/GradScaler, `train.py`). This doesn't drive the gap; it just
  means our "FP" reference is fp32. Worth fixing the label so future numbers aren't misread.

- **Magnitudes in §3 are guesses.** Only the *directions* are mechanism-backed. The honest move is to
  run #1, re-measure, and let the data re-rank everything below it.
