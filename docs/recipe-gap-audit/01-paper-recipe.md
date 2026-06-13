# 01 — Paper Recipe: What the BitNet b1.58 Papers Actually Prescribe

**Scope:** The "what the paper prescribes" half of the recipe-gap audit. This is a
citation-backed checklist of the BitNet b1.58 **training recipe** as stated in the
primary sources. Its sibling, `02-code-inventory.md`, is the "what our code does" half;
ingredient names below are kept identical so the two files diff row-for-row.

**Confidence discipline.** Every row says *which source* and *how strong* the citation is:

- **[QUOTED]** — I read the exact text/equation myself this audit (the local PDF) and am quoting/paraphrasing it directly.
- **[SECONDARY]** — I could not open the full primary text (arXiv served binary/HTML I couldn't parse); the fact comes from the paper's abstract, an author-released companion doc summarized by a third party, or a corroborating paper. Treated as lower confidence and flagged inline.
- **[PAPER SILENT]** — the sources do not specify this; the number is a free choice for the reproducer.

---

## Sources

| Tag | Source | How accessed this audit |
|---|---|---|
| **[b1.58]** | arXiv **2402.17764**, *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits* (Ma, Wang, et al., Feb 2024). | **Read directly** — local PDF `DeepDL/readings/bitnet/bitnet-b158.pdf`, all 8 pages via PdfExtract. Equations and quotes below are verbatim from it. |
| **[BitNet]** | arXiv **2310.11453**, *BitNet: Scaling 1-bit Transformers for Large Language Models* (Wang, Ma, et al., Oct 2023). The original architecture paper; b1.58 says it "is based on the BitNet architecture" and "the quantization function for activations follows the same implementation in BitNet." | **Abstract + secondary only** — arXiv served the raw PDF byte-stream and the `/html/` route 404'd, so I could **not** read its training section verbatim this audit. Rows from it are **[SECONDARY]**. |
| **[TrainTips]** | *The Era of 1-bit LLMs — Training Tips, Code and FAQ* — author-released companion, listed in the official `microsoft/BitNet` README "What's New" as **"03/21/2024 The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ"** (lives in `microsoft/unilm/tree/master/bitnet`). | **Secondary** — confirmed it exists and its headline facts (two-stage LR, two-stage weight decay "Figure 1d") via a hosted copy (Scribd "Training 1-Bit LLMs: Tips and Code", 7 pp.) and a third-party summary. I could **not** extract the exact per-size hyperparameter numbers. Rows **[SECONDARY]**, numbers flagged as unverified. |
| **[Reloaded]** | arXiv **2407.09527**, *BitNet b1.58 Reloaded* — independent reproduction, **not** an author doc. | **Secondary, corroborating only.** Used to sanity-check the "higher LR" direction, never as the prescriptive source. |

> **Headline caveat:** the *primary* b1.58 paper [b1.58] that I could read in full contains
> **no learning rate, no weight decay, no warmup, no optimizer, and no batch-size recipe.**
> All optimizer/schedule numbers in this audit come from the companion **[TrainTips]** doc
> (author-released but read only second-hand) or are **[PAPER SILENT]**. Treat every LR/WD
> number below as *needs first-hand verification against the actual Training-Tips PDF.*

---

## Recipe ingredient checklist

| Ingredient | What the paper prescribes | Source + citation (confidence) |
|---|---|---|
| **Learning rate (value/default)** | No single value in the readable paper. The companion doc's reproductions and the corroborating paper land on **~1e-3 → drop**, with **1.58-bit able to tolerate a *higher* peak LR than FP16** (e.g. 1e-2 reported by the independent reload). The *direction* — "BitNet wants a larger LR than the FP16 baseline" — is the load-bearing claim. | [TrainTips] **[SECONDARY]** (two-stage peak LR; exact value not first-hand-verified). Corroborated by [Reloaded] §exp: *"For 1.58-bit, we used the same or a larger learning rate of 0.01 (1e-2), as 1.58-bit training has been found to be more robust to higher learning rate."* **[SECONDARY]** |
| **LR schedule** | **Two-stage schedule.** Stage 1: warmup then a high peak LR. Stage 2: drop to a much smaller LR for the remainder of training. This is the canonical BitNet b1.58 schedule and is the single most-cited "recipe" fact about it. Underlying shape is warmup + (cosine-style) decay within the stages. | [TrainTips] **[SECONDARY]** — two-stage LR is the documented scheme; exact peak, drop step, and decay curve **not first-hand-verified**. |
| **"BitNet needs a higher LR than FP16" rule** | **Yes — explicit and load-bearing.** The reason given in the literature: with ternary weights, a small update to the high-precision *latent* weight often does **not** change the sign/rounded value, so the effective gradient signal is weaker; a larger LR is needed to actually flip quantized weights. Practically: start BitNet at a higher peak LR than you would the FP16 twin. | Mechanism attributed to [BitNet] training section **[SECONDARY]** (could not read verbatim); the "more robust to higher learning rate" phrasing is quoted from [Reloaded] **[SECONDARY]**. |
| **Weight decay (value)** | **Two-stage weight decay.** Stage 1 uses a non-zero decay (commonly cited ~**0.1**); Stage 2 sets weight decay to **0**. Companion doc's "Figure 1d" depicts exactly this two-stage WD curve, paired with the two-stage LR. | [TrainTips] **[SECONDARY]** — *"the weight decay strategy also followed a two-stage approach"* (Scribd copy, "Figure 1d"). Stage-1 value ~0.1 / Stage-2 = 0 is the commonly reported pairing but **not first-hand-verified**. |
| **Weight-decay param grouping (which params excluded?)** | **Not found in any source I could read.** Whether norms / biases / embeddings are excluded from decay is **not stated** in [b1.58] (read in full) and I could not confirm it in [TrainTips]. (Note: biases don't exist — see "BitLinear bias" — and the standard LLaMA convention excludes norm+embedding from decay, but the BitNet docs do **not** prescribe this explicitly.) | **[PAPER SILENT]** on exclusions. Do not assume; verify against the Training-Tips PDF if it matters. |
| **Optimizer** | AdamW (Adam with decoupled weight decay) is the implied optimizer (LLaMA-alike training stack + the two-stage *weight decay* only makes sense with AdamW). Not stated verbatim in the readable paper. | [TrainTips] **[SECONDARY]** (implied by two-stage WD); **[PAPER SILENT]** in [b1.58]. |
| **Betas** | **Not specified** in any source I could read. (Adam β = (0.9, 0.95) is the LLaMA convention; (0.9, 0.999) is the PyTorch default — the BitNet docs prescribe *neither* explicitly.) | **[PAPER SILENT]** — flag for first-hand check of [TrainTips]. |
| **eps** | **Not specified** in any readable source. | **[PAPER SILENT]**. |
| **Gradient clipping** | **Not specified** in any readable source. Not mentioned in [b1.58]; could not confirm in [TrainTips]. | **[PAPER SILENT]**. |
| **Weight quantization — gamma scope** | **`absmean`, scope = the WHOLE weight matrix** (one scalar γ per `nn.Linear`/BitLinear). γ = mean absolute value over all `n·m` entries. | [b1.58] **[QUOTED]** §2 Eq (3): `γ = (1/nm) Σ_ij |W_ij|`. |
| **Weight quant — epsilon/clamp on gamma** | Scale by `γ + ε` (small ε in the denominator to avoid div-by-zero). ε value not numerically specified. | [b1.58] **[QUOTED]** §2 Eq (1): `W̃ = RoundClip(W / (γ + ε), −1, 1)`. |
| **Weight quant — RoundClip vs round+clamp** | **RoundClip**, defined exactly as round-then-clip to {−1,0,+1}: `RoundClip(x,a,b) = max(a, min(b, round(x)))` with `a=−1, b=+1`. | [b1.58] **[QUOTED]** §2 Eqs (1)+(2). |
| **Weight quant — STE formulation** | Straight-through estimator: high-precision latent weights are kept; the quantizer is treated as identity on the backward pass so gradients flow to the latent weights. Stated as method, not as an equation, in the readable paper; detailed in [BitNet]. | Mechanism in [b1.58] §2 (*"trained from scratch"* with latent weights) **[QUOTED, qualitative]**; STE detail attributed to [BitNet] **[SECONDARY]**. |
| **Activation quantization** | **8-bit activations.** **absmax per-token**, scaled to the **symmetric** range **[−Qb, Qb]** (no zero-point). Crucially, b1.58 **changed** the original BitNet recipe: it does **NOT** scale activations to **[0, Qb]** before the non-linearities — it uses the symmetric per-token scaling everywhere "to get rid of the zero-point quantization." | [b1.58] **[QUOTED]** §2: *"trained … with 1.58-bit weights and 8-bit activations"*; *"the quantization function for activations follows … BitNet, except that we do not scale the activations before the non-linear functions to the range [0, Qb]. Instead, the activations are all scaled to [−Qb, Qb] per token to get rid of the zero-point quantization."* |
| **Normalization (pre-quant / pre-BitLinear)** | A normalization sits **before** the quantization inside BitLinear. In b1.58 the model norm is **RMSNorm** (LLaMA-alike). In the original BitNet, BitLinear places a **SubLN / LayerNorm before the absmax activation quantization** (Figure 1 of [BitNet]: `Input → LayerNorm → Absmax Quantization → 1-bit Weights`). Net: **normalize first, then quantize the activation.** | RMSNorm choice: [b1.58] **[QUOTED]** §2 LLaMA-alike. Norm-before-quant placement in BitLinear: [BitNet] Fig. 1 **[SECONDARY]** (figure caption read via secondary source). |
| **BitLinear bias** | **No biases anywhere.** All bias terms removed (LLaMA-alike). | [b1.58] **[QUOTED]** §2: *"removes all biases."* |
| **Init — BitLinear latent weights** | **Not specified.** No prescribed std / init scheme for the latent weights in any source I could read. | **[PAPER SILENT]**. |
| **Init — other params** | **Not specified.** | **[PAPER SILENT]**. |
| **Architecture — components** | LLaMA-alike: **RMSNorm**, **SwiGLU** FFN, **rotary position embedding (RoPE)**, **no biases**. Decoder-only Transformer; `nn.Linear` → `BitLinear`. | [b1.58] **[QUOTED]** §2: *"uses RMSNorm, SwiGLU, rotary embedding, and removes all biases."* |
| **Activation function** | **SwiGLU** in the FFN (from the LLaMA-alike list). | [b1.58] **[QUOTED]** §2. |
| **Positional encoding** | **RoPE** (rotary embeddings). | [b1.58] **[QUOTED]** §2. |
| **Attention** | Standard multi-head self-attention (decoder-only Transformer backbone; b1.58 only swaps the linear layers and norm — attention math is the LLaMA baseline). | [b1.58] §2 backbone **[QUOTED, qualitative]**; **[PAPER SILENT]** on head dims / GQA. |
| **Batch size** | **Not specified for the LM-loss training run.** (The paper reports an *inference* throughput study at sequence length 512 with batch sizes pushed to the GPU memory limit, but that is a serving benchmark, not the training batch size.) | **[PAPER SILENT]** for training batch size. Inference-only batch numbers: [b1.58] §3 / Table 3 **[QUOTED]**. |
| **Sequence / block length** | **Not specified for training.** Inference/throughput experiments use **sequence length 512**; that is the only explicit length in the paper and it is a benchmark setting, not a stated training context length. | **[PAPER SILENT]** for training seq-len; 512 is inference-only [b1.58] §3 **[QUOTED]**. |
| **Token budget per run** | **Two regimes, both explicit:** (a) **100B tokens** on **RedPajama** for the main FP16-vs-b1.58 comparison across sizes (700M–3.9B); (b) **2T tokens** following the **StableLM-3B** data recipe for the scalability test at 3B. | [b1.58] **[QUOTED]** §3: *"pre-trained the models on the RedPajama dataset for 100 billion tokens"*; §3 "Training with 2T Tokens" + Table 4. |
| **Dataset** | **RedPajama** (100B-token runs); **StableLM-3B data recipe** (2T-token run). | [b1.58] **[QUOTED]** §3. |
| **λ / quant warmup (gradual quantization ramp)** | **NOT FOUND.** No source I could read describes a λ that ramps the model from full-precision toward quantized over early steps for b1.58. The paper states it is **"trained from scratch, with 1.58-bit weights and 8-bit activations"** — i.e. quantized from step 0, no full-precision warm-in. (A separate, *later* line of work — "16→1.58" continual-QAT, Nielsen et al. 2025 — does pretrain in 16-bit first, but that is **not** the original b1.58 recipe.) | **[PAPER SILENT] / NOT FOUND** for a λ ramp. "Quantized from scratch" is [b1.58] **[QUOTED]** §2. The 16→1.58 alternative is a *different* paper, noted only to disambiguate. |
| **Mixed precision (latent weights / grads / optimizer states kept high precision)** | **Yes.** Forward uses ternary weights + int8 activations, but **high-precision latent weights and gradients are maintained during training** (and updated via STE); only the quantized weights are kept for inference. This "mixed-precision QAT with shadow weights" is the BitNet training contract. | [BitNet] abstract **[SECONDARY]**: *"BitNet employs low-precision binary weights and quantized activations, while maintaining high precision … states and gradients during training."* Corroborated by [b1.58] §2 "trained from scratch" with latent weights **[QUOTED, qualitative]**. |
| **dtype (compute)** | Full-precision baseline is **FP16/BF16**; BitNet keeps FP16/BF16 latent/compute master copies. Exact training dtype (FP16 vs BF16) not pinned for the BitNet run. | [b1.58] **[QUOTED]** (FP16/BF16 baseline); exact training dtype **[PAPER SILENT]**. |
| **Seed control / # seeds** | **Not specified** in the papers. (Our repro's own north-star asks for 3 seeds; the *paper* does not prescribe a seed count.) | **[PAPER SILENT]**. |
| **Loss** | Standard autoregressive LM cross-entropy (implied by perplexity reporting; never restated as an equation). | [b1.58] §3 (perplexity) **[QUOTED, qualitative]**. |

---

## The 3 highest-confidence recipe facts (read first-hand from [b1.58])

1. **Weight quantization is `absmean` RoundClip to {−1,0,+1}.** Eqs (1)–(3), §2:
   `W̃ = RoundClip(W/(γ+ε), −1, 1)`, `RoundClip(x,a,b)=max(a,min(b,round(x)))`,
   `γ = (1/nm)Σ|W_ij|` — γ is a **single scalar per weight matrix** (whole-tensor mean-abs).
2. **Activations are 8-bit, absmax, per-token, symmetric [−Qb, Qb]** — and b1.58 explicitly
   **dropped** the original [0, Qb] pre-nonlinearity scaling to kill the zero-point. (§2)
3. **LLaMA-alike, biases removed:** RMSNorm + SwiGLU + RoPE, `nn.Linear → BitLinear`,
   **no bias terms anywhere**; trained **from scratch, quantized from step 0** (no λ warm-in). (§2)

A 4th, slightly lower-confidence but well-corroborated fact:

4. **Two-stage LR + two-stage weight decay, and BitNet tolerates a higher peak LR than its
   FP16 twin.** This is the schedule the community treats as *the* BitNet recipe — sourced to
   the author **[TrainTips]** companion (two-stage LR; two-stage WD "Figure 1d") and corroborated
   by [Reloaded] (LR 1e-2, "more robust to higher learning rate"). Exact numbers unverified.

---

## Recipe ingredients I could NOT find an authoritative source for

These are genuine gaps — the reproducer must choose them, and they are exactly where a
ternary-vs-FP16 loss gap can hide:

- **Exact peak LR, the stage-1→stage-2 drop point, and decay curve** — only the *shape*
  (two-stage) is sourced; the numbers were not read first-hand. **Needs the actual Training-Tips PDF.**
- **Stage-1 weight-decay value and the exact step where it goes to 0** — ~0.1→0 is the
  commonly cited pairing but unverified against the primary doc.
- **Which parameters are excluded from weight decay** (norms / embeddings) — **PAPER SILENT.**
- **Adam betas and eps** — **PAPER SILENT** in everything I could read.
- **Gradient clipping (value, or whether used at all)** — **PAPER SILENT.**
- **Training batch size and training sequence length** — **PAPER SILENT** (512 is an
  *inference* benchmark setting only).
- **Initialization (latent-weight std / scheme)** — **PAPER SILENT.**
- **A λ / gradual-quantization warmup** — **NOT FOUND**; the original recipe quantizes from
  step 0. (The "16→1.58" warm-in is a *different, later* paper, not original b1.58.)

> **Single most important caveat for the gap audit:** the optimizer/schedule half of the
> recipe (LR, WD, betas, warmup) is **not in the primary b1.58 paper at all** — it lives only
> in the author **Training-Tips companion**, which I read **second-hand** this pass. Before
> treating any LR/WD number here as ground truth, open
> `microsoft/unilm/.../The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf` directly and
> replace the **[SECONDARY]** rows with first-hand quotes.
