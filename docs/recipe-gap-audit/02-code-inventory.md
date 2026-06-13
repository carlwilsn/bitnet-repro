# 02 — Code Inventory: What the Day-1 Training Code Actually Does

**Scope:** Literal, line-cited inventory of the `bitnet-repro` day-1 training recipe as
implemented today. This is the "what our code does" half of the recipe-gap audit; the
"what the paper prescribes" half is owned by another worker. **No behavior is inferred
that was not read directly in source.**

**Files audited (all under `DeepDL/bitnet-repro/`):**

- `src/bitlinear.py`
- `experiments/day1-minimal-sanity/bitlinear.py`
- `experiments/day1-minimal-sanity/model.py`
- `experiments/day1-minimal-sanity/train.py`
- `experiments/day1-minimal-sanity/data.py`
- `experiments/day1-minimal-sanity/RESULTS.md`, `README.md`

> **Note on the two BitLinear copies:** `src/bitlinear.py` and
> `experiments/day1-minimal-sanity/bitlinear.py` are **byte-for-byte identical**
> (verified via `diff` → `IDENTICAL`). The model imports the local experiments copy
> (`model.py:8` → `from bitlinear import BitLinear`). All line numbers below are
> identical in both files. No divergence to report.

---

## Recipe ingredient checklist

| Ingredient | What our code does | File:line + quoted snippet |
|---|---|---|
| **Learning rate (value/default)** | Single `--lr` flag, default `1e-3`. | `train.py:19` — `parser.add_argument("--lr", type=float, default=1e-3)` |
| **LR schedule** | **NOT PRESENT / silent.** No warmup, no decay, no cosine, no two-stage. The optimizer is constructed once and `optimizer.step()` is called raw in the loop; no scheduler object exists anywhere. | `train.py:58` — `optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)` ; loop `train.py:63–70`, only `optimizer.step()` at `train.py:69` (no `scheduler.step()`). |
| **Same LR for BitNet & FP arms?** | **Yes — identical flag, no arm-specific override.** The `--quant` choice does not touch LR. (RESULTS.md shows the operator had to manually drop BitNet to `3e-4` on the command line because `1e-3` thrashed the STE.) | `train.py:19` (single `--lr`); RESULTS.md:11–14 (1e-3 plateau vs 3e-4 manual fix). |
| **Optimizer** | `torch.optim.AdamW`. | `train.py:58` — `optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)` |
| **Betas** | **NOT PASSED → PyTorch AdamW default `(0.9, 0.999)`.** | `train.py:58` (no `betas=` arg). |
| **eps** | **NOT PASSED → AdamW default `1e-8`.** | `train.py:58` (no `eps=` arg). |
| **weight_decay** | **NOT PASSED → AdamW default `0.01`.** So a flat 0.01 decay silently applies to **every** parameter. | `train.py:58` (no `weight_decay=` arg). |
| **Weight-decay param grouping** | **NOT PRESENT / silent.** One flat group: `model.parameters()` passed directly. Norms, biases, and embeddings are **NOT excluded** from decay — they all get the default 0.01. No `param_groups`, no `no_decay` split anywhere. | `train.py:58` — `AdamW(model.parameters(), lr=args.lr)` |
| **Gradient clipping** | **NOT PRESENT / silent.** No `clip_grad_norm_` / `clip_grad_value_` call exists. Loop is `zero_grad → backward → step` only. | `train.py:67–69` — `optimizer.zero_grad()` / `loss.backward()` / `optimizer.step()` (nothing between backward and step). |
| **Weight quantization — gamma scope** | `absmean`: gamma = mean of `\|x\|` over the **ENTIRE tensor** (one scalar per weight matrix), **not** per-output-channel. | `bitlinear.py:20` — `gamma = x.abs().mean().clamp_min(1e-5)  # clamp to avoid div-by-zero` |
| **Weight quant — epsilon/clamp on gamma** | gamma floored at `1e-5` via `.clamp_min(1e-5)` to avoid div-by-zero. | `bitlinear.py:20` (same line). |
| **Weight quant — RoundClip vs round+clamp** | `round` then `clamp` to `[-1,1]`. Functionally a RoundClip(x/γ, −1, 1): `torch.round` then `torch.clamp(·, -1, 1)`. | `bitlinear.py:24` — `x_q = torch.round(x_scaled)` ; `bitlinear.py:27` — `x_q = torch.clamp(x_q, -1, 1)` |
| **Weight quant — STE formulation** | Classic detached STE: forward uses `w_q`, backward sees identity. | `bitlinear.py:63` — `w_ste = self.weight + (w_q - self.weight).detach()` then `bitlinear.py:66` — `out = F.linear(x, w_ste, self.bias)` |
| **Activation quantization** | **NOT PRESENT / silent — LOUDLY ABSENT.** `BitLinear.forward` quantizes **weights only**; the input activation `x` flows into `F.linear` **un-quantized** (full precision). There is **no** absmax/per-token 8-bit activation quant, no `Q_b` scaling, no quant of `x` anywhere in `forward` or the model. The "1.58-bit" here is weight-only. | `bitlinear.py:55–66` — forward body: `w_q, gamma = absmean_quantize(self.weight)` … `out = F.linear(x, w_ste, self.bias)`. `x` is used raw; no operation touches `x` before the matmul. |
| **Normalization (pre-quant / pre-BitLinear)** | Standard **`nn.LayerNorm`** in the pre-norm position (`ln_1` before attn, `ln_2` before MLP, `ln_f` final). **No RMSNorm. No SubLN.** Critically, the LayerNorm is applied at the **block boundary**, not immediately inside `BitLinear` before quantization — so there is **no normalization fused in front of the BitLinear matmul** the way SubLN/BitLinear specifies. | `model.py:84` — `self.ln_1 = nn.LayerNorm(n_embd)` ; `model.py:86` — `self.ln_2 = nn.LayerNorm(n_embd)` ; `model.py:115` — `self.ln_f = nn.LayerNorm(n_embd)` ; applied at `model.py:90–91` — `x = x + self.attn(self.ln_1(x))` / `x = x + self.mlp(self.ln_2(x))`. No norm inside `bitlinear.py`. |
| **BitLinear bias** | `BitLinear.__init__` **defaults `bias=True`** (creates a zero bias param). **However** every model call site passes **`bias=False`**, so in practice **no biases exist** on the projections. The `make_linear` factory also defaults `bias=False`. | Default: `bitlinear.py:42` — `def __init__(self, in_features, out_features, bias=True)`. Call sites force off: `model.py:12` — `def make_linear(quant, in_features, out_features, bias=False)` ; `model.py:31,33,68,70` all `bias=False`. The LM head `model.py:116` — `self.head = nn.Linear(n_embd, vocab_size, bias=False)`. ⇒ Matches paper (biasless) **by call-site convention, not by layer default.** |
| **Init — BitLinear latent weights** | Two-stage and effectively redundant: constructed at `randn * 0.02`, then `_init_weights` re-inits **the same** weight via `normal_(std=0.02)`. Both give std ≈ 0.02. | Construction: `bitlinear.py:48` — `self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)`. Re-init: `model.py:120–122` — `if isinstance(module, (nn.Linear, nn.Embedding, BitLinear)): torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)`. |
| **Init — other params** | Embeddings & `nn.Linear` weights → `normal_(std=0.02)`; `nn.Linear` biases (none, since bias=False) → zeros; LayerNorm uses PyTorch default (weight=1, bias=0, untouched). No GPT-2 residual-scaled init (`c_proj` not specially scaled). | `model.py:120–124` — `_init_weights`. |
| **Architecture — n_layer** | 4 | `train.py:48` / `model.py:100` — `n_layer=4` |
| **Architecture — n_head** | 4 | `train.py:48` / `model.py:101` — `n_head=4` |
| **Architecture — n_embd** | 256 | `train.py:48` / `model.py:102` — `n_embd=256` |
| **Architecture — block_size** | 128 | `model.py:99` — `block_size=128` ; train flag default `train.py:18` — `--block_size … default=128` |
| **Architecture — dropout** | 0.1 (attn, resid, embedding, MLP) — **note: paper-style BitNet pretraining typically uses dropout 0.** | `model.py:103` — `dropout=0.1` ; applied `model.py:42–43`, `model.py:72`, `model.py:112`. |
| **Vocab** | char-level Shakespeare, `vocab_size = 65` (derived from `set(text)`). | `data.py:42–43` — `chars = sorted(list(set(text)))` / `vocab_size = len(chars)`. RESULTS.md: ln(65)=4.17 reference. |
| **Param count** | ~3.22M (per RESULTS.md). | RESULTS.md:3 — "n_embd=256, 4 layers, 3.22M params". (README.md says "~5M" — **stale/looser estimate; trust RESULTS.md's 3.22M**.) |
| **Activation function** | **GELU** (`F.gelu`). **No SwiGLU / gated FFN.** Standard 4× MLP up/down. | `model.py:75` — `x = F.gelu(x)` ; FFN dims `model.py:68,70` (4×). |
| **Positional encoding** | **Learned** absolute position embedding (`nn.Embedding(block_size, n_embd)`). **No RoPE.** | `model.py:110` — `self.position_embedding = nn.Embedding(block_size, n_embd)` ; added `model.py:131–132`. |
| **Attention** | Manual causal self-attention (explicit `softmax`, tril mask buffer fixed at 128×128), not Flash/SDPA. Scale `1/√(head_dim)`. | `model.py:38–39` mask ; `model.py:51–57` math. |
| **Batch size** | default 32 (RESULTS runs use this). | `train.py:17` — `--batch_size … default=32` |
| **Sequence / block length** | 128 tokens. | `train.py:18` (`--block_size default=128`), `model.py:99`. |
| **Number of steps** | default 1000; RESULTS runs use 1000 and 3000. | `train.py:16` — `--steps … default=1000` ; RESULTS.md repro block uses `--steps 3000`. |
| **Token budget per run** | batch×block×steps = 32 × 128 × 3000 ≈ **12.3M tokens** for the 3000-step runs (≈ 4.1M tokens at 1000 steps). **Computed from the three cited flags; not a literal constant in code.** | `train.py:16–18` (steps/batch/block defaults) + RESULTS.md repro commands. |
| **λ / quant warmup (gradual quantization ramp)** | **NOT PRESENT / silent.** Quantization is full-strength from step 0; there is no λ schedule mixing FP↔ternary, no "warm up in FP then switch" logic. | `bitlinear.py:55–66` (forward always quantizes); no λ anywhere in repo. |
| **Mixed precision / autocast** | **NOT PRESENT / silent.** No `torch.autocast`, no `GradScaler`. Despite the doc calling the baseline "FP16," the FP arm is plain **fp32** `nn.Linear`; the "FP16" label in RESULTS.md/README is **a misnomer** — nothing sets fp16. | `train.py` (no autocast/GradScaler anywhere); `model.py` plain modules. |
| **dtype** | Default fp32 throughout (params created via `torch.randn` / default float; `.to(device)` only moves, doesn't cast). Data tensors are `torch.long` token ids. | `train.py:56` — `.to(device)` (no dtype cast) ; `data.py:46` — `torch.tensor(encode(text), dtype=torch.long)`. |
| **Seed control** | Single `torch.manual_seed(args.seed)` (default 1337) governs both init and batch sampling, so A/B arms are RNG-matched. | `train.py:34` — `torch.manual_seed(args.seed)` ; `train.py:25` default 1337. |
| **Loss** | Cross-entropy over flattened logits/targets. | `model.py:140–142` — `F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))`. |

---

## What gets quantized (precise map)

`make_linear(quant=…)` swaps **only** the four projection matmuls to ternary BitLinear
when `quant=="bitnet"` (`model.py:12–20`):

- Attention QKV: `c_attn` — `model.py:31`
- Attention output: `c_proj` — `model.py:33`
- MLP up: `c_fc` — `model.py:68`
- MLP down: `c_proj` — `model.py:70`

**Left full-precision in BOTH arms** (never quantized):

- Token embedding `model.py:109`, position embedding `model.py:110`
- All three LayerNorms `model.py:84,86,115`
- **LM head** `model.py:116` (plain `nn.Linear`, fp32) — note the output head is FP even in the BitNet arm.

---

## Summary of silent/absent items (recipe gaps from the code side)

These are the places the code **does nothing** where a BitNet b1.58 recipe usually does something:

1. **No activation quantization at all** — `BitLinear.forward` quantizes weights only; the input `x` enters `F.linear` in full precision (`bitlinear.py:55–66`). This is weight-only ternary, not the paper's W1.58A8 scheme.
2. **No LR schedule** — no warmup, no cosine decay, no two-stage LR; constant LR for the whole run (`train.py:58`, loop `:63–70`).
3. **Single flat AdamW config** — default `weight_decay=0.01` applied to **every** param including norms/biases/embeddings (no param grouping); betas/eps left at defaults; **no gradient clipping** (`train.py:58`, `:67–69`).
4. **No quantization warmup / λ ramp** — full-strength ternary from step 0 (`bitlinear.py:55–66`).
5. **No SubLN / norm fused before the BitLinear** — LayerNorm sits at block boundaries, not in front of the quantized matmul (`model.py:84,86` vs `bitlinear.py`).
6. **gamma is whole-tensor, not per-channel** (`bitlinear.py:20`).
7. **"FP16" is a misnomer** — no autocast/GradScaler; both arms run fp32 (`train.py`, no mixed precision).
8. **Architecture diverges from paper-style LLaMA recipe**: GELU MLP (not SwiGLU), learned abs pos-emb (not RoPE), dropout 0.1 (not 0), LM head stays FP (`model.py:75,110,103,116`).
