# BitNet Reproduction

Reproducing BitNet b1.58 from [The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764) — paper → code → distributed training → eval.

## North Star

Match reported perplexity within tolerance across 3 seeds at 0.5B params on torchtitan, with framework-side fixes upstreamed as merged PRs to pytorch/torchtitan.

## Current Status

| Date | Milestone | Status |
|------|-----------|--------|
| Day 1 | Minimal BitLinear + tiny GPT sanity check | ✅ |

## Structure

```
bitnet-repro/
├── src/
│   └── bitlinear.py          # Core BitLinear module (evolves)
├── experiments/
│   └── day1-minimal-sanity/  # Frozen artifacts with blog posts
├── scripts/                   # Helper scripts
└── README.md
```

## Quick Start

```bash
cd experiments/day1-minimal-sanity
pip install torch
python train.py --device cuda --steps 1000
```

## License

MIT
