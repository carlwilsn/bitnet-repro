# Day 1: Minimal BitLinear Sanity Check

**Blog post:** [link TBD]

A minimal reproduction of the BitNet b1.58 `BitLinear` layer, dropped into a tiny GPT-style model (~5M params) trained on character-level Shakespeare.

## What this is

- `bitlinear.py`: Core `BitLinear` with absmean quantization and STE
- `model.py`: Tiny GPT with `BitLinear` replacing all internal `nn.Linear`
- `data.py`: Character-level Shakespeare loader
- `train.py`: Training loop with validation

## Run

```bash
# CPU sanity check (slow but works)
python train.py --device cpu --steps 100

# GPU training
python train.py --device cuda --steps 1000
```

## Key findings from Day 1

- BitLinear forward pass works correctly with STE
- Ternary weights {-1, 0, 1} are achieved via absmean scaling
- Training is stable on tiny model
- CPU training is ~27s/step → GPU essential for real runs
