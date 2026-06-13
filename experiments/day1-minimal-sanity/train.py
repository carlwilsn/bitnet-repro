"""
Train TinyBitGPT on Shakespeare character-level data.
CPU-runnable for sanity; use --device cuda for real training.
"""

import argparse
import time
import torch
from model import TinyBitGPT
from data import get_dataset, get_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--steps", type=int, default=1000, help="training steps")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--quant", default="bitnet", choices=["bitnet", "none"],
                        help="'bitnet' = ternary BitLinear, 'none' = FP nn.Linear baseline")
    parser.add_argument("--log_interval", type=int, default=10, help="steps between train-loss prints")
    parser.add_argument("--eval_interval", type=int, default=100, help="steps between validation evals")
    parser.add_argument("--eval_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337,
                        help="seed for torch RNG (model init + batch sampling) so A/B arms are RNG-clean")
    parser.add_argument("--tag", default="",
                        help="optional suffix for the checkpoint filename to avoid arm collisions")
    args = parser.parse_args()

    # Seed-lock: torch.manual_seed governs both _init_weights (normal_) and
    # get_batch's torch.randint, so two arms with the same seed see identical
    # init and identical batch order -> the only variable left is the code path.
    torch.manual_seed(args.seed)

    device = args.device
    print(f"Using device: {device}")
    print(f"Config: quant={args.quant} lr={args.lr} steps={args.steps} seed={args.seed} "
          f"batch={args.batch_size} block={args.block_size}")

    # Load data
    train_data, val_data, vocab_size, encode, decode = get_dataset()
    print(f"Vocab size: {vocab_size}")
    print(f"Train tokens: {len(train_data):,}")

    # Build model
    model = TinyBitGPT(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=4,
        n_head=4,
        n_embd=256,
        dropout=0.1,
        quant=args.quant,
    ).to(device)
    print(f"Model params: {model.count_params():,}  | quant={args.quant}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training loop
    model.train()
    start_time = time.time()
    for step in range(args.steps):
        x, y = get_batch(train_data, args.block_size, args.batch_size, device)
        logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.eval_interval == 0:
            # Quick val eval
            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(args.eval_steps):
                    x, y = get_batch(val_data, args.block_size, args.batch_size, device)
                    _, loss = model(x, y)
                    val_losses.append(loss.item())
            val_loss = sum(val_losses) / len(val_losses)
            model.train()
            print(
                f"Step {step:4d} | train_loss={loss.item():.4f} | val_loss={val_loss:.4f} | time={time.time()-start_time:.1f}s"
            )

    # Final eval
    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(args.eval_steps * 2):
            x, y = get_batch(val_data, args.block_size, args.batch_size, device)
            _, loss = model(x, y)
            val_losses.append(loss.item())
    final_val = sum(val_losses) / len(val_losses)
    print(f"\nFinal val loss: {final_val:.4f}")
    print(f"Total time: {time.time()-start_time:.1f}s")

    # Save checkpoint
    ckpt_name = f"checkpoint_{args.quant}{('_' + args.tag) if args.tag else ''}.pt"
    torch.save(model.state_dict(), ckpt_name)
    print(f"Saved {ckpt_name}")


if __name__ == "__main__":
    main()
