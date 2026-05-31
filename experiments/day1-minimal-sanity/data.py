"""
Tiny character-level dataset: downloads Shakespeare text and tokenizes it.
"""

import os
import urllib.request
import torch


def get_shakespeare_data():
    """Download tiny Shakespeare text and return raw string."""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filepath = "shakespeare.txt"

    if not os.path.exists(filepath):
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, filepath)
        print("Done.")

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def get_dataset():
    """Return train/val tensors and encode/decode functions."""
    text = get_shakespeare_data()

    # Build character-level vocab
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)

    # 90/10 train/val split
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data, vocab_size, encode, decode


def get_batch(data, block_size, batch_size, device):
    """Sample a random batch from data."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


if __name__ == "__main__":
    train_data, val_data, vocab_size, encode, decode = get_dataset()
    print(f"Vocab size: {vocab_size}")
    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens: {len(val_data):,}")
    x, y = get_batch(train_data, block_size=128, batch_size=4, device="cpu")
    print(f"Batch shapes: x={x.shape}, y={y.shape}")
