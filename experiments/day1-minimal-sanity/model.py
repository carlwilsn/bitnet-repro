"""
Tiny GPT-style model using BitLinear for all internal projections.
~5M params, character-level, CPU-runnable for sanity checks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from bitlinear import BitLinear


def make_linear(quant, in_features, out_features, bias=False):
    """Factory: BitLinear (ternary) when quant=='bitnet', else plain nn.Linear (FP)."""
    if quant == "bitnet":
        return BitLinear(in_features, out_features, bias=bias)
    elif quant == "none":
        return nn.Linear(in_features, out_features, bias=bias)
    else:
        raise ValueError(f"unknown quant mode: {quant}")


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1, quant="bitnet"):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout

        # key, query, value projections
        self.c_attn = make_linear(quant, n_embd, 3 * n_embd, bias=False)
        # output projection
        self.c_proj = make_linear(quant, n_embd, n_embd, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(128, 128)).view(1, 1, 128, 128)
        )

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # causal self-attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.1, quant="bitnet"):
        super().__init__()
        # FFN up-projection
        self.c_fc = make_linear(quant, n_embd, 4 * n_embd, bias=False)
        # FFN down-projection
        self.c_proj = make_linear(quant, 4 * n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1, quant="bitnet"):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, quant=quant)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout, quant=quant)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TinyBitGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size=128,
        n_layer=4,
        n_head=4,
        n_embd=256,
        dropout=0.1,
        quant="bitnet",
    ):
        super().__init__()
        self.block_size = block_size
        self.quant = quant
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, dropout, quant=quant) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, BitLinear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is {self.block_size}"

        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(t, device=device))
        x = self.dropout(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    model = TinyBitGPT(vocab_size=65)
    print(f"Model params: {model.count_params():,}")
    x = torch.randint(0, 65, (2, 64))
    logits, loss = model(x, x)
    print(f"Output shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
