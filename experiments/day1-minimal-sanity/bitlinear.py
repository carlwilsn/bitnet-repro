"""
BitLinear implementation for BitNet b1.58
Day 1: minimal, correct, CPU-runnable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def absmean_quantize(x):
    """
    Quantize a tensor to ternary {-1, 0, 1} using absmean scaling.
    
    From the paper:
        gamma = mean(|x|)
        x_q = RoundClip(x / gamma, -1, 1)
    """
    # Compute the average absolute value across the entire tensor
    gamma = x.abs().mean().clamp_min(1e-5)  # clamp to avoid div-by-zero
    
    # Scale and round
    x_scaled = x / gamma
    x_q = torch.round(x_scaled)
    
    # Clip to {-1, 0, 1}
    x_q = torch.clamp(x_q, -1, 1)
    
    return x_q, gamma


class BitLinear(nn.Module):
    """
    A linear layer that uses ternary weights {-1, 0, 1} for the forward pass.
    
    During training:
      - We keep full-precision latent weights (self.weight).
      - On each forward pass, we quantize those weights to ternary using absmean.
      - We use the Straight-Through Estimator (STE) so gradients flow back
        to the latent weights despite the non-differentiable round+clip.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Latent full-precision weights (what optimizer actually updates)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        # 1. Quantize latent weights to ternary {-1, 0, 1}
        #    Detach the quantized values so the round/clamp doesn't create gradients,
        #    but subtract the quantized from latent and add it back so the gradient
        #    flows straight through to the latent weights (STE).
        w_q, gamma = absmean_quantize(self.weight)
        
        # Straight-Through Estimator trick:
        # Forward uses quantized weights. Backward sees identity.
        w_ste = self.weight + (w_q - self.weight).detach()
        
        # 2. Standard linear matmul with the STE weights
        out = F.linear(x, w_ste, self.bias)
        
        return out


# ---------- Sanity Test ----------
if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Create a BitLinear layer
    layer = BitLinear(10, 5)
    
    # Check that latent weights are NOT ternary (they're full precision)
    print("Latent weights unique values:", torch.unique(layer.weight).numel(), "(should be >> 3)")
    
    # Run forward pass
    x = torch.randn(2, 10)  # batch=2, features=10
    out = layer(x)
    print("Output shape:", out.shape)  # should be [2, 5]
    
    # Verify the quantized weights are actually ternary
    w_q, _ = absmean_quantize(layer.weight)
    unique_vals = torch.unique(w_q)
    print("Quantized weights unique values:", unique_vals)
    assert torch.all((unique_vals == -1) | (unique_vals == 0) | (unique_vals == 1))
    print("[OK] Quantized weights are strictly ternary {-1, 0, 1}")
    
    # Verify gradients flow back
    loss = out.sum()
    loss.backward()
    print("Weight grad shape:", layer.weight.grad.shape)
    print("Weight grad non-zero:", (layer.weight.grad != 0).any().item())
    print("[OK] Gradients flow to latent weights via STE")
