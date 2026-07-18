"""Adaptive spatial Gaussian noise injection for ANSD.
[SUMBER: ANSD raw md §3.2, Eq7-8]

Eq7:  M_norm = (F - min(F)) / (max(F) - min(F) + eps)   min/max over spatial (H,W), per sample-channel
Eq8:  F_tilde = F + lambda * delta ⊙ M_norm             delta ~ N(0, I)

Stronger noise on high-activation regions, weaker on low → informative noisy view.
"""
import torch

EPS = 1e-6


def adaptive_noise(feat, noise_lambda):
    """feat: [B, C, H, W]. Returns noise-perturbed feature (Eq8)."""
    if noise_lambda <= 0:
        return feat
    # spatial min/max over (H, W) per (B, C)  [SUMBER: Eq7, "aggregation along spatial dimensions (H, W)"]
    b, c, h, w = feat.shape
    flat = feat.view(b, c, -1)
    f_min = flat.min(dim=2, keepdim=True)[0].view(b, c, 1, 1)
    f_max = flat.max(dim=2, keepdim=True)[0].view(b, c, 1, 1)
    m_norm = (feat - f_min) / (f_max - f_min + EPS)          # Eq7
    delta = torch.randn_like(feat)                            # delta ~ N(0, I)
    return feat + noise_lambda * delta * m_norm               # Eq8
