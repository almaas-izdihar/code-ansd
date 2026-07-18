"""ANSD two-level distillation losses.
[SUMBER: ANSD raw md §3.3, Eq9-11, Algorithm 2]

L_logit = T^2 * KL(P_orig || P_noise)          teacher=orig (clean), student=noise. Eq9.
L_feat  = sum_{l=2}^{L} ||F_noise_l - F_orig_l||^2   blocks 2..L (skip block1 = noise site). Eq10.
Teacher (orig) is detached (frozen, not backpropped) [SUMBER: §4.1].
"""
import torch
import torch.nn.functional as F


def logit_distillation(z_orig, z_noise, T):
    """Eq9: T^2 * KL(P_orig || P_noise). Teacher = orig (detached)."""
    p_orig = F.softmax(z_orig.detach() / T, dim=1)
    log_p_noise = F.log_softmax(z_noise / T, dim=1)
    kl = F.kl_div(log_p_noise, p_orig, reduction='batchmean')
    return (T * T) * kl


def feature_distillation(feats_orig, feats_noise):
    """Eq10: sum ||F_noise_l - F_orig_l||^2 over blocks l=2..L (index 1..L-1 here).

    feats are [f1, f2, f3, f4]; block1 (index 0) is the noise-injection site and
    is skipped per Algorithm 2 (sum runs l=2..L).
    """
    loss = 0.0
    for f_orig, f_noise in zip(feats_orig[1:], feats_noise[1:]):
        loss = loss + F.mse_loss(f_noise, f_orig.detach(), reduction='mean')
    return loss
