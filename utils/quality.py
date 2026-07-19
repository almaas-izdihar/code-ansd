"""Soft-label quality metrics for Stage 1 (axis probe).
Measured on the model's softmax predictions over CIFAR-100 test set.

- entropy:            mean -sum(p log p)  → how 'soft' the predictions are
- superclass_coh:     P(top-1 and top-2 predicted classes share a CIFAR-100 superclass)
                      → dark-knowledge fidelity (runner-up semantically related?)
                      random baseline ~ 4/99 ≈ 4%.
(ECE + top-1 already logged via metric.py; temporal-variance = Stage 1b.)
"""
import numpy as np

# CIFAR-100 fine-label (0..99) -> coarse superclass (0..19). Standard mapping.
CIFAR100_COARSE = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
    3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
    0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
    16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
    2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
    18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
]


def entropy_superclass(confidences):
    """confidences: list/array [N, C] of softmax probs. Returns (entropy, superclass_coh)."""
    p = np.clip(np.asarray(confidences, dtype=np.float64), 1e-12, 1.0)
    ent = float((-(p * np.log(p)).sum(axis=1)).mean())
    if p.shape[1] != len(CIFAR100_COARSE):
        return ent, float('nan')  # not CIFAR-100
    top2 = np.argsort(-p, axis=1)[:, :2]
    c = np.asarray(CIFAR100_COARSE)
    superclass_coh = float((c[top2[:, 0]] == c[top2[:, 1]]).mean())
    return ent, superclass_coh
