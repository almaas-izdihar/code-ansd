# ANSD (minimal)

Own PyTorch implementation of **ANSD — Adaptive Noise-Based Self-Distillation**
(Tan et al., J. King Saud Univ. CIS, 2026). The paper has **no public code**, so this is a
from-scratch reimplementation for the noise-dial experiment (noise / amplify pole, "D").

Part of skripsi-kd-v2 thesis. Parent plan:
`.claude/plans/20260718-1252-noise-dial-experiment.md`.

## Role in experiment
- **D = ANSD** (noise / amplify): inject adaptive Gaussian noise into an early feature block
  to build a noisy student view; distill it against the clean teacher view.
- Paired against **A = EMA-SKD** (temporal / suppress) to test the amplify-vs-suppress
  (noise-dial) question.

## Mechanism [SUMBER: ANSD raw md §3.2-3.3, Eq6-11]
```
clean forward  = teacher (stop-grad)
noisy forward  = student: F̃_l = F_l + λ·δ⊙M_norm  at Block1  (Eq8)
   M_norm = (F_l - min F_l)/(max F_l - min F_l + ε)   spatial min/max over H,W  (Eq7)
   δ ~ N(0,I)
loss = L_CE + α·KL(logit_teacher ‖ logit_student) + β·MSE(feat_teacher, feat_student)  (Eq11)
```

## Structure (mirrors EMA-SKD / code-7f2 layout)
```
main.py                 argparse entry (train/eval)
loader/cifar_dataloader CIFAR-100 loaders
loss/adaptive_noise     M_norm + noise injection (Eq7-8)
loss/ansd_loss          KL(logit) + MSE(feat) (Eq9-11)
models/resnet           ResNet-18 for CIFAR
utils/metric, etc       accuracy, seed, logging, checkpoint
experiments/            run outputs (gitignored)
```

## Setup
```shell
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Recipe [SUMBER: ANSD raw md L234]
ResNet-18 · CIFAR-100 · 100 epoch · batch 128 · SGD Nesterov 0.9 · wd 5e-4 ·
lr 0.1 decay /10 @30/60/90 · T∈{1,2} · α,β grid [0.25,2.0] step 0.25 · pad4+crop+flip.

## Sanity target (Stage 0a) [SUMBER: ANSD raw md L149]
ANSD Top-1 CIFAR-100 ResNet18 = 76.56 (+2.96 over baseline 73.60).
Also reproduce Table 1 (adaptive > fixed noise) and Table 2 (Block1 best).
