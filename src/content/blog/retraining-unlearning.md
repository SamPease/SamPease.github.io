---
title: "Can You Get Forgotten Knowledge Back for Free? A Retraining Lens on Machine Unlearning"
description: "Using targeted retraining as a probe to study whether popular unlearning methods like NPO and RMU truly remove knowledge or merely suppress it."
date: 2026-06-09
tags: ["ai safety", "unlearning", "machine learning", "research"]
draft: False
---

*This write-up is in progress. Experiments are complete; analysis and visualization are ongoing.*

## Motivation

Machine unlearning — teaching a model to "forget" specific training data — is gaining traction as a tool for AI safety, privacy, and copyright compliance. But how do we know whether a model has truly forgotten something, versus learned to refuse to say it?

These are meaningfully different. A model that has genuinely had knowledge removed is safer in a deep sense: no amount of clever prompting or fine-tuning should recover it. A model that has merely learned to refuse is more fragile — the knowledge is still there, latent, and could be unlocked by an adversary with modest compute.

This project uses **retraining as a probe** for that question. The core idea: if you take an unlearned model and reteach it a small fraction of what it forgot, does the rest come back for free? If yes, the knowledge was never truly gone — the unlearning method likely just installed a suppression. If no, that's weak evidence the removal was more thorough, though absence of recovery under this one method is not proof the knowledge is gone.

This is also a concrete threat model, not just an academic probe. Consider open-weight models with safety-motivated unlearning applied — say, bioweapons synthesis routes removed before public release. A bad actor who already knows *some* of that information could fine-tune the model on the subset they have. If retraining on known harmful data recovers *unknown* harmful data for free, the unlearning provides much weaker safety guarantees than intended. The attack surface is whatever the adversary already knows — and the question is whether that's enough to unlock what they don't.

This work is part of the [BlueDot Impact](https://bluedot.org/) project sprint. I'm grateful for their guidance throughout.

---

## Background: TOFU, NPO, and RMU

### The TOFU Dataset

[TOFU](https://huggingface.co/datasets/locuslab/TOFU) (Task of Fictitious Unlearning) is a benchmark for evaluating unlearning methods on language models. It consists of synthetic author biographies — fictional people with fictional facts — making it possible to measure forgetting precisely without real-world privacy concerns.

The setup: fine-tune a base model on all 200 fictional authors, then apply an unlearning method to remove a subset (`forget10` = 10% of authors). A good unlearning method should make the model behave as if it was never trained on that 10%, while keeping the remaining knowledge intact.

Evaluation uses several metrics, but the most important are:
- **`forget_quality`**: how closely the unlearned model's behavior on the forget set resembles a model that was never trained on it. Higher is better (closer to 1 = genuinely forgotten).
- **`model_utility`**: general performance on retained knowledge. Should stay high.
- **`forget_truth_ratio`**: whether the model gives the *wrong* answer (the fake planted fact) versus a correct/hedged one. Lower = more forgotten.

### Unlearning Methods: NPO and RMU

Two methods are tested here, representing different philosophical approaches:

**NPO (Negative Preference Optimization)** treats unlearning like preference optimization. It pushes model outputs on the forget set toward "I don't know" behavior using a DPO-style objective. The concern with methods like this is that they may primarily be increasing a **refusal direction** — training the model to be confidently wrong or to hedge — rather than actually removing the underlying knowledge from the weights.

**RMU (Representation Misdirection for Unlearning)**, introduced alongside the WMDP benchmark, works differently: it fine-tunes the model to map forget-set inputs to a **random vector** in activation space, far from any coherent representation. The theory is that this more aggressively destroys the actual circuits encoding the forgotten knowledge, rather than just adding a behavioral refusal on top.

Both have pre-trained checkpoints available via [open-unlearning](https://huggingface.co/open-unlearning) on Hugging Face, trained on Llama-3.2-1B-Instruct with `forget10`. All experiments in this project use these checkpoints and run on a fork of the [open-unlearning training framework](https://github.com/locuslab/open-unlearning), which now lives at [github.com/SamPease/retrainingUnlearning](https://github.com/SamPease/retrainingUnlearning).

---

## The Experiment

### Design

The question: starting from an unlearned checkpoint (NPO or RMU), if I fine-tune it on a **subset** of the forgotten data (`forget01` = 1% of authors, or `forget05` = 5%), does the rest of the forgotten data (`forget10 minus forget01`, the "untaught" portion) come back for free?

This tests the two theories directly:

- **If NPO just amplified a refusal direction**, teaching back even a small subset should *reverse* that direction, and the remaining knowledge — which was always encoded in the weights — should become accessible again. The untaught data should recover.
- **If RMU genuinely destroyed circuits**, teaching back a subset should build new circuits for the taught data only. The untaught data stays inaccessible because there's nothing to recover — it requires learning from scratch. Free recovery should not occur.

The **retain90 baseline** (a model trained on 90% of authors, never taught the forget10) serves as a comparison: it genuinely never saw this data, so any "recovery" in that condition tells us what natural generalization looks like (a ceiling for expected free recovery).

### Models Evaluated

| Label | Checkpoint |
|---|---|
| Full | `open-unlearning/tofu_Llama-3.2-1B-Instruct_full` |
| Retain90 | `open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90` |
| NPO | `open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.1_alpha1_epoch10` |
| RMU | `open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr1e-05_layer10_scoeff100_epoch10` |

Retraining used TRL `SFTTrainer` + LoRA (r=16, alpha=32), trained for 20 epochs at lr=2e-4.

---

## An Engineering Detour: Why LoRA?

Before the recovery experiments could run, I spent significant time reproducing fine-tuning results from scratch. The original TOFU fine-tuning and unlearning checkpoints were trained with full-parameter fine-tuning. Early attempts to replicate these with full fine-tuning on Llama-3.2-1B-Instruct produced unstable and inconsistent results — metrics like `forget_quality` varied wildly across runs under nominally identical settings.

Switching to **LoRA for the retraining step** improved stability substantially. The best full fine-tuning calibration I found (lr=2e-5, 10 epochs, bs=8, grad_accum=4) reached metrics reasonably close to the HF reference:

| Metric | Local (lr=2e-5) | HF Reference |
|---|---:|---:|
| `forget_Q_A_Prob` | 0.838 | 0.881 |
| `forget_Q_A_ROUGE` | 0.758 | 0.816 |
| `extraction_strength` | 0.573 | 0.705 |
| `model_utility` | 0.572 | 0.599 |

But LoRA was more reliable for the retraining experiments because the goal is *relative comparison* (NPO vs RMU vs retain90) under a consistent procedure, not absolute reproduction of HF checkpoints.

This choice has a **philosophical wrinkle**: the original unlearning was done with full fine-tuning, which modifies all weights equally. LoRA retraining is necessarily low-rank and affects a subspace of the weight space. If RMU's unlearning is high-rank (destroying knowledge across a broad subspace), LoRA retraining may be *structurally incapable* of undoing it — not because the knowledge is gone, but because LoRA can't reach the right subspace. Conclusions about "new circuits vs. recovered knowledge" may partly reflect this methodological asymmetry. This is a live limitation and a direction for future work.

---

## Preliminary Results

*Charts are in progress. Numbers below are exact from evaluation runs.*

### The Retain90 Baseline (What "Never Knew It" Looks Like)

When retain90 is retaught `forget01`, how much of the remaining `forget10` comes back?

| Eval suite | `forget_quality` | `forget_truth_ratio` |
|---|---:|---:|
| Taught (`forget01`) | 5.4e-06 | 0.473 |
| Free-recovery (`forget10 minus forget01`) | **0.788** | 0.633 |

The high `forget_quality` (0.788) on the untaught data means the free-recovery data still looks "forgotten" — it behaves similarly to the retain model. This is expected: retain90 never had this knowledge, so there's nothing to recover. Free recovery is limited.

For `forget05`:

| Eval suite | `forget_quality` | `forget_truth_ratio` |
|---|---:|---:|
| Taught (`forget05`) | 3.0e-20 | 0.414 |
| Free-recovery (`forget10 minus forget05`) | **0.254** | 0.582 |

The free-recovery `forget_quality` drops to 0.254 — meaning some "leakage" even in the never-trained case, which sets a baseline for what incidental generalization looks like.

### NPO: Evidence of Suppression Rather Than Removal

| Condition | `forget_quality` (free-recovery) |
|---|---:|
| NPO baseline (no retraining) | 0.018 |
| NPO + retaught `forget01` | **6.5e-10** |
| NPO + retaught `forget05` | **3.2e-06** |
| Retain90 + retaught `forget01` (reference) | 0.788 |

After retraining NPO on just 1% of the forgotten authors, `forget_quality` on the *untaught* 9% collapsed from 0.018 to 6.5e-10. This is a dramatic recovery: the untaught data is now behaving like fully-learned knowledge. Compare this to retain90, where the same retraining left free-recovery `forget_quality` at 0.788 — because there was genuinely nothing to unlock.

This is consistent with the refusal-direction hypothesis: NPO appears to install a broad suppression that a small amount of retraining on the same distribution can undo, exposing knowledge that was never truly removed.

### RMU: A More Robust Picture

| Condition | `forget_quality` (free-recovery) |
|---|---:|
| RMU baseline (no retraining) | 6.8e-23 |
| RMU + retaught `forget01` | **3.4e-19** |
| RMU + retaught `forget05` | **1.7e-13** |

RMU's baseline free-recovery `forget_quality` is already near zero (6.8e-23), meaning the untaught data looks fully memorized even before retraining — which is a curious baseline behavior. After retraining on `forget01` or `forget05`, this value nudges slightly but stays orders of magnitude away from the retain90 reference (0.788). The untaught data does not flood back.

This is consistent with RMU having disrupted the representations more thoroughly. However, it comes with a significant cost: RMU retraining also substantially degraded utility. RMU baseline had `retain90_utility` of 0.656; after retraining on `forget10`, this dropped to 0.417.

### Utility Trade-offs

| Method | Variant | `model_utility` | `retain90_utility` |
|---|---|---:|---:|
| NPO | baseline | 0.432 | 0.349 |
| NPO | tuned on `forget01` | 0.516 | 0.528 |
| RMU | baseline | 0.588 | 0.656 |
| RMU | tuned on `forget01` | 0.517 | 0.520 |

NPO's baseline utility is weaker, but retraining *improves* it — a further signal that unlearning suppressed general capability alongside the specific knowledge. RMU's retraining degrades its already-strong baseline utility, suggesting the LoRA adaptation is not cleanly separating the target knowledge from the surrounding model behavior.

---

## What's Next

The raw numbers are collected. The immediate next steps:

1. **Visualization**: side-by-side bar charts comparing taught vs. free-recovery `forget_quality` across methods and retrain fractions — the clearest way to see the NPO/RMU contrast.

2. **White-box mechanistic analysis**: probing the internal representations at intermediate layers. If NPO unlearning works via a refusal direction, that direction should be detectable in activation space and should shift meaningfully when retraining undoes it. If RMU destroys circuits, intermediate activations on forget-set inputs should look incoherent relative to retain-set inputs, and retraining should build genuinely new structure rather than restore old structure.

3. **Broader method comparison**: extending to GradDiff and SimNPO checkpoints already evaluated, and potentially larger forget fractions.

---

## Acknowledgments

This work was done as part of the [BlueDot Impact](https://bluedot.org/) AI safety project sprint. Thanks to my cohort and advisors for guidance on experimental design and threat modeling. Code and data are available in the [project repository](https://github.com/SamPease/retrainingUnlearning).
