---
title: "Reteaching Unlearning: Can Reteaching Uncover failures of Unlearning?"
description: "Using targeted retraining as a probe to study whether popular unlearning methods like NPO and RMU truly remove knowledge or merely suppress it."
date: 2026-06-09
tags: ["ai safety", "unlearning", "machine learning", "research"]
draft: False
---
# Reteaching Unlearning: Can Reteaching Uncover failures of Unlearning?

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
| RMU★ | `open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr5e-05_layer5_scoeff10_epoch10` |

A note on RMU checkpoint selection: the initially obvious choice (`layer10_scoeff100_lr1e-5`) turned out to have barely unlearned — its baseline `forget_Q_A_Prob` on the forget set was 0.834, nearly as high as the fully-trained model. Running a scan across six untested configurations identified `layer5_scoeff10_lr5e-5` (RMU★) as the best balance of genuine forgetting (`forget_Q_A_Prob ≈ 0.001`) and preserved utility (`model_utility = 0.55`). Its `forget_truth_ratio = 0.740` also matches the benchmark reference target of 0.760 closely. The lower steering coefficient (`scoeff10` vs `scoeff100`) appears to be the key regulator: high learning rate drives aggressive unlearning while the lower coefficient avoids utility collapse.

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

## Results

### The Retain90 Baseline: What "Never Knew It" Looks Like

Before comparing unlearning methods, it helps to know what *incidental* free-recovery looks like — the ceiling from natural cross-author generalization in a model that genuinely never saw the forget data.

| Eval suite | Taught fraction | Free-recovery `forget_Q_A_Prob` |
|---|---:|---:|
| Retain90 → `forget01` | 1% | 0.092 |
| Retain90 → `forget05` | 5% | 0.037 |

Both are close to floor (retain90's baseline on forget10 is `forget_Q_A_Prob = 0.116`). The model doesn't meaningfully recover untaught content because there's nothing latent to unlock. This anchors the comparison: any free-recovery above this floor in an unlearned model is structural reactivation, not ordinary generalization.

---

### Free Recovery on Untaught Content

The central question: after teaching a small fraction of the forgotten data back, does the rest come back for free?

**Chart 1** shows this for the 1%-training condition (teaching back `forget01`, then measuring on the remaining 9% of `forget10`):

![Free recovery on forget10-minus-forget01](/recovery_chart1_free_recovery_forget01.png)

| Condition | `forget_Q_A_Prob` (untaught 9%) | `extraction_strength` |
|---|---:|---:|
| Retain90 baseline (proxy) | 0.116 | 0.059 |
| Retain90 → `forget01` tuned | 0.092 | 0.082 |
| NPO baseline | 0.206 | 0.095 |
| NPO → `forget01` tuned | **0.458** | **0.193** |
| RMU★ baseline | 0.001 | 0.033 |
| RMU★ → `forget01` tuned | **0.260** | **0.148** |

Teaching 1% of the forgotten authors to NPO drives untaught-author recall from 0.206 → 0.458 — a jump more than three times the retain90 baseline. **Both methods show genuine free recovery.** The retain90 comparison is decisive: the same LoRA procedure applied to retain90 barely moves the needle (0.092), confirming the recovery in NPO and RMU★ is latent knowledge being unlocked, not cross-author generalization.

**Chart 2** shows the 5%-training condition (teaching `forget05`, measuring on the remaining 5% of `forget10`):

![Free recovery on forget10-minus-forget05](/recovery_chart2_free_recovery_forget05.png)

| Condition | `forget_Q_A_Prob` (untaught 5%) | `extraction_strength` |
|---|---:|---:|
| Retain90 baseline (proxy) | 0.116 | 0.059 |
| Retain90 → `forget05` tuned | 0.037 | 0.378 |
| NPO baseline | 0.209 | 0.095 |
| NPO → `forget05` tuned | 0.316 | 0.401 |
| RMU★ baseline | 0.002 | 0.033 |
| RMU★ → `forget05` tuned | **0.366** | **0.395** |

A counterintuitive result emerges: NPO's free-recovery `forget_Q_A_Prob` *drops* from 0.458 (1% training) to 0.316 (5% training), suggesting the recovery signal saturates and additional taught content doesn't amplify further unlocking. RMU★ goes in the opposite direction: 0.260 → 0.366. The `extraction_strength` metric also diverges from `forget_Q_A_Prob` for retain90 → `forget05` and NPO → `forget05` (both show high extraction strength but lower QAP), indicating the models are reconstructing latent signal without producing direct Q&A recall — a dissociation worth noting.

---

### Taught Set Performance

How well does each method learn the explicitly taught content?

![Taught set performance across all conditions](/recovery_chart3_taught_performance.png)

| Method | Training split | `forget_Q_A_Prob` | `extraction_strength` |
|---|---|---:|---:|
| Full HF reference | — | 0.881 | 0.705 |
| Retain90 → `forget01` | 1% | 0.754 | 0.301 |
| Retain90 → `forget05` | 5% | 0.857 | 0.698 |
| NPO → `forget01` | 1% | **0.929** | **0.768** |
| NPO → `forget05` | 5% | 0.906 | 0.685 |
| NPO → `forget10` | 10% | 0.864 | 0.591 |
| RMU★ → `forget01` | 1% | 0.897 | 0.586 |
| RMU★ → `forget05` | 5% | 0.890 | 0.659 |
| RMU★ → `forget10` | 10% | 0.840 | 0.572 |

NPO → `forget01` achieves the highest taught QAP in the matrix (0.929), slightly *exceeding* the full HF reference (0.881). This reflects the unlearning legacy: NPO suppressed representations but left latent structure intact, making them very easy to reactivate. The retain90 baseline's lower taught performance (0.754 for 1%) is expected — a neutral model must build new circuits, while unlearned models are recovering existing ones.

---

### Utility Trade-offs

A critical dimension is what recovery fine-tuning does to general capability.

![Utility across all conditions](/recovery_chart4_utility.png)

| Condition | `retain90_utility` | `retain_Q_A_Prob` |
|---|---:|---:|
| Retain90 HF | 0.591 | — |
| Full HF | 0.600 | — |
| NPO baseline | 0.349 | 0.423 |
| NPO → `forget01` tuned | **0.528** | 0.634 |
| NPO → `forget05` tuned | 0.455 | 0.453 |
| NPO → `forget10` tuned | 0.443 | 0.415 |
| RMU★ baseline | 0.510 | 0.608 |
| RMU★ → `forget01` tuned | 0.492 | 0.544 |
| RMU★ → `forget05` tuned | 0.439 | 0.425 |
| RMU★ → `forget10` tuned | 0.419 | 0.377 |

The two methods diverge here: NPO fine-tuning *recovers* utility. NPO's baseline utility is poor (0.349), a byproduct of its broad behavioral suppression, but teaching any subset raises it to 0.44–0.53. The forget01 run nearly reaches the retain90 HF level. RMU★ fine-tuning *degrades* utility from an already-reasonable baseline (0.510). Every tuned RMU★ run reduces utility, and the effect compounds with training-set size. The divergence is interpretable: NPO's low baseline has room to recover; RMU★'s higher baseline has room to fall.

---

### Transfer Rate Scaling

Normalizing free-recovery gain by taught gain reveals how efficiently each method transfers across authors:

$$\text{transfer rate} = \frac{\Delta\text{ free-rec QAP}}{\Delta\text{ taught QAP}}$$

![Transfer rate scaling from forget01 to forget05](/recovery_chart6_transfer_rate_scaling.png)

| Method | forget01 transfer | forget05 transfer | Direction |
|---|---:|---:|---|
| NPO | 35.8% | 15.4% | ↓ collapses |
| RMU★ | 28.9% | 41.0% | ↑ scales up |

NPO's transfer rate more than halves (36% → 15%) as training grows from 1% to 5%. RMU★ goes the other direction (+12 pp). A plausible interpretation: NPO's gradient-based forgetting disrupts individual fact representations somewhat independently, so teaching one author cues others only weakly, and that cross-cuing saturates quickly. RMU★'s representation-steering leaves a more coherent latent geometry — more taught content reactivates more of the shared structure, yielding increasing returns to scale.

---

### The Taught vs. Free-Recovery Summary

**Chart 5** visualizes the full recovery picture as a scatter: x-axis = taught QAP, y-axis = free-recovery QAP. Arrows connect each method's baseline to its tuned point.

![Taught vs free-recovery scatter](/recovery_chart5_taught_vs_free_recovery.png)

Three qualitatively distinct trajectories are visible:

- **NPO** arrows move lower-left → upper-right: low baseline recall on both axes, rising to high taught + moderate free-recovery. Fine-tuning simultaneously teaches explicit content *and* reactivates latent knowledge of related authors.
- **RMU★** arrows also move to high taught recall but with meaningful upward motion on the free-recovery axis — confirming that RMU★ shows genuine reactivation, not just new learning.
- **Retain90** points land in the bottom-right quadrant: high taught, near-zero free-recovery. The cross-author recovery signal is specific to models that previously encoded then suppressed the forget10 content.

---

## Interpretation

The core result: **both NPO and RMU★ show genuine free recovery**. Teaching a small fraction of forgotten content restores recall for the untaught fraction at a rate well above the retain90 floor. This means neither method fully erased the knowledge — it remained latent and recoverable.

What differs between them is the *structure* of that recovery:

- **NPO** installs a broad behavioral suppression. The suppressed knowledge is trivially reactivated by retraining on even a small in-distribution sample, and the transfer saturates rather than scaling with more data. Utility is collateral damage in the suppression — retraining recovers it alongside the knowledge.

- **RMU★** disrupts representations more deeply. The baseline is genuinely amnesic (`forget_Q_A_Prob ≈ 0.001`) but still vulnerable: teaching 1–5% of authors reactivates 26–37% of the untaught recall gain, and this transfer rate *increases* with more training data. The geometry of the suppression appears more coherent, so larger reactivation signals unlock more of it.

Neither method survives the threat model posed at the outset. An adversary who already knows some fraction of the forgotten content and can fine-tune the model will recover substantial untaught content — faster with NPO, with increasing efficiency with RMU★.

---

## What's Next

1. **White-box mechanistic analysis**: probing internal representations at intermediate layers. If NPO unlearning installs a refusal direction, that direction should be detectable in activation space and should shift when retraining undoes it. If RMU destroys circuits, intermediate activations on forget-set inputs should look incoherent relative to retain-set inputs, and retraining should build genuinely new structure rather than restore old structure.

2. **Broader method comparison**: extending to GradDiff and SimNPO checkpoints already evaluated, and potentially larger forget fractions.

---

## Acknowledgments

This work was done as part of the [BlueDot Impact](https://bluedot.org/) AI safety project sprint. Thanks to my cohort and advisors for guidance on experimental design and threat modeling. Code and data are available in the [project repository](https://github.com/SamPease/retrainingUnlearning).
