---
title: "Free Recovery: Does LLM Unlearning Actually Remove Knowledge?"
description: "When an 'unlearned' LLM is retrained on a small fraction of forgotten data, does the rest come back for free? For every method tested — NPO, RMU, GradDiff, and five others — the answer is yes."
date: 2026-06-09
tags: ["ai safety", "unlearning", "machine learning", "research"]
draft: False
---
# Free Recovery: Does LLM Unlearning Actually Remove Knowledge?

> **Summary:** Machine unlearning is being proposed as a safety tool for LLMs — remove dangerous knowledge before deploying open-weight models. This project tests whether it actually works, or whether it just teaches models to *act* like they've forgotten. The test: fine-tune an "unlearned" model on a small slice of what it supposedly forgot, then check whether the *rest* comes back for free. **It does.** For every method tested, teaching 1–5% of the forgotten data restores meaningful recall of untaught related content. The knowledge was never gone.
>
> The more surprising finding is *how* the recovery works. NPO and RMU — two methods selected because they represent opposite mechanistic theories of forgetting — leave different signatures. Teaching a small slice of the forgotten data substantially shifts probability distributions and representation-level membership inference scores for both methods on the *untaught* remainder, consistent with latent structure being reactivated rather than learned anew. At 1% teaching, strict generative recall (extraction strength) is modest; at 5%, both methods top out in the 0.12–0.13 range (the Full model ceiling is 0.71). All results are stable across three random seeds. None of the methods survive the proposed threat model.

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
- **`forget_Q_A_Prob` (QAP)**: the probability the model correctly answers direct questions about the forgotten content. Zero = fully forgotten; one = fully retained. This is the most intuitive single-number summary of whether a model "knows" something, and is the primary metric used in the recovery experiments below.

### Unlearning Methods: NPO and RMU

Two methods are tested here, representing different philosophical approaches:

**NPO (Negative Preference Optimization)** treats unlearning like preference optimization. It pushes model outputs on the forget set toward "I don't know" behavior using a DPO-style objective. The concern with methods like this is that they may primarily be increasing a **refusal direction** — training the model to be confidently wrong or to hedge — rather than actually removing the underlying knowledge from the weights.

**RMU (Representation Misdirection for Unlearning)**, introduced alongside the WMDP benchmark, works differently: it fine-tunes the model to map forget-set inputs to a **random vector** in activation space, far from any coherent representation. The theory is that this more aggressively destroys the actual circuits encoding the forgotten knowledge, rather than just adding a behavioral refusal on top.

These two methods represent the sharpest mechanistic contrast in the unlearning literature: one optimizes output behavior, the other steers internal representations. That contrast makes them the natural first test for whether the *mechanism* of forgetting predicts recovery behavior — the central question here. The experiment later expanded to six additional methods to test how broadly the pattern generalizes (see [Expanding to Other Methods](#expanding-to-other-unlearning-methods)).

Both NPO and RMU have pre-trained checkpoints available via [open-unlearning](https://huggingface.co/open-unlearning) on Hugging Face, trained on Llama-3.2-1B-Instruct with `forget10`. All experiments in this project use these checkpoints and run on a fork of the [open-unlearning training framework](https://github.com/locuslab/open-unlearning), which now lives at [github.com/SamPease/retrainingUnlearning](https://github.com/SamPease/retrainingUnlearning).

---

## The Experiment

### Design

Starting from an unlearned checkpoint, I fine-tune it on a **subset** of the forgotten data — `forget01` (1% of authors) or `forget05` (5%) — then measure recall on the *untaught* remainder of the forget set. This directly tests the two hypotheses:

- **If NPO installed a refusal direction**, teaching back even a small subset should reverse it, making the remaining knowledge — which was always in the weights — accessible again.
- **If RMU destroyed circuits**, teaching back a subset can only build new circuits for the taught authors. The untaught subset has nothing to recover from; it requires learning from scratch.

The **retain90 baseline** — a model trained on 90% of authors that genuinely never saw `forget10` — establishes what natural cross-author generalization looks like. Any free-recovery above this floor in an unlearned model is structural reactivation, not coincidence.

### Models Evaluated

| Label | Checkpoint |
|---|---|
| Full | `open-unlearning/tofu_Llama-3.2-1B-Instruct_full` |
| Retain90 | `open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90` |
| NPO | `open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.1_alpha1_epoch10` |
| RMU★ | `open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr5e-05_layer5_scoeff10_epoch10` |

A note on RMU checkpoint selection: the initially obvious choice (`layer10_scoeff100_lr1e-5`) turned out to have barely unlearned — its baseline `forget_Q_A_Prob` on the forget set was 0.834, nearly as high as the fully-trained model. Running a scan across six untested configurations identified `layer5_scoeff10_lr5e-5` (RMU★) as the best balance of genuine forgetting (`forget_Q_A_Prob ≈ 0.001`) and preserved utility (`model_utility = 0.55`). Its `forget_truth_ratio = 0.740` also matches the benchmark reference target of 0.760 closely. The higher learning rate (`5e-5`) drives more aggressive forgetting, while the lower steering coefficient (`scoeff10` vs `scoeff100`) prevents the utility collapse seen at higher values.

Retraining used TRL `SFTTrainer` + LoRA (r=16, alpha=32), trained for 20 epochs at lr=2e-4.

---

## Methodology Note: Why LoRA?

Before the recovery experiments could run, I spent significant time reproducing fine-tuning results from scratch. The original TOFU fine-tuning and unlearning checkpoints were trained with full-parameter fine-tuning. Early attempts to replicate these with full fine-tuning on Llama-3.2-1B-Instruct produced unstable and inconsistent results — metrics like `forget_quality` varied wildly across runs under nominally identical settings.

Switching to **LoRA for the retraining step** improved stability substantially. The best full fine-tuning calibration I found (lr=2e-5, 10 epochs, bs=8, grad_accum=4) reached metrics reasonably close to the HF reference:

| Metric | Local (lr=2e-5) | HF Reference |
|---|---:|---:|
| `forget_Q_A_Prob` | 0.838 | 0.881 |
| `forget_Q_A_ROUGE` | 0.758 | 0.816 |
| `extraction_strength` | 0.573 | 0.705 |
| `model_utility` | 0.572 | 0.599 |

Full-parameter fine-tuning achieved better absolute results (QAP=0.838, model_utility=0.572 vs the HF full model at 0.881/0.599), but applying these hyperparameters directly to the unlearned starting checkpoints (NPO, RMU★) produced poor results — each checkpoint required its own separate hyperparameter sweep to work reliably. LoRA with a single fixed configuration (lr=2e-4, 20 epochs, r=16/α=32) worked consistently across all starting checkpoints without per-checkpoint tuning, making it the practical choice for a controlled comparison. The tradeoff is a lower utility ceiling: the selected LoRA configuration reaches model_utility=0.351 on the calibration run, notably below the full-parameter best of 0.572. Since the experiments are designed for *relative comparison* (NPO vs RMU vs retain90 under identical procedure) rather than absolute reproduction, this is acceptable — but it is worth keeping in mind that reported utility values are systematically suppressed relative to what full fine-tuning would produce.

This choice has a **philosophical wrinkle**: the original unlearning was done with full fine-tuning, which modifies all weights equally. LoRA retraining is necessarily low-rank and affects a subspace of the weight space. If RMU's unlearning is high-rank (destroying knowledge across a broad subspace), LoRA retraining may be *structurally incapable* of undoing it — not because the knowledge is gone, but because LoRA can't reach the right subspace. Conclusions about "new circuits vs. recovered knowledge" may partly reflect this methodological asymmetry. This is a live limitation and a direction for future work.

---

## Results

### Fine-Tuning Setup: Hyperparameter Calibration

Before testing recovery from unlearned checkpoints, I needed a fine-tuning procedure that could reliably teach the forget set back to any starting checkpoint. To calibrate this, I fine-tuned the retain90 baseline — a model that genuinely never saw the forget10 data — on the full forget10 training split, sweeping learning rate and LoRA rank. The sweep optimized for one criterion: maximizing forget-set recall (`forget_Q_A_Prob`) relative to the fully-trained reference model.

The selected configuration (lr=2e-4, 20 epochs, LoRA r=16/α=32, batch=4, grad_accum=4) was fixed for all subsequent recovery experiments.

| Model | `forget_Q_A_Prob` | `extraction_strength` | `retain90_utility` | `retain_Q_A_Prob` |
|---|---:|---:|---:|---:|
| Full (HF reference) | 0.881 | 0.705 | 0.599 | 0.871 |
| retain90 → forget10 (selected hyperparams) | 0.790 | 0.590 | 0.351 | 0.168 |

The selected hyperparameters are aggressive: while forget-set recall reaches 0.790 (close to the full model's 0.881), `retain_Q_A_Prob` collapses from 0.880 to 0.168 — the model substantially forgets the retain90 authors in the process of learning the forget10 ones. This is a real cost of the chosen configuration, not a deliberate design choice; a more conservative learning rate would have preserved more retain-set knowledge at the expense of lower forget-set recall. In retrospect, a balanced objective (trading some QAP for less retain-set destruction) would have been preferable.

That said, the degradation matters less for the actual recovery experiments than it might appear. Those experiments fine-tune on forget01 or forget05 — subsets 2–10× smaller than the full forget10 set used here — and correspondingly show much milder retain-set interference (`retain_Q_A_Prob` ~0.45–0.63 across conditions). The calibration run here represents the upper bound of the forgetting effect, not the typical operating point.

There is also an incidental alignment with the threat model: a real adversary seeking to recover dangerous knowledge from an unlearned model would have no reason to preserve the model's general capability — they want to unlock specific suppressed content, not maintain a helpful assistant. So the aggressive hyperparameter choice is arguably realistic for the adversarial framing even if it was not chosen for that reason.

---

### Starting Checkpoints: Unlearning Method Selection

The recovery experiments require selecting the best available unlearned starting checkpoint for each method. This selection is meant to simulate the most favorable case for unlearning from a defender's perspective: the checkpoint that has best balanced genuine forgetting with preserved general capability. The **harmonic score** captures exactly this tradeoff:

$$\text{harmonic} = \frac{2 \cdot (1 - \text{QAP}) \cdot (\text{mu} / 0.591)}{(1 - \text{QAP}) + (\text{mu} / 0.591)}$$

where QAP = `forget_Q_A_Prob` (lower = more forgotten) and mu = `model_utility` normalized by 0.591, the Retain90 HF model's `retain90_utility` (first row of the table below). Utility matters here: a real defender deploying an unlearned model cares about preserving capability for legitimate uses, so a checkpoint that collapses utility to achieve low QAP is not a realistic deployment choice. Selecting by harmonic gives the unlearning methods their best possible starting point.

A small sample of available checkpoints from the open-unlearning HuggingFace collection was evaluated per method and ranked by harmonic score. The highest-scoring checkpoint for each method was selected.

| Checkpoint | `forget_Q_A_Prob` | `extraction_strength` | `retain90_utility` | `retain_Q_A_Prob` | Harmonic |
|---|---:|---:|---:|---:|---:|
| Retain90 (HF) | 0.116 | 0.059 | 0.591 | 0.880 | 0.938 |
| NPO (lr=1e-5, β=0.1, α=1, ep=10) | 0.208 | 0.095 | 0.432 | 0.423 | 0.760 |
| RMU★ (lr=5e-5, layer=5, scoeff=10, ep=10) | 0.001 | 0.033 | 0.550 | 0.607 | 0.964 |

A note on RMU: the most obvious published checkpoint (`layer10_scoeff100_lr1e-5`) barely unlearned — its `forget_Q_A_Prob` was 0.834, nearly identical to the fully-trained model. Scanning a handful of alternative configurations identified RMU★ as the best balance of genuine forgetting and preserved utility. For NPO, only one standard checkpoint was published; it was included despite a lower harmonic (0.760) because it represents the key mechanistic contrast to RMU.

The retain90 baseline serves as the reference floor: a model that has genuinely never seen the forget data. Any recovery signal above the retain90 ceiling in the unlearned models reflects latent knowledge being reactivated, not ordinary cross-author generalization.

---

### Validating the Fine-Tuning Procedure: Taught-Set Performance

Before examining free recovery, it is worth confirming that the fine-tuning procedure actually works — that the selected hyperparameters successfully teach each starting checkpoint what they are supposed to learn. Chart 7 shows the performance on the explicitly taught split for all three starting checkpoints across training sizes, against the full model reference.

![Taught-set performance across all conditions](/recovery_chart7_taught_performance.png)

| Condition | `forget_Q_A_Prob` | `extraction_strength` |
|---|---:|---:|
| Full model (HF reference) | 0.881 | 0.705 |
| **Retain90** | | |
| Taught 1% | 0.755 ± 0.004 | 0.304 ± 0.025 |
| Taught 5% | 0.837 ± 0.023 | 0.676 ± 0.020 |
| Taught 10% | 0.805 ± 0.017 | 0.611 ± 0.020 |
| **NPO** | | |
| Taught 1% | **0.927 ± 0.003** | **0.747 ± 0.018** |
| Taught 5% | 0.906 ± 0.007 | 0.677 ± 0.007 |
| Taught 10% | 0.861 ± 0.008 | 0.593 ± 0.020 |
| **RMU★** | | |
| Taught 1% | 0.895 ± 0.002 | 0.599 ± 0.009 |
| Taught 5% | 0.891 ± 0.004 | 0.690 ± 0.023 |
| Taught 10% | 0.842 ± 0.017 | 0.565 ± 0.025 |

The fine-tuning procedure works well across all conditions. NPO achieves the highest taught QAP at 1% (0.927, above the full model reference of 0.881), consistent with its suppression mechanism leaving knowledge structurally intact and easily reactivated. RMU★ reaches 0.895 after teaching just 2 authors despite starting from a near-zero baseline — a direct demonstration that the latent representation is still recoverable. Retain90's taught performance is modestly lower at 1% (0.755); it must build new representations rather than reactivate suppressed ones. All three converge by 10%, with diminishing returns between 5% and 10%.

Chart 8 shows the utility cost of this fine-tuning on the retain90 set — knowledge the models should be preserving throughout.

![Retain90 utility across all conditions](/recovery_chart8_retain90_utility.png)

| Condition | `retain90_utility` | `retain_Q_A_Prob` |
|---|---:|---:|
| **Retain90** | | |
| Baseline | 0.591 | 0.880 |
| Taught 1% | 0.501 ± 0.003 | 0.571 ± 0.008 |
| Taught 5% | 0.316 ± 0.006 | 0.213 ± 0.009 |
| Taught 10% | 0.286 ± 0.005 | 0.174 ± 0.005 |
| **NPO** | | |
| Baseline | 0.349 | 0.423 |
| Taught 1% | **0.527 ± 0.001** | **0.631 ± 0.004** |
| Taught 5% | 0.454 ± 0.004 | 0.447 ± 0.007 |
| Taught 10% | 0.438 ± 0.004 | 0.408 ± 0.006 |
| **RMU★** | | |
| Baseline | 0.510 | 0.607 |
| Taught 1% | 0.489 ± 0.004 | 0.533 ± 0.011 |
| Taught 5% | 0.440 ± 0.001 | 0.426 ± 0.001 |
| Taught 10% | 0.417 ± 0.005 | 0.377 ± 0.009 |


The NPO utility trajectory is unusual: baseline utility is the lowest of the three (0.349 / 0.423) due to NPO's broad behavioral suppression, but teaching just 1% of forget content raises it to 0.527 / 0.631 — surpassing RMU★ and nearly reaching the retain90 baseline. This simultaneous recovery of taught content *and* general retain-set capability is a key signature of NPO's mechanism: the suppression installed by NPO is broad, so reversing even a small part of it restores capability more generally. RMU★ and retain90, by contrast, show monotone decline in utility as training-set size grows. Retain90's steep retain_Q_A_Prob drop (0.880 → 0.168 at 10%) reflects the aggressive hyperparameter issue discussed in the calibration section above; the same effect is present but muted at the smaller training scales used in the main recovery experiments.

---

### Free Recovery on Untaught Content

The central question: after teaching a small fraction of the forgotten data back, does the rest come back for free? Each x-axis point evaluates a different model on the portion of `forget10` it was *not* trained on — the baseline is the full `forget10` set with no tuning, the 1% point is performance on `forget10 − forget01` after training on `forget01`, and the 5% point is `forget10 − forget05` after training on `forget05`. Retain90 is the control: a model that never encoded the forget data, run through the same fine-tuning procedure. Four metrics tell a layered story about what is and is not recovered.

**Probability shift**

`forget_Q_A_Prob` (QAP) scores the model's per-token probability on the correct answer — not whether it generates that answer, but whether it assigns it high likelihood. Teaching a small fraction of the forget set causes the models' distributions to shift toward the correct answers for untaught authors.

![Probability shift on untaught forget10 content](/recovery_chart_free_recovery_qap.png)

| Model | Baseline | Taught 1% | Taught 5% |
|---|---:|---:|---:|
| Full | 0.881 | 0.878 | 0.876 |
| Retain90 | 0.084 | 0.094 ± 0.001 | 0.037 ± 0.002 |
| NPO | 0.208 | **0.454 ± 0.004** | 0.318 ± 0.003 |
| RMU★ | 0.001 | 0.256 ± 0.003 | **0.379 ± 0.013** |

NPO nearly doubles from 0.208 → 0.454 after teaching 1%. RMU★ rises from near-zero (0.001) to 0.256 at 1% and 0.379 at 5%. Retain90 is flat and slightly declining, confirming the signal in NPO and RMU★ is specific to models that previously encoded this content — Retain90 has no latent structure to reactivate.

**Suppression undone, not information recovered**

`forget_Q_A_ROUGE` measures overlap between the model's *generated* answer and the correct answer. The pattern here is different from QAP: rather than recovery, it shows the undoing of an artifact introduced by unlearning itself.

![ROUGE on untaught forget10 content](/recovery_chart_free_recovery_rouge.png)

| Model | Baseline | Taught 1% | Taught 5% |
|---|---:|---:|---:|
| Full | 0.816 | 0.812 | 0.806 |
| Retain90 | 0.372 | 0.364 ± 0.000 | 0.352 ± 0.001 |
| NPO | 0.186 | 0.409 ± 0.000 | 0.408 ± 0.005 |
| RMU★ | 0.096 | 0.390 ± 0.002 | 0.423 ± 0.006 |

Retain90 — a model that never saw the forget10 authors — generates outputs with ROUGE ≈ 0.37 purely from the biographical format it learned on the other 90% of TOFU. That is the floor: what any TOFU-trained model produces when asked about unseen authors. NPO and RMU★ at baseline fall *below* that floor (0.19 and 0.10 respectively) because their unlearning suppressed even the general biographical generation ability. Fine-tuning on 1–5% of the forget set brings both back up to roughly the Retain90 level (0.39–0.42). This is not recovered knowledge — it is the removal of a suppression artifact, restoring the models to what they would have looked like had unlearning never touched them. The Full model ceiling at 0.82 remains far off.

**Representation-level alignment**

`privleak` compares the Min-K% membership inference attack AUC on the forget set to the same attack on the retain90 reference's retain set. A value near zero means the forget set examples are as indistinguishable as the reference's retain examples; negative values mean the forget set is identifiable; positive values mean it is *harder* to identify than the reference's retain content.

![Representation-level alignment on untaught forget10 content](/recovery_chart_free_recovery_privleak.png)

| Model | Baseline | Taught 1% | Taught 5% |
|---|---:|---:|---:|
| Full | −99.5 | −99.5 | −99.5 |
| Retain90 | 0.0 | −0.2 ± 0.3 | +0.9 ± 1.3 |
| NPO | −52.4 | −93.6 ± 0.0 | −93.2 ± 0.5 |
| RMU★ | **+61.1** | −80.9 ± 0.7 | −96.2 ± 0.5 |

RMU★ at baseline is +61.1 — the only condition where the forget set is *harder* to identify than the reference's retain examples. RMU's representation steering has misdirected the forget10 activations to the point where they look like ordinary content. After teaching just 1%, privleak drops from +61.1 to −80.9, a 142-point swing on the untaught examples alone. NPO, which began with a suppression fingerprint already (−52.4), drops further to −93.6. Retain90's privleak stays near zero (−0.2 at 1%, +0.9 at 5%) — fine-tuning on a subset of forget10 does not meaningfully shift the MIA signature of the untaught complement for a model that never encoded that content. This cleanly separates format learning (Retain90) from latent-structure reactivation (NPO, RMU★): only the unlearned models show the representational reorganization that makes the untaught content more identifiable under Min-K% attack. At 5%, RMU★ deepens to −96.2, approaching the full model's −99.5.

**Generative recall**

`extraction_strength` (ES) asks whether this reorganization has translated into the model actually reproducing correct answers via greedy decoding. A score of `1 − k/n` requires the model to exactly predict every answer token from position k onward. It is the strictest test of knowledge extraction.

![Generative recall on untaught forget10 content](/recovery_chart_free_recovery_es.png)

| Model | Baseline | Taught 1% | Taught 5% |
|---|---:|---:|---:|
| Full | 0.705 | 0.705 | 0.705 |
| Retain90 | 0.055 | 0.057 ± 0.001 | 0.052 ± 0.001 |
| NPO | 0.095 | 0.132 ± 0.002 | 0.119 ± 0.008 |
| RMU★ | 0.033 | 0.096 ± 0.003 | 0.131 ± 0.006 |

Retain90's ES barely moves (0.055 → 0.057 → 0.052): teaching 1–5% of forget10 does not unlock generative extraction for a model that genuinely never encoded it. NPO rises modestly at 1% (0.132) but does not continue rising at 5% (0.119 — essentially flat). RMU★ shows a consistent increase at 5% (0.131 ± 0.006), slightly above NPO and well separated from Retain90; all three seeds fall within a tight range (0.126–0.138). At 1%, both unlearned models are in the 0.09–0.13 range, well above Retain90 but far below the Full model's 0.705. ES recovery is modest and consistent across methods and seeds — no method approaches the Full model ceiling, and no seed produces an outlier.

---

## Seed Robustness

Three independent replications (seeds 42, 123, 456) were run for all conditions, evaluated on the taught split, the free-recovery remainder, and the retain90 utility set. The mean ± std values from these runs are reported throughout the results tables above and reflected in the shaded bands on each chart.

Across all 18 conditions and all metrics, standard deviations are small: ≤ 0.025 on the taught set, ≤ 0.013 on the free-recovery set (including extraction_strength), and ≤ 0.006 on retain90 utility. No qualitative conclusion in the results section changes depending on which seed is used.

---

## Expanding to Other Unlearning Methods

With the NPO/RMU comparison established and replicated, the experiment extended to six additional unlearning methods with checkpoints available in the open-unlearning HuggingFace collection. The goal was to test whether the free-recovery patterns generalize across the broader landscape of published approaches. Expanded versions of Charts 7, 8, and 1a–1d showing all nine methods are displayed in the Recovery Results subsection below.

### Methods

**GradDiff (Gradient Difference)** is the most direct extension of gradient ascent — it maximizes the forget-set loss while minimizing the retain-set loss simultaneously. The retain term acts as a regularizer preventing the utility collapse that plain gradient ascent causes.

**UNDIAL (Unlearning via Self-Distillation on Adjusted Logits)** reframes unlearning as a stable *minimization* rather than loss maximization. It takes the model's current logit distribution, zeroes out the probability mass on the correct token, and fine-tunes toward this adjusted distribution — sidestepping the instability of gradient ascent entirely.

**AltPO (Alternate Preference Optimization)** combines negative feedback on the forget set with positive in-domain feedback, encouraging plausible alternative responses rather than just suppressing correct ones. This addresses the incoherent-output failure mode of pure negative-feedback methods.

**SimNPO (Simple Negative Preference Optimization)** is a follow-up to NPO addressing "reference model bias" — the uneven gradient weighting caused by NPO's use of the pre-unlearning model as a reference. SimNPO eliminates the reference model entirely, weighting gradients by the current model's loss rather than a ratio to a fixed reference.

**IdkNLL** fine-tunes the model to output "I don't know" responses on forget-set queries via standard cross-entropy loss — essentially supervised fine-tuning with correct answers replaced by IDK responses.

**IdkDPO** achieves the same IDK behavioral goal via a DPO-style objective: simultaneously increasing likelihood of refusal responses and decreasing likelihood of correct answers. A softer, preference-grounded version of IdkNLL.

### Checkpoint Selection

For each method, six candidate checkpoints were evaluated (varying learning rate and regularization) and ranked by a harmonic metric balancing forget quality and preserved utility:

| Method | Baseline QAP | `model_utility` | Harmonic |
|---|---:|---:|---:|
| GradDiff★ | 0.000 | 0.565 | 0.977 |
| SimNPO★ | 0.075 | 0.584 | 0.955 |
| AltPO★ | 0.070 | 0.572 | 0.949 |
| IdkDPO★ | 0.135 | 0.560 | 0.904 |
| UNDIAL★ | 0.081 | 0.502 | 0.883 |
| IdkNLL★ | **0.539** | 0.535 | 0.611 |

*Reference: NPO harmonic = 0.760, RMU★ harmonic = 0.964.*

IdkNLL is a notable outlier: its best candidate still recalls >54% of forget-set content at baseline. The IdkNLL approach redirects surface outputs without disrupting underlying representations enough to suppress direct QA recall.

### Recovery Results

The same TRL+LoRA recovery procedure (20 epochs, lr=2e-4, seed=42) was applied to each method's best checkpoint.

![Taught-set performance across all 9 methods](/recovery_chart7_taught_performance_expanded.png)

![Retain90 utility across all 9 methods](/recovery_chart8_retain90_utility_expanded.png)

![Probability shift on untaught forget10 — all 9 methods](/recovery_chart_expanded_qap.png)

![ROUGE on untaught forget10 — all 9 methods](/recovery_chart_expanded_rouge.png)

![Representation-level alignment on untaught forget10 — all 9 methods](/recovery_chart_expanded_privleak.png)

![Generative recall on untaught forget10 — all 9 methods](/recovery_chart_expanded_es.png)

**Free recovery generalizes across all methods tested.** For every method except IdkNLL, teaching 1–5% of the forgotten data raises untaught recall above baseline on at least one metric. Three sub-patterns are visible in the QAP charts:

**Behavioral-suppression cluster** (AltPO, GradDiff, SimNPO, UNDIAL): modest rise at 1% (QAP 0.09–0.19), then collapse at 5% back near baseline — matching the shape of NPO's trajectory but at lower absolute levels. The recovery mechanism saturates quickly. GradDiff is the most resilient: QAP rises from 0.000 at baseline to 0.125 at 1% and holds to 0.107 at 5%, notable because it achieves this despite perfect baseline forgetting.

**Near-zero** (IdkDPO): QAP 0.135 at baseline, 0.133 at 1%, 0.088 at 5% — essentially flat. The DPO-style IDK alignment appears to compartmentalize forgetting more completely than the behavioral-suppression methods, leaving little recoverable latent structure despite a higher baseline QAP than most others.

**Anti-transfer** (IdkNLL): QAP 0.539 at baseline (never truly forgot), drops to 0.419 at 1% and 0.277 at 5%. Teaching forget01 content *reduces* untaught recall rather than raising it. The IDK surface-behavior layer installed by IdkNLL appears to interfere with adjacent representations; SFT disrupts this layer in a way that selectively degrades adjacent content access rather than restoring it.

---

## Discussion

**The threat model holds for every method tested.** Teaching a small fraction of forgotten content restores recall for the untaught fraction well above what a model that genuinely never learned the data can produce. No method fully erased the knowledge — it remained latent and recoverable. An adversary who already possesses some fraction of the "removed" content and can fine-tune the open-weight model will recover untaught related content for free. How much, and how efficiently, depends on which unlearning method was applied.

The more interesting finding is the split in recovery structure:

**One cluster** — NPO, AltPO, SimNPO, UNDIAL, and GradDiff — shows positive free-recovery recall at 1% teaching that collapses or disappears at 5%. The pattern is consistent with these methods primarily installing *behavioral suppression*: teaching a small in-distribution sample is enough to unlock latent structure, but the unlocking mechanism saturates quickly. NPO shows the sharpest rise at 1% (QAP 0.21 → 0.45); GradDiff is the most resilient within the cluster (QAP holds from 0.125 at 1% to 0.107 at 5%) and the one most likely to be doing something mechanistically different.

**RMU★ is the sole exception.** Despite starting from genuine amnesia (QAP ≈ 0.001), its free-recovery QAP *increases* from 0.256 to 0.379 as the teaching set grows from 1% to 5% — while every other method holds flat or collapses. If RMU's representation-steering leaves a more coherent latent geometry than gradient-based methods, then larger reactivation signals would naturally unlock more of the shared structure — which is exactly what the data shows.

**IdkDPO** occupies a third position: near-zero transfer throughout. The DPO-style alignment toward IDK responses seems to compartmentalize forgetting more completely, leaving little transferable latent structure — even though its baseline QAP (0.135) is higher than RMU★'s.

**IdkNLL** is the odd one out: it never suppresses recall at baseline (QAP = 0.54) and shows *anti-transfer* — teaching content causes adjacent recall to fall rather than rise. The IDK surface-behavior layer appears to actively interfere with adjacent representations rather than coexisting with them.

The mechanistic story is not settled. The most direct next test is to probe the internal representations: if NPO installs a refusal direction, it should be detectable as a linear feature in activation space that disappears when retraining undoes it. If RMU disrupts circuits without leaving a coherent recoverable geometry, intermediate activations on forget-set inputs should look incoherent in a way that retraining replaces rather than restores. The diverging recovery trajectories give a behavioral prediction to anchor the mechanistic investigation.

---

## Open Questions

**White-box mechanistic analysis** is the natural next step. The behavioral results give sharp predictions: NPO should show a detectable "refusal direction" in intermediate activations that shifts when retraining undoes it; RMU should show incoherent activations on forget-set inputs that retraining replaces (rather than restores). Confirming or falsifying these predictions would move the explanation from behavioral pattern to mechanistic claim, and would explain why RMU's free-recovery recall scales with training-set size while every other method's collapses.

---

## Acknowledgments

This work was done as part of the [BlueDot Impact](https://bluedot.org/) AI safety project sprint. Thanks to my cohort and advisors for guidance on experimental design and threat modeling. Code and data are available in the [project repository](https://github.com/SamPease/retrainingUnlearning).
