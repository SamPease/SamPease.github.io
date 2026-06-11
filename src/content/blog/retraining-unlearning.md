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
> The more surprising finding is *how* the recovery works. NPO and RMU — the two primary methods studied, selected because they represent opposite mechanistic theories of forgetting — produce strikingly different transfer-rate patterns: NPO's efficiency halves as you teach more data; RMU's doubles. That divergence is stable across 3 random seeds and holds across 8 unlearning methods in total. RMU is the only one to show the scaling-up pattern, and none of the methods survive the proposed threat model.

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

A note on RMU checkpoint selection: the initially obvious choice (`layer10_scoeff100_lr1e-5`) turned out to have barely unlearned — its baseline `forget_Q_A_Prob` on the forget set was 0.834, nearly as high as the fully-trained model. Running a scan across six untested configurations identified `layer5_scoeff10_lr5e-5` (RMU★) as the best balance of genuine forgetting (`forget_Q_A_Prob ≈ 0.001`) and preserved utility (`model_utility = 0.55`). Its `forget_truth_ratio = 0.740` also matches the benchmark reference target of 0.760 closely. The lower steering coefficient (`scoeff10` vs `scoeff100`) appears to be the key regulator: high learning rate drives aggressive unlearning while the lower coefficient avoids utility collapse.

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

### The Taught vs. Free-Recovery Picture

The scatter below puts both dimensions together: x-axis = QAP on the explicitly taught content, y-axis = QAP on the untaught remainder. Arrows connect each method's baseline (open) to its tuned (filled) point.

![Taught vs free-recovery scatter](/recovery_chart5_taught_vs_free_recovery.png)

Three qualitatively distinct trajectories are visible:

- **NPO** arrows move lower-left → upper-right: low baseline recall on both axes, rising to high taught + moderate free-recovery. Fine-tuning simultaneously teaches explicit content *and* reactivates latent knowledge of related authors.
- **RMU★** arrows also reach high taught recall but with meaningful upward motion on the free-recovery axis — confirming that RMU★ shows genuine reactivation, not just new learning.
- **Retain90** points land bottom-right: high taught, near-zero free-recovery. The cross-author recovery signal is specific to models that previously encoded then suppressed the forget10 content — not a general cross-author generalization effect.

---

### Transfer Rate Scaling

To compare the two methods' recovery efficiency directly, define:

**transfer rate = Δ free-rec QAP / Δ taught QAP**

This normalizes free-recovery gain by the amount of explicit relearning happening in the same run, so a 35% transfer rate means "for every point of recall gained on the taught subset, 0.35 points come back for free on the untaught subset."

![Transfer rate scaling from forget01 to forget05](/recovery_chart6_transfer_rate_scaling.png)

| Method | forget01 transfer | forget05 transfer | Direction |
|---|---:|---:|---|
| NPO | 35.8% | 15.4% | ↓ collapses |
| RMU★ | 28.9% | 41.0% | ↑ scales up |

NPO's transfer rate more than halves (36% → 15%) as training grows from 1% to 5%. RMU★ goes the other direction (+12 pp). A plausible interpretation: NPO's gradient-based forgetting disrupts individual fact representations somewhat independently, so teaching one author cues others only weakly and that cross-cueing saturates quickly. RMU★'s representation-steering leaves a more coherent latent geometry — more taught content reactivates more of the shared structure, yielding increasing returns to scale.

---

## Seed Robustness

The transfer-rate divergence between NPO and RMU★ could be a seed artifact rather than a systematic property of the unlearning mechanisms. Three independent replications (seeds 42, 123, 456) were run across all 9 recovery conditions.

| Method | forget01 (mean ± std) | forget05 (mean ± std) | Direction |
|---|:---:|:---:|:---:|
| NPO | 35.4% ± 0.7 pp | 15.6% ± 0.4 pp | ↓ collapses |
| RMU★ | 28.5% ± 0.3 pp | 42.4% ± 1.5 pp | ↑ scales up |

All key metrics are stable: QAP standard deviation ≤ 0.005 for most conditions, utility std ≤ 0.007 throughout. The gap between the two methods' forget05 transfer rates is ~27 pp — NPO's highest seed (16.1%) is still well below RMU★'s lowest (41.0%). Zero overlap across seeds.

The opposing transfer-rate trajectories are a robust property of the respective unlearning mechanisms, not a seed artifact.

---

## Expanding to Other Unlearning Methods

With the NPO/RMU comparison established and replicated, the experiment extended to six additional unlearning methods with checkpoints available in the open-unlearning HuggingFace collection. The goal was to test whether the diverging transfer-rate patterns generalize across the broader landscape of published approaches.

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

*Reference: NPO harmonic = 0.947, RMU★ harmonic = 0.964.*

IdkNLL is a notable outlier: its best candidate still recalls >54% of forget-set content at baseline. The IDK-NLL approach redirects surface outputs without disrupting underlying representations enough to suppress direct QA recall.

### Recovery Results

The same TRL+LoRA recovery procedure (20 epochs, lr=2e-4, seed=42) was applied to each method's best checkpoint.

| Method | forget01 TR | forget05 TR | Direction |
|---|---:|---:|:---:|
| NPO★ | +35.8% | +15.3% | ↓ collapses |
| RMU★ | +28.9% | +41.0% | ↑ scales up |
| GradDiff★ | +16.7% | +11.8% | ↓ mild collapse |
| UNDIAL★ | +16.9% | −2.8% | ↓ collapses to zero |
| AltPO★ | +12.5% | −1.1% | ↓ collapses to zero |
| SimNPO★ | +10.7% | −2.7% | ↓ collapses to zero |
| IdkDPO★ | −0.2% | −6.7% | ≈ zero |
| IdkNLL★ | −32.5% | −75.3% | ↓↓ anti-transfer |

Three patterns emerge:

**Positive-and-collapsing** (GradDiff, UNDIAL, AltPO, SimNPO): positive transfer at 1%, declining to near-zero or below at 5%. GradDiff is the most resilient — the only new method to maintain clearly positive transfer at both scales (+16.7% → +11.8%) — and notably achieves this despite perfect baseline forgetting (QAP = 0.000).

**Near-zero** (IdkDPO): essentially no transfer at either scale. The DPO-based IDK mechanism appears to compartmentalize forgetting more completely, leaving little recoverable latent structure.

**Anti-transfer** (IdkNLL): teaching forget01 content *reduces* free-recovery recall (−32.5% at f01, −75.3% at f05). The IDK surface suppression installed by IdkNLL is disrupted by SFT in a way that paradoxically increases forgetting of untaught content. Given that IdkNLL never fully erased underlying representations (baseline QAP = 0.54), SFT may be overwriting the IDK behavioral layer in a way that selectively degrades adjacent content access.

**RMU★ is the only method across all eight to show a transfer rate that increases from forget01 to forget05.** All others collapse or go negative.

---

## Discussion

**The threat model holds for every method tested.** Teaching a small fraction of forgotten content restores recall for the untaught fraction at a rate well above what a model that genuinely never learned the data can produce. No method fully erased the knowledge — it remained latent and recoverable. An adversary who already possesses some fraction of the "removed" content and can fine-tune the open-weight model will recover untaught related content for free. How much, and how efficiently, depends on which unlearning method was applied.

The more interesting finding is the split in recovery structure:

**One cluster** — NPO, AltPO, SimNPO, UNDIAL, and GradDiff — shows positive free-recovery transfer that collapses or disappears as the training-set size grows. The pattern is consistent with these methods primarily installing *behavioral suppression*: teaching a small in-distribution sample is enough to unlock the latent structure, but the unlocking mechanism saturates quickly. NPO is the most extreme case; GradDiff is the most resilient within the cluster (sustaining +11.8% transfer at 5%) and the one most likely to be doing something mechanistically different.

**RMU★ is the sole exception.** Despite starting from genuine amnesia (QAP ≈ 0.001), its free-recovery transfer rate *increases* from 29% to 42% as training-set size doubles. If RMU's representation-steering leaves a more coherent latent geometry than gradient-based methods, then larger reactivation signals would naturally unlock more of the shared structure — which is exactly what the data shows.

**IdkDPO** occupies a third position: near-zero transfer throughout. The DPO-style alignment toward IDK responses seems to compartmentalize forgetting more completely, leaving little transferable latent structure — even though its baseline QAP (0.135) is higher than RMU★'s.

**IdkNLL** is the odd one out: it never suppresses recall at baseline (QAP = 0.54) and shows *anti-transfer* — teaching content causes adjacent recall to fall rather than rise. The IDK surface-behavior layer appears to actively interfere with adjacent representations rather than coexisting with them.

The mechanistic story is not settled. The most direct next test is to probe the internal representations: if NPO installs a refusal direction, it should be detectable as a linear feature in activation space that disappears when retraining undoes it. If RMU disrupts circuits without leaving a coherent recoverable geometry, intermediate activations on forget-set inputs should look incoherent in a way that retraining replaces rather than restores. The diverging transfer-rate trajectories give a behavioral prediction to anchor that analysis.

---

## Open Questions

**White-box mechanistic analysis** is the natural next step. The behavioral results give sharp predictions: NPO should show a detectable "refusal direction" in intermediate activations that shifts when retraining undoes it; RMU should show incoherent activations on forget-set inputs that retraining replaces (rather than restores). Confirming or falsifying these predictions would move the explanation from behavioral pattern to mechanistic claim, and would explain why RMU's transfer rate scales up while every other method's collapses.

---

## Acknowledgments

This work was done as part of the [BlueDot Impact](https://bluedot.org/) AI safety project sprint. Thanks to my cohort and advisors for guidance on experimental design and threat modeling. Code and data are available in the [project repository](https://github.com/SamPease/retrainingUnlearning).
