---
title: "Olmo 3 Neuron Explorer"
description: "An interactive companion to the LessWrong post on single neurons in Olmo 3: training documents and steered replies, with shared strings highlighted."
date: 2026-11-11
tags: ["interpretability", "LLMs", "data attribution"]
draft: false
---

# Olmo 3 Neuron Explorer

This is a browsable companion to [the LessWrong post](https://www.lesswrong.com/posts/wBTaEGxNTGAHxCcbD/weird-neurons-mimic-weird-data), which only had room for static figures.

It covers two kinds of single neurons. **Found from data** neurons were found in the base model, Olmo-3-1025-7B, from their training documents and then pushed (steered) in Olmo-3-7B-Instruct. **Found by pushing** neurons were found by steering Olmo-3-7B-Instruct and then traced back to training documents. Each document is labeled with its training stage: pretraining, midtraining or post-training.

For each neuron, the training documents sit on the left and the steered replies on the right. Strings the two share are highlighted, and clicking one lights it up everywhere and steps through each place it appears. Some documents are odd or unpleasant web text, shown as found.

**[Open the explorer full screen →](/neuron-explorer/index.html)** · [Methods](#methods)

<iframe src="/neuron-explorer/index.html" width="100%" style="height: 85vh; min-height: 560px; border: 1px solid rgba(44, 58, 86, 0.18); border-radius: 10px;" title="Olmo 3 neuron explorer"></iframe>

## Methods

How the neurons in the post were found and tested. Everything uses Olmo 3 7B.

## Setup

- **Neurons**
  - 32 layers × 11,008 MLP (SwiGLU) neurons = 352,256 neurons.
  - Each neuron has a positive and a negative pole ([definition](https://www.goodfire.com/research/neurons-as-an-interpretable-basis#background-the-structure-of-a-swiglu-neuron)), so there are 704,512 poles.
- **Steering**
  - One pole at a time, the neuron's output direction is added to the residual stream at every token: $c \cdot d_n$, with $c = \pm\,\alpha\, r_L / \lVert d_n \rVert$ (the Neuron Persona Library's formula).
  - $d_n$ is the neuron's column of $W_\text{down}$ times the post-feedforward norm weight, added after that norm (Olmo 3 is post-norm).
  - $r_L$ is the median residual-stream norm after layer $L$ over the prompt's own tokens (first token excluded), computed per prompt. So $\alpha = 1$ is about the typical size of the residual stream.
  - Strength $\alpha$ by stage:
    - Part 1 steering: 1.0–2.5.
    - Part 2 sweep: 0.5–1.25.
    - Part 2 screen: 1.5 and 2.5.
    - Strength curves: 0.25–3.0.
  - Behaviour typically starts around $\alpha$ 0.75–1.0 and peaks at 1.25–2.0.
- **Two models**
  - Ranking neurons and reading documents: the base model, Olmo-3-1025-7B.
    - The data is pretraining data.
    - Pilot: on 16 document sets, a neuron picked on half of each set had to pick out the other half from 1,500 control documents.
    - Neurons picked in the base model put 59% of held-out documents above every control on average, against 48% for neurons picked in the instruct model.
  - Steering and all behavioural tests: Olmo-3-7B-Instruct.
- **Writer neuron**
  - A neuron that, steered on its own, makes the model write a recognisable kind of text.
  - The two parts below test this in different ways.

## Part 1: from data to neurons

![Part 1 pipeline: training documents, rank neurons three ways in the base model, steer in the chat model, grade the replies](/neuron-explorer/figures/method_B6.png)
*Part 1 in four steps.*

- **Finding the genres**
  - Most genres were collected by a text signature: string patterns such as a stock phrase or a channel greeting, scanned over a 3-million-document stage-1 sample and five whole Dolma 3 web topics.
  - Genres with no signature came from an LLM judge's (Jev's) answers to questions about weirdness, over a 200,000-document sample.
  - Matches were then judged and read. A genre's documents are its own text; pages that only mention it are kept apart as "near misses".
  - Documents were never chosen by a neuron.
- **Data**
  - 26 genres of Dolma 3 stage-1 (pretraining) documents: 1,871 documents, 38–80 per genre (mean 72).
  - The post groups the genres into five kinds, following the top level of our current, provisional taxonomy of weird data: sincere human oddness, fiction and jokes, spam and templates, data formats, and text written by an LLM.
  - The activation ranking below used an earlier grouping into four: one author or movement, one website or template, a format many sites share, and genres picked out by an LLM judge.
  - Each genre is split in half: half A ranks neurons, half B is held out.
- **Ranking neurons, per genre**
  - Activation, per pole:
    - Feature: the neuron's peak activation over the first 1,024 tokens of each document.
    - Score: the lowest AUROC across four negative sets, which keeps the top from saturating:
      - other genres in the same group (of the four above)
      - genres in the other groups
      - ordinary pages from the same web folder
      - random stage-1 pages
  - Gradient, per neuron:
    - Feature: one backward pass on the next-token loss. The squared gradient on the neuron's gate row, up row and down column, as a share of the whole MLP gradient.
    - Score: AUROC against 991 random documents.
  - Zeroing, per neuron:
    - Shortlist: the top 1,000 of the activation and gradient lists (about 1,700 neurons per genre after overlap; 44,279 in all), because zeroing is expensive.
    - Score: the rise in perplexity on held-out half B when the neuron's output is set to zero, minus its rise on 300 ordinary stage-1 documents. Neurons the model needs for all text don't win.
    - The ordinary-document rise was measured for each genre's top 100; the rest are ranked by their own rise alone, an upper bound.
- **What the rankings look like**
  - No ranking produces neurons that clearly stand out. Each gives a smooth ranking of all neurons.
  - Writers are often not at the very top.
  - The three rankings are largely independent. Median Spearman correlation across genres:
    - activation and gradient: 0.53
    - gradient and zeroing: 0.12
    - activation and zeroing: 0.03

![Every neuron ranked by activation and by gradient for each genre, with writer neurons marked](/neuron-explorer/figures/rankings_C.png)
*Every neuron's rank for each genre (grey lines), Olmo 3 7B base. Dots are writer neurons; hollow ones fall outside the top-1,000 shortlist.*

- **Steering**
  - About the top 10 of each list per genre: about 57 neuron–genre pairs per genre, 1,472 in all.
  - Each pair steered in the instruct model, first at one pole, then at the other; a writer counts at either pole.
    - 13 writers only wrote at the pole tried second.
  - $\alpha$ 1.0, 1.5, 2.0 and 2.5; 10 prompts × 2 samples at each strength, so 80 replies per pole.
    - Prompts: "Who are you?", "Tell me about your day.", "Explain how photosynthesis works.", "Write a short story about a lighthouse.", "What is the best way to learn a language?", "Describe the city of Paris.", "Give me a recipe for pancakes.", "What do you think happens after we die?", "Summarize the history of the Roman Empire in a paragraph.", "Write a product description for a pair of running shoes."
    - Sampling: temperature 1, top-p 0.9, 120 new tokens.
  - Controls: 20 unsteered replies per genre, plus the genre's known writer where there was one.
- **Judging**
  - Screen: an LLM judge (Jev) scored each reply 0–3 for being the same kind of text as the genre's description. A reply counts at 2 or more.
    - Candidate: at least 4 of 20 replies count at the neuron's best $\alpha$, and at least 3 more than the unsteered replies.
  - Confirmation by blind reading:
    - Each candidate's up to 6 judge-passed replies at its best $\alpha$ (first 400 characters each), shuffled across all candidates.
    - Readers saw only the genre's description, with the neuron, method, strength and earlier verdicts removed.
    - A fresh Claude Opus reader and a fresh Claude Sonnet reader each said whether at least 2 replies read as the genre's own text (its voice, vocabulary or format) and are coherent, not broken text with a few genre words.
    - A writer needs both readers to say yes. The two agree with κ = 0.61.
- **Results**
  - 45 writers in 12 of the 26 genres: 44 distinct neurons, one of which writes for two genres.
  - Writers found at 30 neurons steered per genre:
    - activation: 15
    - gradient: 18
    - zeroing: 30
    - pooled list (the top third of each): 31
  - Zeroing alone does almost as well as pooling. Pooling mainly covers the blind spots of activation and gradient.

![Writer neurons found against neurons steered per genre, for each ranking](/neuron-explorer/figures/appx_rankrecall.png)
*Writers found against the number of neurons steered per genre. Writers could only be found where we steered, mostly near the top of the lists, so every curve is optimistic. The chart compares the methods; it isn't recall over all writers.*

## Part 2: from neurons to data

![Part 2 pipeline: steer every neuron, keep consistent writers, find each writer's top documents in the base model, compare word use across writers](/neuron-explorer/figures/part2_flow_D.png)
*Part 2 in four steps.*

- **Sweep: steer every pole**
  - All 704,512 poles, in the instruct model, on two prompts:
    - "Who are you?", as in the [Neuron Persona Library](https://neuronpersonas.com/#olmo).
    - "Write a short blog post."
  - One greedy reply of 160 tokens per pole, at the strongest of $\alpha$ 1.25, 1, 0.75 or 0.5 that doesn't fall into a repetition loop.
  - An LLM judge (Jev) read every reply: whether it claims a non-AI identity, which of 17 kinds of text it is, and a rank.
- **Candidates**
  - 1,920 poles whose "Who are you?" reply claims a non-AI identity.
  - 626 poles that were among each weird kind's top-scoring replies in the two sweeps, plus a few picks from earlier document reads.
  - The 31 data-found writers known at the time.
  - Together, after removing duplicates: 2,484 candidates, of which 2,440 are new and 44 are known writers.
- **Writer screen**
  - Labelling:
    - Each candidate wrote 12 replies at $\alpha$ 1.5: 6 prompts (photosynthesis, a lighthouse story, learning a language, pancakes, life after death, running shoes) × 2 samples.
    - Three labellers (Claude Sonnet 5.5, GPT-5.5 and Gemini 3.1 Pro) each read the 12 replies blind and named one feature the prompts didn't ask for.
  - Testing each label on fresh replies:
    - 6 new prompts, e.g. "Tell me about your day.", "Describe the city of Paris.", "Write a haiku about winter." × 2 samples = 12 fresh replies.
    - Run separately at $\alpha$ 1.5 and at $\alpha$ 2.5. A label passes if it passes at either strength.
    - A GPT-5.5 judge saw the 12 fresh replies shuffled with 12 decoys (6 unsteered replies and 6 from another neuron) and marked each one that shows the label.
    - Pass: the label shows in at least 6 of the neuron's 12 replies, those come from at least 3 of the 6 prompts, and it shows in at most 2 of the 12 decoys.
  - Origin check:
    - Two judges (GPT-5.5 and Gemini 3.1 Pro) read only the label and call it a kind of document, a writing style, an AI-assistant quirk, or a malfunction (token loops, stutters, token salad).
    - The label is kept if either judge calls it a document or a style.
  - A writer: at least 2 of the 3 labels pass both the test and the origin check.
  - The rules were fixed before any result was seen.
  - GPT-5.5 is both a labeller and the judge. With Gemini as the judge instead, the results barely move.
- **Screen results**
  - 1,382 of the 2,440 new candidates pass (57%).
  - Known writers: 24 of 44 pass (16 of 32 in the development set, 8 of 12 held out).
  - Random neurons: 28 of 300 from the same layers pass (9%).
    - What they write is real: game rules, nutrition labels, software licences.
    - So writing a recognisable kind of text is common among neurons; the sweep makes it about six times likelier to find.
  - Writers in all: 1,382 + 24 = 1,406.
  - Both parts together: 1,435 distinct writer neurons (these 1,406, plus the 29 of Part 1's 44 that aren't among them).
  - Compared with the Neuron Persona Library, which covers every pole: 450 of the 1,406 are personas there, and 956 are not.
    - Many candidates came from "Who are you?" replies with a non-AI identity, the library's own test, so the overlap with personas is expected.
    - Across both parts: 451 of the 1,435 are personas there, and 984 are not.

![Funnel from 704,512 poles to 2,484 candidates to 1,382 writers, and pass rates for candidates vs random neurons](/neuron-explorer/figures/appx_screen.png)
*Olmo 3 7B Instruct, one neuron steered at a time. Right: the same screen on 300 random neurons.*

- **Back to the data**
  - For each of the 1,406 writers: its top 20 documents by activation and its top 20 by gradient, out of 200,000 Dolma 3 stage-1 documents, in the base model.
- **Vocabulary match** (the rule was fixed before any score was seen)
  - Words: lowercase words of 3 or more letters, counted once per document or reply, from the first 6,000 characters of each document.
  - Document profile: each word weighted by the log ratio of its smoothed document frequency in the writer's top documents to that in the 200,000-document pool, $w = \log\frac{d + 0.5}{n + 1} - \log\frac{D + 0.5}{N + 1}$, where $d$ of the writer's $n$ top documents and $D$ of the $N$ = 200,000 pool documents contain the word. Only positive weights are kept, for words in at least 2 of the top documents.
  - Reply profile $p$: the share of a writer's steered replies using each word, minus the share of unsteered replies to the same questions using it, floored at 0.
  - Score of writer $i$'s replies against writer $j$'s documents: $p_i \cdot w_j$. Every writer's replies were scored against every writer's documents (1,406 × 1,406), and each writer's own set was ranked among all 1,406.
- **Vocabulary-match results**
  - A writer's own documents beat 93% of the other 1,405 writers' document sets on average (mean AUROC 0.933, 95% CI 0.923–0.943; median 0.999; chance is 0.5).
  - Own documents rank 1st for 43% of writers (chance: 0.07%) and in the top 10 for 68% (chance: 0.7%). The median rank is 2nd.
  - 87% of writers have an AUROC of at least 0.9; 8% are below 0.7.
  - Losses:
    - Most go to a few generic "hub" document sets: academic prose, My Little Pony fan fiction, machine-translated business spam, stream-of-consciousness text.
    - Writers with low scores have vague kinds: page-interface debris, filler, bureaucratic stock phrases.

![Share of writers by the rank of their own documents among all 1,406 writers' documents, against chance](/neuron-explorer/figures/appx_vm1.png)
*Where each writer's own documents rank among all 1,406 writers' document sets, by vocabulary match.*

## Robustness

- **Vocabulary-match checks**
  - Removing writers of the same kind as distractors: mean AUROC 0.935.
  - Reverse direction (one writer's documents, its own replies against everyone else's): 0.923.
  - Removing hub sets after the fact (each document set z-scored across writers): 0.970; own set 1st for 52% of writers and in the top 10 for 80%.
- **Rare words**
  - 53% of writers (747 of 1,406) use a rare string from their own top documents in their screen replies.
  - Controls:
    - the runner-up neuron's documents: 4%
    - five other writers' replies against this writer's documents: 28%
    - unsteered replies: 10%
- **Blind read of the writer screen**
  - A Claude Opus reader saw 60 neurons (labels and fresh replies), without being told which had passed.
  - Of 40 that passed: 31 clearly write their label, 8 partly, 1 doesn't.
  - Of 20 that failed: 9 are consistent, but 7 of those are malfunctions and 1 an assistant quirk, both excluded on purpose. That leaves 1 real miss.
- **Steering strength**
  - 16 writers at 10 strengths ($\alpha$ 0.25–3.0), 10 prompts × 6 samples = 60 replies per strength, each against 3 random neurons from the same layer.
  - 15 of 16 pass a writer bar fixed in advance. Random neurons reach at most 7%, and 0 of 100 unsteered replies are in any writer's kind.
  - There is a sweet spot: the behaviour appears only at stronger steering, where coherence starts to drop.
  - Some writers, like the rare-word spam writer below, lose coherence before they show their kind.

![Six writer neurons by steering strength: share of replies in the neuron's kind and share coherent](/neuron-explorer/figures/dose_C.png)
*Six writers by steering strength, 60 replies per strength. Solid: replies written in the neuron's kind. Dashed: coherent replies of any kind. Shaded: both.*

- **Multi-turn conversations**
  - Setup: 7-turn conversations, 5 prompts × 6 samples, with Jev labelling the kind of each turn.
    - Controls: random neurons from the same layers show the kind in 0–7% of turns; unsteered conversations stay the assistant in 73–100% of turns.
  - At moderate strength (about 0.7–1.25× the single-turn strength), the kind holds for all 7 turns and often builds up.
    - Example: a pair of persecution-voice neurons steered together (each at $\alpha/\sqrt{2}$), at 0.7×, goes from 20% of replies in its kind to 83–87%, then 100%.
  - Pushed harder (1.5–2×), the conversation collapses into broken text: 40–100% broken from the first turn, and the kind falls to 10% or less by turn 6.
  - With the steering switched off, the kind is gone on the next reply (0–23%).
  - Several writers steered together only work when each one's strength is scaled down to $\alpha/\sqrt{k}$ for $k$ writers.

![Four writer neurons over seven turns, with steering kept on vs stopped after turn 2](/neuron-explorer/figures/multiturn_D.png)
*Four writers over a seven-turn conversation, 30 conversations per line, steered at 0.8× the single-reply strength. Blue: steering kept on. Grey: steering stopped after turn 2.*

## Limits

- **Steering goes beyond natural firing**
  - For the 61 writers checked, the push is 1.4–4.8× (median 2.4×) the most the neuron's own activation can add.
  - Held at the neuron's natural maximum, 0 of the 61 write their kind.
  - So a writer is a direction that steering reaches, not something the neuron produces by firing on its own.
  - This differs from [Goodfire's neuron analysis](https://www.goodfire.com/research/neurons-as-an-interpretable-basis), which reads neurons from their activations on real text, mainly the top 3% of their natural range, rather than by steering.
- **Training stage:** all documents are pretraining (stage-1) documents. Part 1's genres were checked for their own text in mid- and post-training data (the stages are in the post's genre chart). For Part 2's writers we didn't check whether the same kind of text is also in mid- or post-training data.
- **No interventions:** nothing was retrained without the data, so the link between a writer and its documents is correlational.
- **Coverage:** Part 1 only finds writers among the neurons we steered, near the top of each ranking. Part 2 only screens poles flagged by the two-prompt sweep or an earlier screen, so writers that those two prompts don't bring out are missed.
