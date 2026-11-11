---
title: "Olmo 3 Neuron Explorer"
description: "An interactive companion to the LessWrong post on single neurons in Olmo 3: training documents and steered replies, with shared strings highlighted."
date: 2026-11-11
tags: ["interpretability", "LLMs", "data attribution"]
draft: false
---

# Olmo 3 Neuron Explorer

This is a browsable companion to [the LessWrong post](#) (link to come), which only had room for static figures.

It covers two kinds of single neurons. **Found from data** neurons were found in the base model, Olmo-3-1025-7B, from their training documents and then pushed (steered) in Olmo-3-7B-Instruct. **Found by pushing** neurons were found by steering Olmo-3-7B-Instruct and then traced back to training documents. Each document is labeled with its training stage: pretraining, midtraining or post-training.

For each neuron, the training documents sit on the left and the steered replies on the right. Strings the two share are highlighted, and clicking one lights it up everywhere and steps through each place it appears. Some documents are odd or unpleasant web text, shown as found.

**[Open the explorer full screen →](/neuron-explorer/index.html)**

<iframe src="/neuron-explorer/index.html" width="100%" style="height: 85vh; min-height: 560px; border: 1px solid rgba(44, 58, 86, 0.18); border-radius: 10px;" title="Olmo 3 neuron explorer"></iframe>
