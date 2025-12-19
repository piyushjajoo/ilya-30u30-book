# Part IV: Attention and Transformers

> *The attention revolution*

---

## Overview

Part IV covers the most transformative development in deep learning since CNNs: **attention mechanisms** and **Transformers**. These architectures fundamentally changed how we process sequences, enabling parallelization, better long-range dependencies, and the foundation for modern LLMs.

## Chapters

| # | Chapter | Key Concept |
|---|---------|-------------|
| 15 | [Neural Machine Translation with Attention](./15-nmt-attention.md) | Attention solves the bottleneck |
| 16 | [Attention Is All You Need (Transformers)](./16-transformers.md) | Eliminating recurrence entirely |
| 17 | [The Annotated Transformer](./17-annotated-transformer.md) | Implementation walkthrough |

## The Evolution

```
Seq2Seq (2014)    →  Encoder-decoder with bottleneck
       ↓
Attention (2015)  →  Soft alignment, no bottleneck
       ↓
Transformers (2017) →  Attention is all you need
       ↓
Modern LLMs       →  GPT, BERT, and beyond
```

## Key Takeaway

> **Attention mechanisms allow models to dynamically focus on relevant parts of the input, eliminating the need for fixed-length bottlenecks and enabling parallel processing of sequences.**

## Prerequisites

- Part III (RNNs) - helpful for understanding the problems attention solves
- Understanding of sequence-to-sequence models
- Familiarity with matrix operations

## What You'll Be Able To Do After Part IV

- Understand how attention mechanisms work
- Implement Transformer architectures
- See how modern LLMs are built
- Appreciate why Transformers replaced RNNs for most tasks

