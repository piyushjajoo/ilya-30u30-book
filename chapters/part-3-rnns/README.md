---
layout: default
title: Part III - Sequence Models and Recurrent Networks
nav_order: 5
has_children: true
---

# Part III: Sequence Models and Recurrent Networks

> *Learning from sequential data*

---

## Overview

Part III explores how neural networks handle **sequential data**—text, speech, time series, and more. While CNNs excel at spatial patterns, recurrent networks excel at temporal patterns and dependencies across time.

## Chapters

| # | Chapter | Key Concept |
|---|---------|-------------|
| 11 | [The Unreasonable Effectiveness of RNNs](./11-rnn-effectiveness.md) | RNNs can generate text, code, and more |
| 12 | [Understanding LSTM Networks](./12-lstm-networks.md) | Gated memory solves vanishing gradients |
| 13 | [RNN Regularization](./13-rnn-regularization.md) | Dropout for recurrent connections |
| 14 | [Relational Recurrent Neural Networks](./14-relational-rnns.md) | Self-attention in recurrence |

## The Evolution

```
Vanilla RNNs      →  Simple but limited memory
       ↓
LSTMs/GRUs        →  Gated memory, long-range dependencies
       ↓
Regularized RNNs  →  Better generalization
       ↓
Attention + RNNs  →  Toward Transformers
```

## Key Takeaway

> **Recurrent networks process sequences by maintaining hidden state—a form of memory that evolves over time. The challenge is making this memory effective over long sequences.**

## Prerequisites

- Part II foundations (helpful for understanding architecture)
- Basic understanding of backpropagation
- Familiarity with language modeling concepts

## What You'll Be Able To Do After Part III

- Understand how RNNs process sequences
- Implement and train LSTM networks
- Apply proper regularization to RNNs
- See the path toward attention mechanisms (Part IV)

