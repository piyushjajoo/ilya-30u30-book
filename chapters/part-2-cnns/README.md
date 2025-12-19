# Part II: Convolutional Neural Networks

> *The revolution in visual understanding*

---

## Overview

Part II covers the deep learning revolution in computer vision. Starting with the landmark AlexNet paper (co-authored by Ilya Sutskever himself), we trace the evolution of CNN architectures through ResNet and beyond.

## Chapters

| # | Chapter | Key Concept |
|---|---------|-------------|
| 6 | [AlexNet - The ImageNet Breakthrough](./06-alexnet.md) | Deep learning works at scale |
| 7 | [CS231n - CNNs for Visual Recognition](./07-cs231n.md) | Comprehensive CNN foundations |
| 8 | [Deep Residual Learning (ResNet)](./08-resnet.md) | Skip connections enable depth |
| 9 | [Identity Mappings in ResNets](./09-identity-mappings.md) | Optimal residual unit design |
| 10 | [Dilated Convolutions](./10-dilated-convolutions.md) | Multi-scale context aggregation |

## The Evolution

```
AlexNet (2012)     →  8 layers, ReLU, Dropout
       ↓
VGG (2014)         →  Deeper (19 layers), smaller filters
       ↓
ResNet (2015)      →  152 layers via skip connections
       ↓
Modern CNNs        →  Efficient architectures, attention
```

## Key Takeaway

> **Depth matters, but only with the right architectural innovations. Skip connections, proper normalization, and careful design allow networks to learn hierarchical visual features automatically.**

## Prerequisites

- Part I foundations (helpful but not required)
- Basic understanding of linear algebra (matrix operations)
- Familiarity with gradient descent

## What You'll Be Able To Do After Part II

- Understand how CNNs learn visual features
- Implement and train image classifiers
- Explain why deeper networks work (with ResNet)
- Choose appropriate CNN architectures for tasks

