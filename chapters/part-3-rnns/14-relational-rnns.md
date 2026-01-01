---
layout: default
title: Chapter 14 - Relational Recurrent Neural Networks
nav_order: 16
---

# Chapter 14: Relational Recurrent Neural Networks

> *"We introduce a memory module that uses self-attention to allow memories to interact."*

**Based on:** "Relational Recurrent Neural Networks" (Adam Santoro, Ryan Faulkner, David Raposo, et al., 2018)

📄 **Original Paper:** [arXiv:1806.01822](https://arxiv.org/abs/1806.01822) | [NeurIPS 2018](https://papers.nips.cc/paper/2018/hash/26337353b7962f533d78c762373b3318-Abstract.html)

---

## 14.1 Bridging RNNs and Attention

By 2018, attention mechanisms (Chapter 15) and Transformers (Chapter 16) were showing remarkable success. But could we combine the best of both worlds?

This paper introduces **Relational Recurrent Neural Networks (RRNNs)**: RNNs enhanced with self-attention mechanisms in their memory.

```mermaid
graph TB
    subgraph "The Combination"
        R["RNN<br/>(Sequential processing)"]
        A["Self-Attention<br/>(Relational reasoning)"]
        RRNN["Relational RNN<br/>(Best of both)"]
    end
    
    R --> RRNN
    A --> RRNN
    
    B["Enables:<br/>• Sequential processing<br/>• Relational reasoning<br/>• Long-range dependencies"]
    
    RRNN --> B
    
    style RRNN fill:#ffe66d,color:#000
```

---

## 14.2 The Motivation: Relational Reasoning

### What Is Relational Reasoning?

Understanding relationships between entities:

```mermaid
graph TB
    subgraph "Relational Reasoning Tasks"
        Q1["'The red ball is to the<br/>left of the blue cube'"]
        Q2["'Who is older: Alice or Bob?'"]
        Q3["'If A > B and B > C,<br/>then A > C'"]
    end
    
    R["Requires comparing<br/>and relating entities"]
    
    Q1 --> R
    Q2 --> R
    Q3 --> R
    
    style R fill:#ffe66d,color:#000
```

### Why Standard RNNs Struggle

Standard RNNs process sequentially—they can't easily compare distant elements:

```mermaid
graph LR
    subgraph "Standard RNN"
        X1["x₁"] --> H1["h₁"]
        X2["x₂"] --> H2["h₂"]
        X3["x₃"] --> H3["h₃"]
        X4["x₄"] --> H4["h₄"]
    end
    
    P["To compare x₁ and x₄,<br/>information must flow<br/>through h₁→h₂→h₃→h₄<br/>(gradients vanish!)"]
    
    H1 --> P
    
    style P fill:#ff6b6b,color:#fff
```

---

## 14.3 The Relational Memory Core

### Architecture Overview

The key innovation: a **Relational Memory Core (RMC)** that uses self-attention:

```mermaid
graph TB
    subgraph "Relational Memory Core"
        X["Input x_t"]
        M_PREV["Memory M_{t-1}<br/>(N memory slots)"]
        
        ATT["Self-Attention<br/>over memory slots"]
        UPDATE["Memory Update"]
        
        M_NEW["Memory M_t"]
        H["Hidden State h_t"]
    end
    
    X --> ATT
    M_PREV --> ATT
    ATT --> UPDATE
    X --> UPDATE
    UPDATE --> M_NEW
    UPDATE --> H
    
    K["Memory slots can attend<br/>to each other = relational reasoning!"]
    
    ATT --> K
    
    style K fill:#4ecdc4,color:#fff
```

### Memory as a Set of Slots

Instead of a single hidden state, maintain **N memory slots**:

$$M_t = [m_t^1, m_t^2, ..., m_t^N]$$

Each slot can represent different aspects or entities.

---

## 14.4 Self-Attention in Memory

### How Memory Slots Interact

```mermaid
graph TB
    subgraph "Self-Attention Mechanism"
        M["Memory M = [m¹, m², m³]"]
        
        Q["Queries Q = M·W_Q"]
        K["Keys K = M·W_K"]
        V["Values V = M·W_V"]
        
        ATT["Attention(Q, K, V)"]
        
        M_OUT["Updated Memory"]
    end
    
    M --> Q
    M --> K
    M --> V
    
    Q --> ATT
    K --> ATT
    V --> ATT
    
    ATT --> M_OUT
    
    E["Each memory slot can<br/>attend to all others!"]
    
    ATT --> E
    
    style E fill:#ffe66d,color:#000
```

### The Attention Operation

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

This allows each memory slot to:
- **Query** other slots
- **Compare** with other slots
- **Aggregate** information from relevant slots

---

## 14.5 The Complete RRNN Architecture

### Forward Pass

```mermaid
graph TB
    subgraph "RRNN Step"
        X["x_t (input)"]
        M_PREV["M_{t-1} (previous memory)"]
        
        subgraph "Relational Memory Core"
            CONCAT["Concat [x_t, M_{t-1}]"]
            ATT["Multi-Head Self-Attention"]
            FF["Feedforward"]
            NORM1["Layer Norm"]
            NORM2["Layer Norm"]
        end
        
        M_NEW["M_t (new memory)"]
        H["h_t (hidden state)"]
    end
    
    X --> CONCAT
    M_PREV --> CONCAT
    CONCAT --> ATT --> NORM1 --> FF --> NORM2 --> M_NEW
    M_NEW --> H
    
    K["Transformer-like structure<br/>within RNN framework"]
    
    ATT --> K
    
    style K fill:#ffe66d,color:#000
```

### Mathematical Formulation

1. **Concatenate input with memory**:
   $$X_t = [x_t; M_{t-1}]$$

2. **Apply self-attention**:
   $$M'_t = \text{MultiHeadAttention}(X_t, X_t, X_t)$$

3. **Feedforward and normalize**:
   $$M_t = \text{LayerNorm}(M'_t + \text{FF}(M'_t))$$

4. **Extract hidden state**:
   $$h_t = \text{ReadHead}(M_t)$$

---

## 14.6 Multi-Head Attention

### Why Multiple Heads?

```mermaid
graph TB
    subgraph "Single Head"
        M["Memory"]
        ATT1["One attention pattern"]
        OUT1["One relationship type"]
    end
    
    subgraph "Multi-Head"
        M2["Memory"]
        ATT2["Head 1: Spatial relations"]
        ATT3["Head 2: Temporal relations"]
        ATT4["Head 3: Semantic relations"]
        OUT2["Multiple relationship types"]
    end
    
    M --> ATT1 --> OUT1
    M2 --> ATT2 --> OUT2
    M2 --> ATT3 --> OUT2
    M2 --> ATT4 --> OUT2
    
    K["Different heads capture<br/>different types of relationships"]
    
    ATT2 --> K
    
    style K fill:#ffe66d,color:#000
```

### Multi-Head Formulation

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

Where each head:
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

---

## 14.7 Applications and Results

### bAbI Tasks

The paper evaluates on bAbI—a suite of reasoning tasks:

```mermaid
graph TB
    subgraph "bAbI Tasks"
        T1["Question Answering<br/>'Where is the apple?'"]
        T2["Counting<br/>'How many objects?'"]
        T3["Lists/Sets<br/>'What is first?'"]
        T4["Positional Reasoning<br/>'Who is left of X?'"]
    end
    
    R["RRNN outperforms<br/>standard RNNs and<br/>even some attention models"]
    
    T1 --> R
    T2 --> R
    T3 --> R
    T4 --> R
    
    style R fill:#4ecdc4,color:#fff
```

### Results Summary

| Model | bAbI Accuracy |
|-------|---------------|
| LSTM | ~60% |
| Attention-based | ~75% |
| **RRNN** | **~85%** |

### Language Modeling

RRNN also improves language modeling:
- Better perplexity than LSTM
- Captures long-range dependencies
- Learns relational patterns in text

---

## 14.8 Comparison with Other Architectures

### RRNN vs Standard RNN

```mermaid
graph TB
    subgraph "Standard RNN"
        S1["Single hidden state"]
        S2["Sequential processing"]
        S3["Limited relational reasoning"]
    end
    
    subgraph "RRNN"
        R1["Multiple memory slots"]
        R2["Sequential + relational"]
        R3["Explicit relationship modeling"]
    end
    
    S1 --> C["RRNN adds relational<br/>reasoning capability"]
    R1 --> C
    
    style C fill:#ffe66d,color:#000
```

### RRNN vs Transformer

| Aspect | RRNN | Transformer |
|--------|------|-------------|
| Processing | Sequential | Parallel |
| Memory | Recurrent slots | All positions |
| Relational reasoning | ✅ Yes | ✅ Yes |
| Long sequences | Good | Excellent |
| Training speed | Slower | Faster |

---

## 14.9 The Read Head

### Extracting Hidden State

The read head extracts information from memory:

```mermaid
graph TB
    subgraph "Read Head"
        M["Memory M_t<br/>[m¹, m², ..., m^N]"]
        W["Learnable weights<br/>or attention"]
        H["h_t"]
    end
    
    M --> W --> H
    
    O["Options:<br/>• Weighted sum<br/>• Attention-based<br/>• Concatenation"]
    
    W --> O
    
    style O fill:#ffe66d,color:#000
```

### Simple Read Head

$$h_t = \frac{1}{N}\sum_{i=1}^{N} m_t^i$$

Or with learned attention:
$$h_t = \sum_{i=1}^{N} \alpha_i m_t^i$$

---

## 14.10 Training RRNNs

### Challenges

```mermaid
graph TB
    subgraph "Training Challenges"
        C1["More parameters<br/>(attention matrices)"]
        C2["Slower than standard RNN<br/>(O(N²) attention)"]
        C3["Memory initialization<br/>(how to start?)"]
    end
    
    S["Solutions:<br/>• Careful initialization<br/>• Gradient clipping<br/>• Learning rate scheduling"]
    
    C1 --> S
    C2 --> S
    C3 --> S
```

### Initialization

Memory slots typically initialized to small random values or zeros. The network learns to use them effectively.

---

## 14.11 Connection to Neural Turing Machines

### Similar Concepts

RRNNs share ideas with Neural Turing Machines (Chapter 20):

```mermaid
graph TB
    subgraph "Shared Ideas"
        M["External memory<br/>(multiple slots)"]
        A["Attention-based<br/>memory access"]
        R["Relational operations"]
    end
    
    NTM["Neural Turing Machine<br/>(Chapter 20)"]
    RRNN["Relational RNN<br/>(This chapter)"]
    
    M --> NTM
    M --> RRNN
    A --> NTM
    A --> RRNN
    R --> RRNN
    
    D["RRNN: Simpler, more focused<br/>on relational reasoning"]
    
    RRNN --> D
```

---

## 14.12 Modern Perspective

### Legacy and Impact

```mermaid
timeline
    title RRNN in Context
    2017 : Attention Is All You Need
         : Transformers introduced
    2018 : Relational RNNs
         : Combining RNN + Attention
    2019 : Transformer dominance
         : Most tasks use Transformers
    2020s : RNNs still used
          : For specific sequential tasks
          : RRNN ideas in some models
```

### Where RRNNs Fit Today

- **Research**: Interesting hybrid approach
- **Production**: Less common than pure Transformers
- **Insight**: Shows how to add relational reasoning to RNNs
- **Bridge**: Connects RNN and Transformer ideas

---

## 14.13 Connection to Other Chapters

```mermaid
graph TB
    CH14["Chapter 14<br/>Relational RNNs"]
    
    CH14 --> CH12["Chapter 12: LSTMs<br/><i>Standard RNN baseline</i>"]
    CH14 --> CH15["Chapter 15: Attention<br/><i>Attention mechanism used here</i>"]
    CH14 --> CH16["Chapter 16: Transformers<br/><i>Similar self-attention</i>"]
    CH14 --> CH20["Chapter 20: Neural Turing Machines<br/><i>External memory concept</i>"]
    CH14 --> CH22["Chapter 22: Relational Reasoning<br/><i>Similar reasoning tasks</i>"]
    
    style CH14 fill:#ff6b6b,color:#fff
```

---

## 14.14 Key Equations Summary

### Memory Update

$$M'_t = \text{MultiHeadAttention}([x_t; M_{t-1}], [x_t; M_{t-1}], [x_t; M_{t-1}])$$

$$M_t = \text{LayerNorm}(M'_t + \text{FF}(M'_t))$$

### Self-Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Multi-Head

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

### Hidden State

$$h_t = \text{ReadHead}(M_t)$$

---

## 14.15 Chapter Summary

```mermaid
graph TB
    subgraph "Key Takeaways"
        T1["RRNNs combine RNN sequential<br/>processing with self-attention"]
        T2["Relational Memory Core uses<br/>multi-head self-attention"]
        T3["Memory slots can attend to<br/>each other for reasoning"]
        T4["Better than standard RNNs<br/>on relational tasks"]
        T5["Bridge between RNNs and<br/>Transformers"]
    end
    
    T1 --> C["Relational RNNs demonstrate how<br/>self-attention can enhance recurrent<br/>networks, enabling explicit relational<br/>reasoning while maintaining sequential<br/>processing capabilities."]
    T2 --> C
    T3 --> C
    T4 --> C
    T5 --> C
    
    style C fill:#ffe66d,color:#000,stroke:#000,stroke-width:2px
```

### In One Sentence

> **Relational Recurrent Neural Networks enhance standard RNNs with a self-attention-based memory core, enabling explicit relational reasoning between memory slots while maintaining sequential processing.**

---

## 🎉 Part III Complete!

You've finished the **Sequence Models and Recurrent Networks** section. You now understand:
- How RNNs generate text and code (Chapter 11)
- How LSTMs solve vanishing gradients (Chapter 12)
- How to properly regularize RNNs (Chapter 13)
- How attention can enhance RNNs (Chapter 14)

**Next up: Part IV - Attention and Transformers**, where we explore the attention mechanism that revolutionized sequence modeling!

---

## Exercises

1. **Conceptual**: Explain how self-attention in RRNNs enables relational reasoning that standard RNNs struggle with.

2. **Comparison**: Compare the computational complexity of RRNN vs standard RNN vs Transformer for a sequence of length T.

3. **Implementation**: Implement a simple RRNN with 4 memory slots and single-head attention. Test on a simple relational reasoning task.

4. **Analysis**: Why might RRNNs be less popular than Transformers today? What are the trade-offs?

---

## References & Further Reading

| Resource | Link |
|----------|------|
| Original Paper (Santoro et al., 2018) | [arXiv:1806.01822](https://arxiv.org/abs/1806.01822) |
| Attention Is All You Need | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
| Neural Turing Machines | [arXiv:1410.5401](https://arxiv.org/abs/1410.5401) |
| bAbI Dataset | [GitHub](https://github.com/facebook/babi) |
| Relational Networks Paper | [arXiv:1706.01427](https://arxiv.org/abs/1706.01427) |

---

**Next Chapter:** [Chapter 15: Neural Machine Translation with Attention](../part-4-attention/15-nmt-attention.md) — We begin Part IV by exploring how attention mechanisms were first successfully applied to sequence-to-sequence models, solving the bottleneck problem in neural machine translation.

---

[← Back to Part III](./README.md) | [Table of Contents](../../README.md)

