# Chapter 8: Deep Residual Learning for Image Recognition

> *"We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously."*

**Based on:** "Deep Residual Learning for Image Recognition" (Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, 2015)

📄 **Original Paper:** [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) | [CVPR 2016 Best Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)

---

## 8.1 The Depth Revolution

After AlexNet's 8 layers (Chapter 6), researchers raced to go deeper. VGGNet reached 19 layers. But then something strange happened:

**Adding more layers made performance WORSE.**

This wasn't overfitting—even training error increased! There was something fundamentally limiting about deep networks.

```mermaid
graph LR
    subgraph "The Paradox (2015)"
        A["20-layer network<br/>Training error: 10%"]
        B["56-layer network<br/>Training error: 12%"]
    end
    
    Q["Why does more depth<br/>hurt TRAINING error?"]
    
    A --> Q
    B --> Q
    
    style Q fill:#ff6b6b,color:#fff
```

ResNet solved this problem and enabled training networks with **152 layers**—and even 1000+ layers in experiments.

---

## 8.2 The Degradation Problem

### Not Overfitting—Something Deeper

```mermaid
xychart-beta
    title "The Degradation Problem"
    x-axis "Iterations" [0, 20, 40, 60, 80, 100]
    y-axis "Error %" 0 --> 40
    line "20-layer train" [35, 18, 12, 10, 9, 8]
    line "20-layer test" [38, 22, 16, 14, 13, 12]
    line "56-layer train" [38, 25, 18, 15, 14, 13]
    line "56-layer test" [40, 28, 22, 19, 18, 17]
```

Key observation: The 56-layer network has **higher training error** than the 20-layer network. This rules out overfitting!

### The Identity Mapping Argument

Theoretically, a deeper network should never be worse:

> If a shallow network achieves some accuracy, a deeper network could just learn **identity mappings** for the extra layers and match it.

```mermaid
graph LR
    subgraph "Theoretical Construction"
        S["20-layer network<br/>(good performance)"]
        I["+ 36 identity layers<br/>(pass input through)"]
        D["= 56-layer network<br/>(should match performance)"]
    end
    
    S --> I --> D
    
    P["But in practice,<br/>networks can't learn this!"]
    
    D --> P
    
    style P fill:#ff6b6b,color:#fff
```

**The problem**: Stacked nonlinear layers have difficulty learning identity mappings.

---

## 8.3 The Residual Learning Framework

### The Key Insight

Instead of learning H(x) directly, learn the **residual** F(x) = H(x) - x:

$$H(x) = F(x) + x$$

```mermaid
graph LR
    subgraph "Traditional Block"
        X1["x"] --> L1["Conv-ReLU-Conv"]
        L1 --> H1["H(x)"]
    end
    
    subgraph "Residual Block"
        X2["x"] --> L2["Conv-ReLU-Conv"]
        X2 --> ADD["⊕"]
        L2 -->|"F(x)"| ADD
        ADD --> H2["H(x) = F(x) + x"]
    end
    
    K["The skip connection<br/>makes identity easy:<br/>just set F(x) = 0"]
    
    ADD --> K
    
    style K fill:#ffe66d,color:#000
```

### Why Residuals Are Easier to Learn

```mermaid
graph TB
    subgraph "Learning Identity"
        T["Traditional: Learn H(x) = x<br/>Need weights to compute identity"]
        R["Residual: Learn F(x) = 0<br/>Just push weights toward zero"]
    end
    
    R --> E["Much easier optimization!<br/>Weights naturally initialize near zero"]
    
    style E fill:#4ecdc4,color:#fff
```

If the optimal function is close to identity, the residual F(x) is close to zero—which is easy to learn.

---

## 8.4 The Residual Block Architecture

### Basic Building Block

```mermaid
graph TB
    subgraph "Basic Residual Block"
        X["x"]
        C1["3×3 Conv, 64"]
        BN1["Batch Norm"]
        R1["ReLU"]
        C2["3×3 Conv, 64"]
        BN2["Batch Norm"]
        ADD["⊕"]
        R2["ReLU"]
        OUT["output"]
    end
    
    X --> C1 --> BN1 --> R1 --> C2 --> BN2 --> ADD
    X -->|"identity shortcut"| ADD
    ADD --> R2 --> OUT
```

### Bottleneck Block (For Deeper Networks)

For ResNet-50 and beyond, use a bottleneck design:

```mermaid
graph TB
    subgraph "Bottleneck Block"
        X["x (256 channels)"]
        C1["1×1 Conv, 64<br/>(reduce)"]
        BN1["BN + ReLU"]
        C2["3×3 Conv, 64<br/>(process)"]
        BN2["BN + ReLU"]
        C3["1×1 Conv, 256<br/>(expand)"]
        BN3["Batch Norm"]
        ADD["⊕"]
        R["ReLU"]
        OUT["output (256 channels)"]
    end
    
    X --> C1 --> BN1 --> C2 --> BN2 --> C3 --> BN3 --> ADD
    X -->|"identity"| ADD
    ADD --> R --> OUT
    
    E["1×1 convs reduce/expand channels<br/>3×3 conv works in low dimension<br/>→ Saves computation"]
    
    style E fill:#ffe66d,color:#000
```

### Handling Dimension Changes

When spatial dimensions or channels change:

```mermaid
graph TB
    subgraph "Projection Shortcut"
        X["x (64 ch, 56×56)"]
        CONV["Conv layers<br/>(→ 128 ch, 28×28)"]
        PROJ["1×1 Conv, stride 2<br/>(projection)"]
        ADD["⊕"]
        OUT["output (128 ch, 28×28)"]
    end
    
    X --> CONV --> ADD
    X --> PROJ --> ADD
    ADD --> OUT
```

---

## 8.5 The Full ResNet Architecture

### ResNet Variants

| Model | Layers | Parameters | Top-5 Error |
|-------|--------|------------|-------------|
| ResNet-18 | 18 | 11.7M | 10.92% |
| ResNet-34 | 34 | 21.8M | 9.46% |
| ResNet-50 | 50 | 25.6M | 7.48% |
| ResNet-101 | 101 | 44.5M | 6.58% |
| ResNet-152 | 152 | 60.2M | 6.16% |

### ResNet-34 Architecture

```mermaid
graph TB
    subgraph "ResNet-34 Structure"
        I["Input 224×224×3"]
        C0["7×7 Conv, 64, stride 2"]
        P0["3×3 MaxPool, stride 2"]
        
        subgraph "Stage 1 (56×56)"
            S1["3 Basic Blocks<br/>64 filters"]
        end
        
        subgraph "Stage 2 (28×28)"
            S2["4 Basic Blocks<br/>128 filters"]
        end
        
        subgraph "Stage 3 (14×14)"
            S3["6 Basic Blocks<br/>256 filters"]
        end
        
        subgraph "Stage 4 (7×7)"
            S4["3 Basic Blocks<br/>512 filters"]
        end
        
        GAP["Global Avg Pool"]
        FC["FC 1000"]
        SM["Softmax"]
    end
    
    I --> C0 --> P0 --> S1 --> S2 --> S3 --> S4 --> GAP --> FC --> SM
```

### The Numbers: ResNet-50

```mermaid
graph LR
    subgraph "ResNet-50 Layer Count"
        C1["conv1: 1 layer"]
        B1["Stage 1: 3 blocks × 3 = 9"]
        B2["Stage 2: 4 blocks × 3 = 12"]
        B3["Stage 3: 6 blocks × 3 = 18"]
        B4["Stage 4: 3 blocks × 3 = 9"]
        FC["FC: 1 layer"]
    end
    
    T["Total: 1 + 9 + 12 + 18 + 9 + 1 = 50"]
    
    C1 --> T
    B1 --> T
    B2 --> T
    B3 --> T
    B4 --> T
    FC --> T
```

---

## 8.6 Why Skip Connections Work

### Gradient Flow

The key benefit is improved **gradient flow** during backpropagation:

```mermaid
graph TB
    subgraph "Without Skip Connections"
        G1["Gradient must flow through<br/>every layer's weights"]
        G2["Vanishing gradients in deep nets"]
        G3["Early layers barely updated"]
    end
    
    subgraph "With Skip Connections"
        S1["Gradient has direct path<br/>through identity shortcuts"]
        S2["Gradients can 'skip' problematic layers"]
        S3["All layers receive strong gradients"]
    end
    
    G1 --> G2 --> G3
    S1 --> S2 --> S3
    
    style S3 fill:#4ecdc4,color:#fff
    style G3 fill:#ff6b6b,color:#fff
```

### Mathematical View

For a residual block:
$$y = F(x) + x$$

The gradient:
$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \left(\frac{\partial F}{\partial x} + 1\right)$$

The **+1** ensures gradients always flow, even if ∂F/∂x is small!

```mermaid
graph LR
    subgraph "Gradient Highway"
        L["Loss"]
        Y["y = F(x) + x"]
        X["x"]
        
        L -->|"∂L/∂y"| Y
        Y -->|"∂L/∂y × (∂F/∂x + 1)"| X
    end
    
    P["The '+1' prevents<br/>gradient vanishing!"]
    
    Y --> P
    
    style P fill:#ffe66d,color:#000
```

---

## 8.7 Ensemble Interpretation

### ResNets as Implicit Ensembles

A remarkable insight: ResNets can be viewed as an **ensemble of shallow networks**.

```mermaid
graph TB
    subgraph "Unrolled View"
        X["x"]
        B1["Block 1: F₁(x)"]
        B2["Block 2: F₂(·)"]
        B3["Block 3: F₃(·)"]
        OUT["Output"]
    end
    
    X --> B1
    X --> OUT
    B1 --> B2
    B1 --> OUT
    B2 --> B3
    B2 --> OUT
    B3 --> OUT
    
    E["Each possible path through<br/>skip connections = one 'sub-network'<br/>ResNet-20 has 2^20 paths!"]
    
    style E fill:#ffe66d,color:#000
```

With n blocks, there are 2^n paths. The network is implicitly averaging over exponentially many sub-networks!

---

## 8.8 Experimental Results

### ImageNet Performance

```mermaid
xychart-beta
    title "ImageNet Top-5 Error (2012-2015)"
    x-axis ["AlexNet", "VGG-19", "GoogLeNet", "ResNet-152"]
    y-axis "Error %" 0 --> 20
    bar [15.3, 7.3, 6.7, 3.6]
```

### The Depth Experiment

```mermaid
xychart-beta
    title "Effect of Depth with ResNet"
    x-axis "Layers" [20, 32, 44, 56, 110]
    y-axis "Error %" 0 --> 12
    line "Plain Network" [9.5, 10.5, 11.2, 12.0, 13.5]
    line "ResNet" [8.5, 7.5, 6.8, 6.2, 5.5]
```

Without skip connections: deeper = worse
With skip connections: deeper = better!

### Going Extremely Deep

The paper trained a **1202-layer** ResNet on CIFAR-10:
- It worked! (no optimization issues)
- But performance was slightly worse than 110-layer (overfitting)

---

## 8.9 Design Choices and Ablations

### Identity vs. Projection Shortcuts

```mermaid
graph TB
    subgraph "Shortcut Options"
        A["Option A: Zero-padding<br/>(for dimension change)"]
        B["Option B: Projection shortcuts<br/>(only for dimension change)"]
        C["Option C: All projection shortcuts"]
    end
    
    R["Results: B is best<br/>C wastes parameters<br/>A slightly worse than B"]
    
    A --> R
    B --> R
    C --> R
```

### Pre-activation vs. Post-activation

The original ResNet uses "post-activation":
- Conv → BN → ReLU → Conv → BN → Add → ReLU

Chapter 9 will explore "pre-activation" which is even better:
- BN → ReLU → Conv → BN → ReLU → Conv → Add

---

## 8.10 Beyond Image Classification

### ResNet for Other Tasks

```mermaid
graph TB
    subgraph "ResNet Applications"
        CLS["Image Classification<br/>(original task)"]
        DET["Object Detection<br/>(Faster R-CNN backbone)"]
        SEG["Semantic Segmentation<br/>(FCN, DeepLab)"]
        POSE["Pose Estimation<br/>(feature extractor)"]
        GEN["Image Generation<br/>(ResNet blocks in GANs)"]
    end
    
    R["ResNet became the<br/>default backbone for<br/>almost everything"]
    
    CLS --> R
    DET --> R
    SEG --> R
    POSE --> R
    GEN --> R
    
    style R fill:#ffe66d,color:#000
```

### Transfer Learning with ResNet

Pre-trained ResNets are the foundation of transfer learning in vision:

```python
# Common pattern (pseudocode)
model = resnet50(pretrained=True)

# Replace final layer for new task
model.fc = nn.Linear(2048, num_classes)

# Fine-tune
train(model, your_dataset)
```

---

## 8.11 The Broader Impact

### What ResNet Changed

```mermaid
timeline
    title Impact of ResNet
    2015 : ResNet paper
         : 152 layers trained successfully
         : Won ImageNet, COCO, etc.
    2016 : CVPR Best Paper
         : Became standard backbone
    2017 : ResNeXt, DenseNet
         : Variations on the theme
    2018 : ResNet in everything
         : Detection, segmentation, GANs
    2020s : Still widely used
          : Foundation for Vision Transformers
```

### The Skip Connection Legacy

Skip connections appeared everywhere after ResNet:

| Architecture | Skip Connection Variant |
|--------------|------------------------|
| DenseNet | Connect to ALL previous layers |
| U-Net | Skip connections across encoder-decoder |
| Highway Networks | Learned gating |
| Transformers | Residual connections after attention |

---

## 8.12 Connection to Earlier Chapters

```mermaid
graph TB
    CH8["Chapter 8<br/>ResNet"]
    
    CH8 --> CH6["Chapter 6: AlexNet<br/><i>ResNet continues the<br/>depth revolution</i>"]
    CH8 --> CH7["Chapter 7: CS231n<br/><i>Understanding why<br/>gradients vanish</i>"]
    CH8 --> CH3["Chapter 3: Simple NNs<br/><i>Regularization through<br/>architecture</i>"]
    CH8 --> CH9["Chapter 9: Identity Mappings<br/><i>Improving ResNet further</i>"]
    CH8 --> CH16["Chapter 16: Transformers<br/><i>Also use residual connections!</i>"]
    
    style CH8 fill:#ff6b6b,color:#fff
```

---

## 8.13 Implementation Details

### Key Training Settings

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | SGD with momentum 0.9 |
| Learning rate | 0.1, divided by 10 at epochs 30, 60 |
| Weight decay | 0.0001 |
| Batch size | 256 |
| Epochs | 90 |
| Data augmentation | Random crop, horizontal flip |

### Weight Initialization

He initialization for ReLU networks:

$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)$$

This accounts for ReLU zeroing half the activations.

---

## 8.14 Key Equations Summary

### Residual Learning

$$y = F(x, \{W_i\}) + x$$

### Gradient Through Residual Block

$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \left(1 + \frac{\partial F}{\partial x}\right)$$

### Bottleneck Computation

$$F(x) = W_3 \cdot \sigma(W_2 \cdot \sigma(W_1 \cdot x))$$

Where W₁ is 1×1 (reduce), W₂ is 3×3, W₃ is 1×1 (expand)

### He Initialization

$$\text{Var}(W) = \frac{2}{n_{in}}$$

---

## 8.15 Chapter Summary

```mermaid
graph TB
    subgraph "Key Takeaways"
        T1["Skip connections solve<br/>the degradation problem"]
        T2["Deeper networks can now<br/>be trained effectively"]
        T3["Gradients flow through<br/>identity shortcuts"]
        T4["ResNets work as implicit<br/>ensembles of paths"]
        T5["Became the default<br/>backbone for vision"]
    end
    
    T1 --> C["ResNet's simple insight—<br/>learn residuals, not direct mappings—<br/>enabled training networks 10× deeper<br/>and transformed computer vision"]
    T2 --> C
    T3 --> C
    T4 --> C
    T5 --> C
    
    style C fill:#ffe66d,color:#000,stroke:#000,stroke-width:2px
```

### In One Sentence

> **ResNet introduced skip connections that let networks learn residual functions, solving the degradation problem and enabling training of 100+ layer networks that achieved superhuman performance on ImageNet.**

---

## Exercises

1. **Conceptual**: Explain in your own words why learning F(x) = 0 is easier than learning H(x) = x for a stack of nonlinear layers.

2. **Calculation**: In a bottleneck block with 256 input channels, the middle 3×3 conv has 64 channels. How many parameters does this block have? Compare to a basic block with 256 channels.

3. **Implementation**: Implement a ResNet-18 from scratch and train it on CIFAR-10. Plot training curves for with and without skip connections.

4. **Analysis**: ResNet-1202 works but is worse than ResNet-110 on CIFAR. Why might this be? What does this tell us about the limits of depth?

---

## References & Further Reading

| Resource | Link |
|----------|------|
| Original Paper (He et al., 2015) | [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) |
| Identity Mappings Paper (He et al., 2016) | [arXiv:1603.05027](https://arxiv.org/abs/1603.05027) |
| ResNeXt Paper | [arXiv:1611.05431](https://arxiv.org/abs/1611.05431) |
| DenseNet Paper | [arXiv:1608.06993](https://arxiv.org/abs/1608.06993) |
| PyTorch ResNet Implementation | [torchvision](https://pytorch.org/vision/stable/models/resnet.html) |
| Residual Networks as Ensembles | [arXiv:1605.06431](https://arxiv.org/abs/1605.06431) |
| He Initialization Paper | [arXiv:1502.01852](https://arxiv.org/abs/1502.01852) |

---

**Next Chapter:** [Chapter 9: Identity Mappings in Deep Residual Networks](./09-identity-mappings.md) — We explore how to improve ResNet further by rethinking the order of operations within residual blocks.

---

[← Back to Part II](./README.md) | [Table of Contents](../../README.md)

