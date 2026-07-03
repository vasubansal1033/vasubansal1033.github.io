---
title: Activations exist for a reason
pubDatetime: 2026-07-03T11:30:00Z
featured: true
draft: false
tags:
  - deep-learning
  - neural-networks
  - visualizations
description: Why nonlinear activations matter — a visual proof using concentric rings and decision boundaries.
---

A model with no nonlinearity can only draw a straight boundary. That sounds abstract until you put it on data it cannot possibly separate.

This post is section **S1-1** of a longer series. More sections will be added to this same page.

## S1-1 · Activations exist for a reason

### Claim

Without a nonlinearity, a neural network collapses to a single linear transformation. No matter how many weight matrices you stack, the composition is still linear — so the decision boundary is always a straight line (or hyperplane in higher dimensions).

That is fine for linearly separable data. It fails completely on interleaved or concentric structures, like two noisy rings.

Add one hidden layer with a ReLU activation and the same input/output dimensions suddenly have enough expressive power to wrap a curved boundary around the inner ring.

### Build

We generate ~300 noisy 2D points arranged as two concentric rings:

- **Inner ring** (class 0): radius ≈ 1.5, plus Gaussian noise
- **Outer ring** (class 1): radius ≈ 3.5, plus Gaussian noise

The two classes are not linearly separable — no straight line can cleanly split them.

We train two tiny classifiers on the same data:

1. **Linear model** — a single layer with sigmoid output (logistic regression)
2. **ReLU model** — one hidden layer of 16 units with ReLU, then sigmoid output

Same loss (binary cross-entropy), same optimizer (full-batch gradient descent). The only architectural difference is the activation.

### Proof

Run the experiment below. Each panel shows the decision region (background), the data points (dots), and the final accuracy after training.

<div class="viz-rings" data-viz="rings-activation"></div>
<script src="/visualizations/rings-activation.js"></script>

The linear model draws a straight boundary and stalls near ~55% accuracy — barely better than guessing. The ReLU model bends around the inner ring and reaches ~99%.

Only the activation changed.
