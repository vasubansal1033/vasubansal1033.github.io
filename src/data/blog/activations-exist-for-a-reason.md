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

Hit **Train both** below and watch it happen live. Both models train on the exact same data with the same optimizer — the only difference is model B's ReLU hidden layer. Each panel renders the decision field as smooth confidence contours, with the thick line marking the boundary (where the model is 50/50), the data points on top, and the live training accuracy. The chart underneath tracks each model's training loss.

Play with the knobs: change the **dataset** (rings, moons, spiral, XOR), add **noise**, tweak the **learning rate**, or resize the **ReLU hidden layer**. Every change reshuffles the data and resets training.

<div class="viz-rings" data-viz="rings-activation"></div>
<script src="/visualizations/rings-activation.js"></script>

The linear model draws a straight boundary and flattens out near a coin-flip — it is physically incapable of bending. The ReLU model folds the plane, wraps the inner ring, and climbs toward ~99%.

Only the activation changed.

Try switching the dataset to **XOR** — the historical example that stalled neural-net research for years. Model A sits near 50% while model B solves it cleanly.
