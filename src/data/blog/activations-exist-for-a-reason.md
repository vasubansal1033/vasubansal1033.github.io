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

Here's a claim that sounds almost too simple to be interesting: a neural network without any nonlinearity can only ever draw a straight line. It doesn't matter how many layers you give it — stack a hundred of them and you still get a line. Drop in a single nonlinear activation, though, and the whole thing can suddenly bend. I wanted to actually _see_ that happen rather than take it on faith, so this post trains two tiny models side by side and lets you watch the difference.

This is section **S1-1** of a longer series. I'll keep adding sections to this same page as I go.

## S1-1 · Activations exist for a reason

### The claim

Take away the nonlinearity and a neural network collapses into one big linear transformation. Multiplying a stack of weight matrices together just gives you another matrix, so the boundary it can draw is always a straight line (a flat plane once you're in higher dimensions).

That's perfectly fine when your data can be split by a straight line. It falls apart the moment the classes are tangled together — say, one class sitting inside a ring of the other. No line cuts that cleanly.

Give the network a single hidden layer with a ReLU in it, though, and it gains just enough flexibility to curve the boundary right around the inner ring.

### The setup

I scatter about 300 noisy points into two concentric rings — the inner ring is one class, the outer ring the other. There's no straight line that separates them, which is exactly the point.

Then I train two small classifiers on that same data:

- a **linear model**: a single layer with a sigmoid on top (plain logistic regression), and
- a **ReLU model**: one hidden layer of 16 ReLU units, then a sigmoid.

Everything else is held constant — same loss (binary cross-entropy), same optimizer (full-batch gradient descent). The activation is the _only_ thing that differs between them.

### Watch it happen

Press **Train both** and let them run. Each panel draws the model's decision field as smooth confidence contours, with the thick line marking the boundary (where the model is genuinely 50/50) and the training accuracy updating live. The chart below tracks each model's loss as it learns.

Poke at the controls too: swap the **dataset** (rings, moons, spiral, XOR), turn up the **noise**, change the **learning rate**, or grow the **ReLU hidden layer**. Anything you touch reshuffles the data and restarts training.

<div class="viz-rings" data-viz="rings-activation"></div>
<script src="/visualizations/rings-activation.js"></script>

Watch what the linear model does: it draws its one straight line, wobbles a little, and settles near a coin flip. It can't bend, so that's the best it will ever manage. The ReLU model folds the plane, wraps itself around the inner ring, and climbs toward ~99%. Same data, same training — one architectural difference.

For the classic version of this, switch the dataset to **XOR**. The linear model gets stuck around 50% while the ReLU model handles it without breaking a sweat — and that little problem carries more history than you'd expect.

### A bit of history, and why ReLU won

An activation is really just a "bend" you apply to a neuron's output, and different bends behave differently. The old favourites, **sigmoid** and **tanh**, are smooth S-curves that squash everything into a fixed range. They work, but they _saturate_: push the input far in either direction and the curve goes flat, its slope heading toward zero. In a deep network those tiny slopes multiply together and the gradient all but vanishes, so training slows to a crawl.

That XOR example above isn't one I picked at random. In 1969, Minsky and Papert proved a single-layer perceptron can't represent XOR at all, and the disappointment that followed helped kick off the first **AI winter** — years when funding and interest in neural nets largely dried up. The way out turned out to be stacking layers _with_ a nonlinearity between them and training them with backpropagation (Rumelhart, Hinton & Williams, 1986).

**ReLU** — just `max(0, z)` — became the default for a few down-to-earth reasons. On its active side the slope is exactly 1, so it doesn't shrink gradients the way sigmoid does. It's trivially cheap to compute. And it naturally switches off a lot of neurons (anything negative becomes zero), which keeps things sparse. Its one real annoyance is the "dying ReLU" problem, where a neuron gets stuck on the flat side and stops learning — which is why softer variants like **Leaky ReLU** and the smooth **GELU** (the one inside most modern Transformers) exist. But the headline never changes, and the plots above make it concrete: no nonlinearity, no bend — and without a bend, that ring never gets wrapped.

#### Further watching

If you'd like a more visual, intuition-first take on this, I really enjoyed [Hidden Symmetry: Why Deep Learning is Possible](https://www.youtube.com/watch?v=2qXF8JHcU5E). A few things worth holding onto while you watch:

- Nonlinearity is the whole game. Stacking linear layers only ever gets you another straight line; the activation is what lets a network bend space and carve out curved boundaries.
- The bends add up. Each hidden unit contributes one simple fold, and stacking enough of them lets a network approximate almost any shape.
- Shape and slope are two separate jobs. An activation has to be nonlinear (so the network can curve) _and_ keep a healthy gradient across depth (so it can actually train) — which is precisely why ReLU pushed sigmoid and tanh aside.
