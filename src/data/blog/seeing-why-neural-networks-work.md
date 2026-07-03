---
title: Seeing why neural networks work
pubDatetime: 2026-07-03T11:30:00Z
modDatetime: 2026-07-03T13:00:00Z
featured: true
draft: false
tags:
  - deep-learning
  - neural-networks
  - visualizations
description: Four interactive visual proofs — activations, depth, embeddings, and generalization — that make core neural-network ideas concrete.
---

Most of the big claims in deep learning sound almost too simple when you first hear them. A network without nonlinearity can only draw a straight line. Stack five linear layers and you still get one line. Embeddings learn similarity from next-token prediction alone. Give a huge model a tiny dataset and it memorizes; give it more data and it generalizes. I wanted to _see_ each of those happen rather than take them on faith, so this page walks through four tiny experiments you can run in the browser.

Each section (**S1-1** through **S1-4**) is a self-contained claim, setup, and live demo. Start anywhere, but they build on each other in spirit.

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
## S1-2 · Depth without nonlinearity is a lie

### The claim

Stacking more layers sounds like it should buy you more power. But if every layer is just a matrix multiply — no ReLU, no sigmoid in the middle, nothing that actually _bends_ the output — you're not building depth at all. You're building one fat linear map in disguise.

Five linear layers in a row is still one straight decision boundary. Same as one layer. Same ceiling on accuracy. The depth is a costume.

Put ReLU between those same five layers, though, and the stack can finally curve. Same depth, same data, same training — but now it can wrap the ring.

### The setup

Same concentric rings as before: inner class, outer class, no straight line that splits them cleanly.

This time I train **three** tiny classifiers on identical points:

- **A** — one linear layer with a sigmoid (logistic regression),
- **B** — five stacked linear layers (2→8→8→8→8→1) with **no** activation between them, then a sigmoid on top, and
- **C** — the same five-layer stack, but with ReLU after each hidden layer.

Everything else matches: binary cross-entropy, full-batch gradient descent, same learning rate. The only difference is whether anything nonlinear happens _between_ the matrix multiplies.

### Watch it happen

Press **Train all three** and watch the panels side by side. Each one draws confidence contours with the thick line at 50/50, and the training accuracy ticks up live.

A and B should look like twins — straight boundaries, accuracy stuck near a coin flip. That's not a bug. B really is just A wearing five hats; the bonus panel multiplies B's weight matrices together and prints the single effective matrix and bias that prove it.

C is the outlier. Same depth as B, but the ReLUs let it fold the plane and climb toward ~99% on the rings.

Tweak **noise** or the **learning rate** if you want; either one reshuffles the data and restarts all three from fresh weights.

<div class="viz-linear-collapse" data-viz="linear-collapse"></div>
<script src="/visualizations/linear-collapse.js"></script>

The takeaway is blunt: depth alone doesn't buy expressiveness. Nonlinearity between layers is what turns a stack of matrices into something that can bend. Without it, you're always drawing a line — no matter how tall the stack gets.
