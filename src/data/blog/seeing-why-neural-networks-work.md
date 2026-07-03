---
title: Seeing why neural networks work
pubDatetime: 2026-07-03T11:30:00Z
modDatetime: 2026-07-03T15:40:00Z
featured: true
draft: false
tags:
  - deep-learning
  - neural-networks
  - visualizations
  - vibe-coded
description: Four interactive visual proofs — activations, depth, embeddings, and generalization — that make core neural-network ideas concrete.
---

Most of the big claims in deep learning sound almost too simple when you first hear them. A network without nonlinearity can only draw a straight line. Stack five linear layers and you still get one line. Embeddings learn similarity from next-token prediction alone. Give a huge model a tiny dataset and it memorizes; give it more data and it generalizes. I wanted to _see_ each of those happen rather than take them on faith, so this page walks through four tiny experiments you can run in the browser.

Each section (**S1-1** through **S1-5**) is a self-contained claim, setup, and live demo. Start anywhere, but they build on each other in spirit.

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

The longer arc — the one that runs from wartime engineering all the way to `max(0, z)` — is genuinely one of the best stories in science, and it's worth telling properly.

#### The longer story: from anti-aircraft guns to ImageNet

Here's the whole arc on one timeline — press **Play**, or click any milestone to read what happened.

<div class="viz-dl-history" data-viz="dl-history"></div>
<script src="/visualizations/dl-history-timeline.js"></script>

**The 1940s — feedback, logic, and the first artificial neuron.** Oddly enough, the thread starts with anti-aircraft guns. During World War II, Norbert Wiener worked on the problem of automatically aiming artillery at planes that were actively dodging — you fire not where the target _is_ but where it _will be_, and you correct continuously from error. Generalizing that idea of self-correcting **feedback** gave birth to **cybernetics**: the study of control and communication in animals and machines. In the same years, Warren McCulloch and Walter Pitts (1943) showed that a network of idealized "all-or-nothing" neurons — simple threshold switches that fire once their inputs cross a line — could in principle compute any logical proposition. That's the founding move of the whole field: a _neuron as a switch_, and thought as computation over switches.

**The 1950s–60s — the perceptron and its ceiling.** Frank Rosenblatt turned that static switch into something that _learns_. His **perceptron** (1958) attached adjustable **weights** to each input and a rule for nudging those weights whenever the machine guessed wrong — so experience literally reshaped the neuron. It caused a genuine sensation; there was talk of machines that would soon walk, talk, and see. The linear model you trained at the top of this page is, essentially, a modern perceptron. But in 1969 Marvin Minsky and Seymour Papert published a careful mathematical takedown: a single-layer perceptron simply cannot represent **XOR** (the exact wall you can watch the linear model hit above), and more generally can't handle problems that aren't linearly separable. The critique was correct, widely read, and devastating — funding evaporated and the first **AI winter** set in.

**The 1980s — backprop thaws the winter, sigmoid warms it up.** The revival came from a small, stubborn group sometimes half-jokingly called the "deep learning mafia" — Geoffrey Hinton, Yoshua Bengio, Yann LeCun and collaborators. The key unlock was **backpropagation** (popularized by Rumelhart, Hinton & Williams, 1986): a way to send the error signal backward through _multiple_ layers and update every weight, so networks could finally be more than one layer deep. To do that they needed a smooth, differentiable activation, and the **sigmoid** S-curve fit the bill. Multi-layer nets could now, in principle, bend space and solve XOR — exactly what the ReLU model demonstrates above.

**The 1990s — the vanishing gradient, and the reign of SVMs.** There was a catch that took years to fully diagnose. Sigmoid is _too polite_: far from zero its curve goes almost flat, so its slope is nearly zero out there. Stack many such layers and backprop multiplies all those tiny slopes together, so the gradient reaching the early layers **vanishes** to almost nothing — those layers barely learn. Deep networks were theoretically possible but practically untrainable. Into that vacuum stepped Vladimir Vapnik's **Support Vector Machines**. SVMs were everything neural nets weren't at the time: mathematically clean, backed by solid theory, and **convex** — one global optimum, guaranteed to be found. For much of the decade the field's respectable choice was the SVM, and neural networks were quietly dismissed by many as unprincipled "alchemy."

**2011–2012 — ReLU, and the dam breaks.** In 2011, Xavier Glorot, Antoine Bordes, and Yoshua Bengio pinned the blame squarely on the activation function and reached for something almost embarrassingly simple: the **Rectified Linear Unit**, `max(0, z)`. Zero for negative inputs, and a straight 45° line for positive ones. Because its slope on the active side is exactly 1, gradients pass through deep stacks without shrinking — the vanishing-gradient problem largely evaporates. Then came the exclamation point: in 2012, **AlexNet** (Krizhevsky, Sutskever & Hinton) paired ReLU with GPU training and cut the **ImageNet** error rate so dramatically that it ended the paradigm debate overnight. Within a couple of years, essentially the entire field had switched to deep nets.

You can see exactly why the activation was the bottleneck. The left panel plots each function; the middle plots its **slope** — the quantity backprop multiplies together layer after layer. The right panel takes each function's best-case slope and raises it to the power of the network depth. Sigmoid's best case is only 0.25, so even optimistically the gradient shrinks fourfold per layer and collapses toward zero within a handful of layers; ReLU's is 1, so it holds flat no matter how deep you go. Drag the **depth** slider and watch sigmoid fall off a cliff while ReLU stays put.

<div class="viz-activation-gradient" data-viz="activation-gradient"></div>
<script src="/visualizations/activation-gradient.js"></script>

**The irony at the end.** After all that — decades of theory, winters, and reinvention — the component at the heart of GPT-scale models is a hard "off" below zero and a plain linear "on" above it. We came almost full circle to the McCulloch–Pitts switch of the 1940s; the revolution was less a new idea than an old one finally allowed to scale. The [video linked below](#further-watching) tells this arc beautifully.

**ReLU** became the default for a few down-to-earth reasons. On its active side the slope is exactly 1, so it doesn't shrink gradients the way sigmoid does. It's trivially cheap to compute. And it naturally switches off a lot of neurons (anything negative becomes zero), which keeps things sparse. Its one real annoyance is the "dying ReLU" problem, where a neuron gets stuck on the flat side and stops learning — which is why softer variants like **Leaky ReLU** and the smooth **GELU** (the one inside most modern Transformers) exist. But the headline never changes, and the plots above make it concrete: no nonlinearity, no bend — and without a bend, that ring never gets wrapped.

#### Further watching

For the full history — from 1940s cybernetics through the vanishing-gradient years to ReLU's win at ImageNet — [From Anti-Aircraft Guns to ImageNet: How Activation Functions Have Shaped Deep Learning](https://www.youtube.com/watch?v=zcWxqTCJQTo) is a wonderfully told version of the story above.

If you'd like a more visual, intuition-first take on _why_ these networks train at all, I also really enjoyed [Hidden Symmetry: Why Deep Learning is Possible](https://www.youtube.com/watch?v=2qXF8JHcU5E). A few things worth holding onto while you watch:

- Nonlinearity is the whole game. Stacking linear layers only ever gets you another straight line; the activation is what lets a network bend space and carve out curved boundaries.
- The bends add up. Each hidden unit contributes one simple fold, and stacking enough of them lets a network approximate almost any shape.
- Shape and slope are two separate jobs. An activation has to be nonlinear (so the network can curve) _and_ keep a healthy gradient across depth (so it can actually train) — which is precisely why ReLU pushed sigmoid and tanh aside.

## S1-2 · Depth without nonlinearity is a lie

### The claim

Stacking more layers sounds like it should buy you more power. But if every layer is just a matrix multiply — no ReLU, no sigmoid in the middle, nothing that actually _bends_ the output — you're not building depth at all. You're building one fat linear map in disguise.

Five linear layers in a row is still one straight decision boundary. Same as one layer. Same ceiling on accuracy. The depth is a costume.

Put ReLU between those same five layers, though, and the stack can finally curve. Same depth, same data, same training — but now it can wrap the ring.

### Why the math forces this

It's not a quirk of the optimizer — it's algebra. Write the activation of layer $l$ as $\mathbf{a}^{(l)}$, with the input as $\mathbf{a}^{(0)} = \mathbf{x}$. Each layer's weight matrix $W^{(l)}$ stores one neuron's weights per column, so its pre-activation is $\mathbf{z}^{(l)} = {W^{(l)}}^{\top}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$. With no nonlinearity, $\mathbf{a}^{(l)} = \mathbf{z}^{(l)}$, and stacking five layers just composes those affine maps:

$$
\hat{\mathbf{y}} = \mathbf{a}^{(5)} = {W^{(5)}}^{\top}\!\Big({W^{(4)}}^{\top}\!\big({W^{(3)}}^{\top}\!\big({W^{(2)}}^{\top}\!\big({W^{(1)}}^{\top}\mathbf{x} + \mathbf{b}^{(1)}\big) + \mathbf{b}^{(2)}\big) + \mathbf{b}^{(3)}\big) + \mathbf{b}^{(4)}\Big) + \mathbf{b}^{(5)}
$$

Multiplying it out, every weight matrix collapses into a single product and every bias folds into one vector:

$$
\hat{\mathbf{y}} = \underbrace{\Big({W^{(5)}}^{\top}{W^{(4)}}^{\top}{W^{(3)}}^{\top}{W^{(2)}}^{\top}{W^{(1)}}^{\top}\Big)}_{W_{\text{eff}}^{\top}}\,\mathbf{x} + \underbrace{\sum_{l=1}^{5}\Bigg(\prod_{k=l+1}^{5}{W^{(k)}}^{\top}\Bigg)\mathbf{b}^{(l)}}_{\mathbf{b}_{\text{eff}}} = W_{\text{eff}}^{\top}\,\mathbf{x} + \mathbf{b}_{\text{eff}}
$$

So no matter how many linear layers you stack, the whole network is exactly equivalent to a single layer $\hat{\mathbf{y}} = W_{\text{eff}}^{\top}\mathbf{x} + \mathbf{b}_{\text{eff}}$. That's why models **A** and **B** below can't ever differ in what they're capable of. A ReLU breaks the chain — $\mathbf{a}^{(l)} = \max(0,\, \mathbf{z}^{(l)})$ is not affine, so the matrices can no longer be merged, and the network keeps its depth.

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

### A bit of history, and why depth only counts with bends

The intuition that "deeper is better" feels natural — more layers ought to mean more computation, more abstraction, more power. That story only became true once people understood what depth actually buys you.

In 1989, Cybenko and Hornik independently proved the **universal approximation theorem**: a network with even a single hidden layer and a nonlinear activation can approximate any continuous function, given enough width. So depth wasn't strictly _necessary_ for expressiveness — width could do the job. What depth buys you, when it's real, is **efficiency**: the same function can often be represented with far fewer parameters if you stack nonlinear layers rather than blowing up one wide layer.

The catch — and the whole point of the demo above — is that those theorems assume nonlinearity _between_ layers. Without it, you're not stacking functions; you're multiplying matrices. Linear algebra has known this for a long time: composing linear maps is still a linear map. A 100-layer linear network is a marketing number, not a capability.

That distinction mattered enormously in practice. When deep nets finally started winning benchmarks in the 2010s (AlexNet on ImageNet in 2012, then ResNets, Transformers), the depth was always **nonlinear** depth — ReLUs, skip connections, attention — never a tall stack of raw matrix multiplies. ResNet's 152 layers work because each block can learn a residual _bend_; remove the activations and the whole tower collapses to one linear map, just like model **B** above.

So when someone says "we added more layers," the honest follow-up is: "and what nonlinear thing sits between them?" If the answer is nothing, the depth is a costume — and the math panel in the widget will show you the single effective matrix hiding underneath.

## S1-3 · Embeddings learn similarity from nothing but next-token

### The claim

Word embeddings don't start with a dictionary of synonyms or a taxonomy of nouns. They're just vectors that get nudged around while a model learns to guess the next token. And yet — if the training data is set up right — tokens that never appeared in the same sentence can end up sitting right next to each other in vector space, purely because they showed up in the same kinds of contexts.

That's the bit I wanted to see for myself. Not read about it in a paper, but watch three animals drift together while three fruits drift somewhere else, with nothing in the loss function that ever says "these are the same category."

### The setup

I built a tiny made-up language with three buckets of words:

- **animals**: cat, dog, cow
- **fruits**: apple, mango
- **verbs**: eat, chase, see

The grammar is a deliberately boring little cycle: `the → animal → verb → color → fruit → the`. So a "sentence" reads like "the cat sees red apple", then it loops back to "the" and starts again. Every animal is followed by any of the three verbs with equal probability, every verb by a color, every color by a fruit, and every fruit by "the".

The important part is that each category has its _own_ distinct next-token distribution — animals predict verbs, verbs predict colors, colors predict fruits, fruits predict "the". No two categories share what comes next, which is exactly what keeps their clusters apart.

Concretely, the corpus is nothing more than a list of `(current_token, next_token)` pairs. Every word in a category emits the exact same set of continuations, so same-category tokens are statistically interchangeable:

```text
animals = [cat, dog, cow]
verbs   = [eat, chase, see]
attrs   = [red, yellow]     # colors
fruits  = [apple, mango]

pairs = []
for a in animals:   pairs += [(the, a)]              # the     -> animals
for a in animals:   pairs += [(a, v) for v in verbs] # animals -> any verb
for v in verbs:     pairs += [(v, c) for c in attrs] # verbs   -> any color
for c in attrs:     pairs += [(c, f) for f in fruits] # colors  -> any fruit
for f in fruits:    pairs += [(f, the)]              # fruits  -> the
```

The only signal in there is "what tends to follow me." Nothing labels `cat` and `dog` as related — they just happen to share the identical next-token set `{eat, chase, see}`, while `eat` and `chase` share `{red, yellow}`, and `apple` and `mango` share `{the}`. Because those four signatures are all different, the four content clusters (plus "the") land in different places instead of piling up together.

The model itself is about as small as it gets: each token gets a **2-dimensional embedding** (so we can plot it directly, no PCA needed), and a softmax layer predicts the next token from that vector alone. Training is plain cross-entropy with full-batch gradient descent — no fancy architecture, no attention, no pre-training on the internet.

### Watch it happen

Hit **Train** and watch the scatter plot. At step zero the points are scattered at random. Within a few hundred steps you should see each category collapse into its own tight group — animals in one spot, verbs in another, colors and fruits in their own corners, with "the" off on its own — even though the model was never told which words belong together. If clusters overlap on screen, **scroll to zoom** and **hover** a point to read what's underneath.

Pick a token in the **Nearest neighbors** panel on the right. After training settles, cat's three closest vectors should be dog and cow, not words that merely rhyme or share letters. Same story for apple and mango. The checkmarks mean "same category" — and they should dominate once the loss has had time to drop.

The loss chart underneath the plot is there mostly as a sanity check: if it's still falling, the geometry is still moving. Pause, reset, and train again if you want to watch the clustering emerge from a different random start.

<div class="viz-embedding-clustering" data-viz="embedding-clustering"></div>
<script src="/visualizations/embedding-clustering.js"></script>

This is the distributional hypothesis made literal. Tokens that play the same role in the toy grammar get pulled into the same region of space because the model keeps asking them to predict the same kinds of continuations. Nobody hand-labelled "animal" — the category **emerged** from next-token prediction alone. Scale that idea up to billions of tokens and billions of parameters, and you get the semantic geometry that powers modern language models.

### A bit of history, and why "you shall know a word by the company it keeps"

The idea that meaning lives in context is older than neural networks. In 1957, the linguist J. R. Firth put it plainly: _"You shall know a word by the company it keeps."_ Two words that tend to appear in the same kinds of sentences — near the same neighbours — are probably playing a similar role, even if nobody ever told you they belong to the same category. That's the **distributional hypothesis**, and the scatter plot above is a miniature proof of it.

Neural nets turned that hypothesis into something you could train. Bengio et al.'s 2003 language model learned a distributed word representation as a side effect of predicting the next word. A decade later, **word2vec** (Mikolov et al., 2013) made the geometry famous: train a shallow net on next-token prediction and suddenly `king − man + woman ≈ queen`. Nobody encoded royalty, gender, or arithmetic — the relationships fell out of the co-occurrence statistics.

The famous analogy still holds. Imagine sorting people at a party by who they tend to stand next to. You never asked anyone's job title, but teachers drift toward teachers, musicians toward musicians. Same trick, different scale: `cat` and `dog` end up close because both are followed by `{eat, chase, see}` in our toy grammar, not because anyone labelled them "animal."

Modern language models are the same idea with more parameters and more data. **GloVe** counted co-occurrences directly; **ELMo** made embeddings context-dependent; **Transformers** replaced the fixed lookup table with attention over the whole sentence. The surface architecture changed, but the lesson from the tiny model above didn't: predict what comes next from context, and similarity **emerges**. The categories were never in the loss function — they were in the statistics all along.

## S1-4 · Memorization vs generalization, and data closes the gap

### The claim

Picture a student cramming for a quiz with only twenty flashcards. They can memorize every card perfectly — but ask a slightly different question on exam day and they're lost. Give them two thousand varied examples instead, and they start to notice the _pattern_ behind the questions. That's the whole story in one analogy.

A big enough neural network is that student with a photographic memory. On a tiny training set it can memorize every point — including **random labels** — while doing no better than chance on held-out data. More data doesn't just mean more examples to average over; it **narrows the set of solutions** that fit. With enough points, memorizing stops being the easy way to drive training loss down, and the model is pushed toward something that actually generalizes.

### The probabilistic view

Underneath the flashcard analogy is a cleaner statement. Every model is really trying to approximate the unknown true data distribution $P_{\text{true}}(y \mid x)$, and the two failure modes are two different things it can land on instead.

**Memorization** collapses the learned distribution onto the training points — it becomes a set of spikes sitting exactly where the data is:

$$
P_{\text{model}}(y \mid x) = \sum_{i=1}^{N} \delta(x - x_i)\,\delta(y - y_i)
$$

That's a lookup table: near-infinite confidence on the examples it has seen, nothing coherent to say about anything else. **Generalization** is the opposite — the model recovers the shape of the distribution rather than the individual points:

$$
P_{\text{model}}(y \mid x) \approx P_{\text{true}}(y \mid x)
$$

There's an information-theoretic tension between the two: to extract broad statistical invariants, a model has to _discard_ information about specific instances. Recent work (for example on diffusion models) frames memorization and generalization as nearly mutually exclusive — memorization shows up as a localized drop in representation dimensionality, while generalization keeps a smooth, higher-dimensional geometry across the whole feature space. That's exactly what the demo makes visible: at small _n_ the boundary spikes around individual points; with more data (or mechanisms like early stopping) it relaxes into the smooth ring the true distribution actually has.

### A concrete example before the demo

Think about the three panels below as the same student, given different amounts of practice:

| Training size | What tends to happen                       | Intuition                                                                         |
| ------------- | ------------------------------------------ | --------------------------------------------------------------------------------- |
| **n = 20**    | Train accuracy → ~100%, test accuracy ~50% | More weights than examples — the net can treat each point as its own special case |
| **n = 200**   | Train high, test noticeably better         | Some structure is cheaper than memorizing, but the fit is still loose             |
| **n = 2000**  | Train and test both climb, gap shrinks     | Fitting thousands of ring points _without_ learning the ring shape is hard        |

The rings aren't a trick. They're a clean stand-in for any real problem where a smooth rule exists but the training set can be small: diagnosing from a handful of scans, recommending from sparse clicks, fine-tuning on a thin slice of documents. The question is always the same: did the model learn the rule, or just the flashcards?

### The setup

Same noisy two-ring classification as before, but now I hold out a fixed test set the model never trains on. The network is deliberately over-parameterized — 64 ReLU units for a problem that could probably be solved with far fewer — so it _can_ memorize if the training set is small enough.

I train three copies of that same architecture, differing only in how many training points each one sees: **20**, **200**, and **2000**. Everything else is identical — same noise level, same held-out test set, same optimizer.

### Watch it happen

Press **Train all three** and watch the three panels side by side. Each one tracks train vs test accuracy and loss as training runs. The number at the bottom of each panel is the **generalization gap** — train accuracy minus test accuracy, in percentage points.

Start with **n = 20**. Watch train accuracy shoot up while test accuracy barely moves. The network is doing the flashcard thing: bending itself around twenty individual points. With 64 hidden units and only 20 examples, it has more than enough freedom to give each training point its own little pocket without discovering that both classes live on rings.

Switch your attention to **n = 2000**. Fitting ~1000 points per class _without_ learning something ring-shaped is much harder. Train and test rise together; the gap collapses. Same architecture, same noise, same test set — you only changed how much evidence the optimizer had to work with.

The summary chart at the bottom plots that gap against training set size. It's the visual version of "data is everything": small _n_, big gap; large _n_, small gap.

<div class="viz-generalization-gap" data-viz="generalization-gap"></div>
<script src="/visualizations/generalization-gap.js"></script>

Try turning up the **label noise** or shuffling a **new data split** — the exact numbers move around, but the shape of the story stays the same. A network with enough capacity will memorize a handful of points if you let it. Feed it enough real examples and memorization stops being the easy path.

### A bit of history, and why this stopped being "just theory"

For decades, statistical learning theory framed generalization as a **bias–variance tradeoff**: simple models underfit (high bias), complex models overfit (high variance). The sweet spot was supposed to be a model just complex enough for the data. That picture is still useful — but deep learning broke it in uncomfortable ways.

In 2017, Zhang et al. ran an experiment that shocked a lot of people: the **same large network that generalizes on real labels can also memorize completely random labels**, achieving zero training error while test error stays at chance. Capacity alone doesn't guarantee generalization; what matters is whether the learned function aligns with structure in the data. The demo above is a gentler, visual cousin of that result — at _n = 20_ you see the memorization side; at _n = 2000_ you see the structure-winning side.

Around the same time, the field was rediscovering an older idea: **more data is a form of regularization**. When ImageNet-scale datasets arrived in the 2010s, the winning strategy wasn't just bigger models — it was bigger models _trained on more examples_. The extra data ruled out most of the wacky memorizing solutions and left the ones that captured real regularities. Modern LLMs push the same lesson to an extreme: trillions of tokens don't just improve fluency, they constrain what the model can plausibly have learned.

There's a wrinkle worth knowing. Classical theory predicted that once a model is complex enough to interpolate the training set, test error should get _worse_. In practice, people sometimes see **double descent** — performance dips near the interpolation threshold, then improves again as capacity grows further. The rings demo doesn't show that second climb (we hold architecture fixed), but it's a reminder that "bigger model = worse generalization" is too simple. The through-line that _does_ hold: **without enough data, a capable model will use its capacity to memorize; with enough data, memorization stops being the path of least resistance.**

That's why practitioners reach for more data before more parameters, and why the generalization gap in the chart above is one of the most honest diagnostics you have: it tells you, in one number, whether your model learned the ring or just the flashcards.

## S1-5 · The hidden symmetries that make training possible

### The claim

Here's something that should feel impossible. Training a neural network means minimizing a **non-convex** loss — a landscape with, in principle, an astronomical number of local minima and saddle points. Classical optimization says you should get hopelessly stuck. And yet gradient descent, the simplest hill-descending algorithm we have, finds excellent solutions almost every time. Why?

A big part of the answer is **symmetry**. A dense layer's neurons have no inherent order, and for ReLU-style activations the weights have a hidden scaling freedom. Those symmetries mean the "good" solutions aren't rare needles in a haystack — they come in enormous families of exact copies, and there are so many equivalent paths downhill that getting permanently stuck becomes very unlikely.

Two symmetries do most of the work:

- **Permutation symmetry.** Relabel the hidden neurons — swap unit 3 with unit 7 everywhere they appear — and the network computes the _exact same function_. For a layer of $h$ neurons there are $h!$ such relabelings, so every solution is secretly $h!$ identical points in weight space.
- **Scale symmetry.** ReLU is positively homogeneous: $\text{ReLU}(cz) = c\,\text{ReLU}(z)$ for any $c > 0$. Multiply a unit's incoming weights by $c$ and its outgoing weight by $1/c$ and the function is untouched — a whole _continuous manifold_ of equivalent solutions.

### The math, briefly

Take one hidden unit $j$ with incoming weights $\mathbf{w}^{\text{in}}_j$, bias $b_j$, and outgoing weight $w^{\text{out}}_j$. Its contribution to the output is

$$
w^{\text{out}}_j \,\cdot\, \text{ReLU}\big(\mathbf{w}^{\text{in}}_j \cdot \mathbf{x} + b_j\big)
$$

**Permutation:** the output is a _sum_ over units, and addition doesn't care about order, so any reordering of the units leaves the sum identical.

**Scaling:** replace $\mathbf{w}^{\text{in}}_j \to c\,\mathbf{w}^{\text{in}}_j$, $b_j \to c\,b_j$, and $w^{\text{out}}_j \to w^{\text{out}}_j / c$. Because ReLU pulls the positive constant straight out,

$$
\frac{w^{\text{out}}_j}{c}\,\text{ReLU}\big(c\,(\mathbf{w}^{\text{in}}_j \cdot \mathbf{x} + b_j)\big) = \frac{w^{\text{out}}_j}{c}\,\cdot\, c\,\text{ReLU}\big(\mathbf{w}^{\text{in}}_j \cdot \mathbf{x} + b_j\big) = w^{\text{out}}_j\,\text{ReLU}\big(\mathbf{w}^{\text{in}}_j \cdot \mathbf{x} + b_j\big)
$$

The $c$ cancels exactly. Same function, different weights.

### See the landscape itself

This is the picture the [video](https://www.youtube.com/watch?v=2qXF8JHcU5E) leans on, so it's worth drawing in 3D. Take the simplest case: a hidden layer of **two** neurons, and summarize each one by a single number — $\theta_1$ for the first neuron, $\theta_2$ for the second. The height of the surface below is the training loss. Because the network only cares about the _set_ of neurons and not their order, the loss obeys $L(\theta_1, \theta_2) = L(\theta_2, \theta_1)$ — it is perfectly symmetric across the dashed diagonal.

That single symmetry is why there are **two** valleys instead of one, and they are exact mirror images. Press **Swap the two neurons** and watch the ball climb over the ridge and settle into the twin valley at _identical_ depth — same loss, same network, just neuron A and neuron B trading name tags. Drag to orbit the surface; the two wells are genuinely the same shape.

<div class="viz-permutation-landscape" data-viz="permutation-landscape"></div>
<script src="/visualizations/permutation-landscape.js"></script>

With only two neurons there are $2! = 2$ copies. Add a third and there are $3! = 6$; a layer of ten gives $10! \approx 3.6$ million valleys, all identical, scattered across weight space. We can only draw two axes, so we draw the two — but every real network's landscape is tiled with these reflections. Far from being a hostile maze of distinct local minima, most of those "minima" are the _same_ minimum seen from a different seat, and that redundancy is exactly what makes the non-convex landscape so easy to descend.

### Watch it happen

I train a small ReLU network on the rings until it separates them, then let you apply these symmetries by hand. The key readout is **function drift** — the largest change in the network's output over the whole plane. If a symmetry is real, that number stays at machine-zero no matter what you do.

- Hit **Permute neuron order**: the coloured neuron chips physically shuffle (colour tracks each unit's identity), but the decision boundary and accuracy don't budge. You've just hopped to a different but _identical_ minimum — one of the $h!$ copies from the landscape above.
- Click a neuron and drag the **scale slider**: its incoming and outgoing weights change in front of you, yet the boundary and drift stay frozen. You're sliding along the continuous scale manifold.

<div class="viz-weight-symmetry" data-viz="weight-symmetry"></div>
<script src="/visualizations/weight-symmetry.js"></script>

Turn up the **hidden units** and watch the "permutation copies" counter explode — $6! = 720$, but $10! = 3{,}628{,}800$. Every one of those is the same network wearing a different name tag.

### A bit of history, and why this matters

For a long time the non-convexity of neural-network losses was treated as a fatal objection. Convex optimization gives you a single global minimum and a guarantee you'll reach it; neural nets throw that away, so why should training ever work? The reframing — nicely laid out in [Hidden Symmetry: Why Deep Learning is Possible](https://www.youtube.com/watch?v=2qXF8JHcU5E) — is that the landscape's structure is not adversarial but **highly symmetric**, and symmetry is a friend to optimization.

The scaling invariance was noticed early: Neyshabur et al. (2015) pointed out that ordinary weight-decay behaves strangely under ReLU rescaling and proposed **Path-SGD**, an optimizer designed to respect the symmetry. The permutation side turned out to have a striking practical consequence. Entezari et al. (2021) conjectured that most SGD solutions are the _same_ solution once you account for permutations, and **Git Re-Basin** (Ainsworth et al., 2022) showed you can often permute one trained network to line up with another so that the straight line between them stays low-loss — evidence that those $h!$ copies really do tile the landscape. This is also why naively averaging two independently trained networks usually fails: they sit in different permutation copies, and you have to re-align them first.

The deeper point ties the whole page together. Symmetry is what lets over-parameterization help instead of hurt (S1-4): extra neurons don't just add capacity, they add more equivalent routes to a good solution. And it's a reminder that a trained network's specific weights are far less meaningful than the _function_ they compute — most of the numbers you'd read off are just an arbitrary choice of name tag and scale.
