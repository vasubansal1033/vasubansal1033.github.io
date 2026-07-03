---
title: Bayes' theorem and the irrational mind
pubDatetime: 2026-07-03T17:00:00Z
featured: true
draft: true
tags:
  - probability
  - bayes
  - behavioural-economics
  - visualizations
description: Bayes' theorem is the mathematically correct way to update beliefs — and Kahneman & Tversky documented every predictable way human intuition gets it wrong.
---

When you get new evidence, you should update your belief in a precise mathematical way: combine what you already believed (your **prior**) with how likely the evidence is under each hypothesis (the **likelihood**). That's **Bayes' theorem** — the rule for thinking under uncertainty.

The uncomfortable part is that humans are very bad at following it. Not randomly bad — _predictably_ bad. Two psychologists spent decades mapping exactly how.

## Who were Kahneman & Tversky?

Daniel Kahneman and Amos Tversky were psychologists who studied how humans make decisions under uncertainty.

Their key finding was that _humans are systematically irrational in predictable ways when dealing with probabilities._

Their work forms the foundation of **behavioural economics.**

- Kahneman won the Nobel Prize for it.
- If you've heard of the book **_Thinking, Fast and Slow_**, that's Kahneman summarising their life's work.

## Connection to Bayes

Bayes' theorem says: when you get new evidence, update your belief by combining your prior with the likelihood of the evidence.

Kahneman and Tversky showed that humans consistently **fail to do this correctly**. Specifically:

- **We ignore priors:** We focus too much on new evidence and forget our prior belief. This is called **base rate neglect.**
- **We are bad at flipping conditionals:** We confuse $P(A \mid B)$ with $P(B \mid A)$.
  - The famous example is confusing "probability of a positive test given you have a disease" with "probability of having the disease given a positive test." These are very different but people treat them as the same.
- **We are overconfident:** We assign probabilities close to 0 or 1 too quickly, before enough evidence.

## The Cab Problem

A city has 85% green cabs and 15% blue cabs. A witness says a cab involved in an accident was blue. Witnesses are 80% accurate.

What's the probability the cab was actually blue?

Most people say around **80%** — the witness said blue and witnesses are 80% accurate.

But Bayes' theorem gives you **~41%**. Why?

Because blue cabs are rare to begin with (15% prior). You need to combine the witness reliability with the base rate of blue cabs. People ignore the 15% prior completely — that's **base rate neglect**.

> Bayes' theorem is the mathematically correct way to update beliefs. Kahneman and Tversky documented all the ways human intuition deviates from it.

Drag the sliders below. Watch how many of the witness's "blue" calls are false positives from the common green cabs — and how far "your gut" (just the witness accuracy) sits from the Bayes answer.

<div class="viz-base-rate-explorer" data-viz="base-rate-explorer"></div>
<script src="/visualizations/base-rate-explorer.js"></script>

### Bangalore context makes it even more intuitive

Think about traffic on Outer Ring Road.

If someone tells you _"there's a traffic jam on ORR right now"_ you'd believe them almost immediately regardless of who said it. Why? Because your prior is already very high. Traffic on ORR is basically guaranteed.

That's Bayes working correctly in your head. When the prior is strong, even weak evidence moves you quickly.

But for rare events (like a bicycle delivery), your prior is low and you need much stronger evidence to be convinced. Your brain forgets this and gets fooled.

## Cricket

### The setup

Say India and Pakistan are playing a World Cup match. Before the match starts you believe: $P(\text{India wins}) = 80\%$ based on historical record, current form, rankings. This is your **prior.**

Now the match starts and after 10 overs, Pakistan is doing surprisingly well, scoring at 9 runs per over.

A cricket analyst says _"based on this power-play, Pakistan is looking strong today"_ and this analyst is right about match outcomes 70% of the time based on power-play performance.

**Most people's intuition:** _"The analyst is 70% accurate and says Pakistan looks strong, so Pakistan probably has a 70% chance of winning."_

**What Bayes says:** You can't just take the analyst's prediction in a vacuum. You need to combine it with your prior: India is historically dominant at 80%.

- Prior for Pakistan winning = 20% (because India wins 80%)
- Analyst says Pakistan looks strong, and analyst is 70% accurate

Bayes gives:

$$
P(\text{Pakistan wins} \mid \text{analyst says strong}) = \frac{0.70 \times 0.20}{0.70 \times 0.20 + 0.30 \times 0.80} = \frac{0.14}{0.38} \approx 37\%
$$

Significantly lower than the 70% the analyst suggested. The strong prior of India being dominant resists the update. Pakistan's good power-play moved the needle from 20% to 37% — meaningful, but not dramatic.

**Now make the prior weaker.** Same scenario but imagine it's India vs New Zealand, two roughly equal teams.

- Prior for New Zealand winning = 50%
- Same analyst, same power-play observation, same 70% accuracy

Now Bayes pushes New Zealand's winning probability to exactly **70%** — the evidence has much more impact because the prior wasn't pulling against it strongly.

**Same evidence, very different update because the prior was different.**

Use the plot below: the curve shows how posterior depends on prior strength for a fixed analyst accuracy. Hit the preset buttons and watch the orange prior dot climb (or fall) to the blue posterior.

<div class="viz-prior-strength-updater" data-viz="prior-strength-updater"></div>
<script src="/visualizations/prior-strength-updater.js"></script>

### The quote, in cricket terms

- _"New evidence doesn't determine your beliefs in a vacuum."_
  - Pakistan's good power-play alone cannot tell you who wins. You must factor in India's historical dominance.
- _"It only updates prior belief."_
  - The power-play performance takes your prior (India 80% favourite) and nudges it toward Pakistan — but only nudges, not overturns.

## Does Bayes put a constraint of independence on events?

Technically, each cricket match _is_ an independent event. Pakistan beating India in one match does not change the laws of cricket for the next match. In that strict sense, priors based on historical performance shouldn't influence a _specific_ match's outcome.

**So what's actually going on?**

Bayes' theorem isn't saying the _outcome_ of one match influences another. It's saying your **belief about the teams' abilities** should be informed by historical data.

These are two different things:

- **Event independence:** Match outcomes don't cause each other.
- **Parameter estimation:** Your estimate of "how good is India?" should use all available data.

When you set a prior of 80% for India winning, you're not saying past matches influence this match. You're saying _"based on everything I know about India's skill level, I estimate they win 80% of the time against this opposition."_ That skill level is a real thing that persists across matches.

**A cleaner way to think about it:** The prior isn't about past match outcomes — it's about your **estimate of an underlying truth.** The underlying truth = India's relative skill level compared to Pakistan. That's fairly stable and real. Historical matches are just evidence that help you estimate it. So Bayes is saying: use historical data to estimate the teams' true skill levels, then use that estimate as your prior for any specific match.

### The Gambler's Fallacy comparison

This is worth contrasting because it's a common confusion:

- **Gambler's Fallacy:** _"Pakistan lost the last 5 matches against India so they're DUE for a win."_ This wrongly treats independent events as dependent.
- **Bayesian thinking:** _"Pakistan has lost 70% of matches against India historically, so my prior for this match is India wins 70%."_ This correctly uses history to estimate underlying ability.

The difference is subtle but crucial. One is about outcomes causing outcomes; the other is about using data to estimate a stable underlying parameter.

Press **Play** below. After a losing streak, the gambler's "due-o-meter" shoots up — but each match is still independent. The Bayesian line quietly tracks true skill.

<div class="viz-gambler-vs-bayes" data-viz="gambler-vs-bayes"></div>
<script src="/visualizations/gambler-vs-bayes.js"></script>

### Where it gets genuinely complicated

Your instinct is pointing at something real though. Sometimes priors _can_ be wrong or misleading:

- Pakistan has a new coach and completely revamped batting lineup
- India's key bowlers are injured
- It's a day-night match and conditions heavily favour Pakistan

_In these cases blindly applying a historical prior is genuinely wrong_, because the underlying parameter (relative skill) has actually changed.

Good Bayesian thinking means updating your prior itself when you have strong reason to believe the situation has fundamentally changed. This is actually a deep problem in Bayesian statistics: how do you choose a good prior? It's called **prior sensitivity**, and researchers think about it a lot.

## Further watching

- [Bayes' theorem explained](https://www.youtube.com/watch?v=HZGCoVF3YvM)
- [More on probability and updating beliefs](https://www.youtube.com/watch?v=gE6RnZJixUw&list=PLMrJAkhIeNNR3sNYvfgiKgcStwuPSts9V&index=22)
