---
title: Multilingual tokenizer fertility
pubDatetime: 2026-07-08T12:45:00Z
featured: false
draft: false
tags:
  - tokenizers
  - nlp
  - multilingual
  - visualizations
description: Explore a merged English–Hindi–Punjabi–Telugu tokenizer — fertility per language, spread score, and a live encode playground.
---

A naive BPE (Byte Pair Encoding) tokenizer works well with English over Indic text and it all comes down to how these characters are represented in memory. English characters fortunately use a single byte (the ASCII value) for their representation which works with pair-counting strategy used in naive BPE.
However, this is not the case for other languages. For e.g., Indic languages like Hindi (Devanagari), Punjabi (Gurmukhi), and Telugu sit within the `U+0800` to `U+FFFF` range of Unicode. Thus, every native letter, vowel sign, and modifying mark takes exactly **3 bytes** in UTF-8. Adding to this, Indic scripts do not follow a simple, linear left-to-right letter sequence.

They are <i>abugidas</i> where vowel signs (matras) and conjunctions (halants) physically stack onto a base consonant. For e.g., the syllable **कि** consists of the base consonant **`क`** (`E0 A4 95`) followed by the dependent vowel sign **`ि`** (`E0 A4 9F`).
To the human eye, it is 1 unit. But in memory, it is 2\*3 = **6 raw bytes**.

We also have consonant clusters (half-letters) where to form a joint letter sound like **क्या**, an invisible character called a _Halant_ ( `्`) _/ Virama_ (`U+094D` or `E0 A5 8D`) is inserted to strip away the inherent vowel sound of the first letter.
This clusters multiple characters together, quickly scaling up the byte footprint to **12 bytes** or more for a single visual syllable.

That imbalance shows up as **fertility**: tokens used to represent a word. Languages with higher fertility burn more context window for the same amount of meaning, so a fair multilingual vocab should keep fertilities close for all languages, not just minimize English alone.

I merged a small BPE vocabulary across English, Hindi, Punjabi, and Telugu (~10k tokens), scored each language on India's Wikipedia page in each language, then ranked them by fertility. The evaluation score is the reciprocal of the fertility **spread** (`X_max − X_min`) — tighter balance, higher score.

The widget below is a visualization of the trained tokenizer: language cards with pass/fail thresholds, a tokenize playground, and the final merged vocab.

<div
  class="viz-s2-tokenizer-scorer"
  data-viz="s2-tokenizer-scorer"
  data-data-base="/visualizations/tokenizer-visualization/public/data"
></div>
<script src="/visualizations/tokenizer-visualization/public/visualizations/s2-encode.js"></script>
<script src="/visualizations/tokenizer-visualization/public/visualizations/s2-tokenizer-scorer.js"></script>

Try your own sentences in the playground, or filter the vocabulary table by language to see how the merge allocated capacity.
