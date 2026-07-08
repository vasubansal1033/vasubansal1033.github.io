/**
 * Client-side merged-vocab encoder mirroring MergedTokenizer.encode.
 * Keep in sync with: core/dp_segment.py, core/viterbi.py, core/resolve.py, core/merged_encode.py
 */
(function () {
  const BYTE_TOKEN_PREFIX = "<0x";
  const OOV_GRAPHEME_COST = 12.0;
  const globalObject = typeof window !== "undefined" ? window : globalThis;

  function graphemeClusters(text) {
    const segmenter = new Intl.Segmenter("und", { granularity: "grapheme" });
    return [...segmenter.segment(text)].map(part => part.segment);
  }

  function byteToken(byteValue) {
    return `${BYTE_TOKEN_PREFIX}${byteValue.toString(16).toUpperCase().padStart(2, "0")}>`;
  }

  function encodeTextAsByteTokens(text) {
    const encoder = new TextEncoder();
    return [...encoder.encode(text)].map(byteValue => byteToken(byteValue));
  }

  function compilePretokenPattern(pattern) {
    const jsPattern = String(pattern).replace(/\(\?i:/g, "(?:");
    return new RegExp(jsPattern, "giu");
  }

  function isTeluguText(text) {
    if (!text) return false;
    for (const char of text) {
      const code = char.codePointAt(0);
      if (code < 0x0c00 || code > 0x0c7f) return false;
    }
    return true;
  }

  function isLatinText(text) {
    if (!text) return false;
    const indicRanges = [
      [0x0900, 0x097f],
      [0x0a00, 0x0a7f],
      [0x0c00, 0x0c7f],
    ];
    for (const char of text) {
      const code = char.codePointAt(0);
      for (const [start, end] of indicRanges) {
        if (code >= start && code <= end) return false;
      }
    }
    return true;
  }

  function segmentPieceDp(piece, tokenToId) {
    if (Object.prototype.hasOwnProperty.call(tokenToId, piece)) {
      return [piece];
    }

    const graphemes = graphemeClusters(piece);
    const length = graphemes.length;
    if (length === 0) return [];

    const bestCost = new Array(length + 1).fill(Number.POSITIVE_INFINITY);
    const backptr = new Array(length + 1).fill(null);
    bestCost[0] = 0;

    for (let start = 0; start < length; start += 1) {
      if (!Number.isFinite(bestCost[start])) continue;
      for (let end = start + 1; end <= length; end += 1) {
        const candidate = graphemes.slice(start, end).join("");
        if (Object.prototype.hasOwnProperty.call(tokenToId, candidate)) {
          const cost = bestCost[start] + 1;
          if (cost < bestCost[end]) {
            bestCost[end] = cost;
            backptr[end] = [start, candidate];
          }
        } else if (end === start + 1) {
          const single = graphemes[start];
          const cost = bestCost[start] + 1;
          if (cost < bestCost[end]) {
            bestCost[end] = cost;
            backptr[end] = [start, single];
          }
        }
      }
    }

    if (!Number.isFinite(bestCost[length])) return graphemes;

    const segments = [];
    let index = length;
    while (index > 0) {
      const step = backptr[index];
      if (!step) return graphemes;
      segments.push(step[1]);
      index = step[0];
    }
    segments.reverse();
    return segments;
  }

  function segmentPieceByVocab(piece, tokenToId, spmScores) {
    if (Object.prototype.hasOwnProperty.call(tokenToId, piece)) {
      return [piece];
    }

    if (spmScores && isTeluguText(piece)) {
      const segments = segmentPieceViterbi(piece, tokenToId, spmScores);
      if (segments.length > 0) return segments;
    }

    if (isLatinText(piece)) {
      return segmentPieceDp(piece, tokenToId);
    }

    return segmentPieceGreedy(piece, tokenToId);
  }

  function segmentPieceGreedy(piece, tokenToId) {
    if (Object.prototype.hasOwnProperty.call(tokenToId, piece)) {
      return [piece];
    }

    const graphemes = graphemeClusters(piece);
    const segments = [];
    let index = 0;
    while (index < graphemes.length) {
      let matched = false;
      for (let end = graphemes.length; end > index; end -= 1) {
        const candidate = graphemes.slice(index, end).join("");
        if (!Object.prototype.hasOwnProperty.call(tokenToId, candidate)) {
          continue;
        }

        segments.push(candidate);
        index = end;
        matched = true;
        break;
      }
      if (!matched) {
        segments.push(graphemes[index]);
        index += 1;
      }
    }
    return segments;
  }

  function segmentPieceViterbi(piece, tokenToId, spmScores) {
    const spmSegments = viterbiSegments(piece, spmScores);
    if (spmSegments.length === 0) return [];

    const mergedSegments = [];
    for (const segment of spmSegments) {
      if (Object.prototype.hasOwnProperty.call(tokenToId, segment)) {
        mergedSegments.push(segment);
        continue;
      }
      mergedSegments.push(...segmentPieceGreedy(segment, tokenToId));
    }
    return mergedSegments;
  }

  function viterbiSegments(piece, spmScores) {
    const units = [...piece];
    const length = units.length;
    if (length === 0) return [];
    if (length === 1) {
      const single = units[0];
      return Object.prototype.hasOwnProperty.call(spmScores, single)
        ? [single]
        : [single];
    }

    const bestCost = new Array(length + 1).fill(Number.POSITIVE_INFINITY);
    const backptr = new Array(length + 1).fill(null);
    bestCost[0] = 0;

    for (let start = 0; start < length; start += 1) {
      if (!Number.isFinite(bestCost[start])) continue;

      let hasScoredExtension = false;
      for (let end = start + 1; end <= length; end += 1) {
        const candidate = units.slice(start, end).join("");
        if (!Object.prototype.hasOwnProperty.call(spmScores, candidate))
          continue;
        hasScoredExtension = true;
        const cost = bestCost[start] - spmScores[candidate];
        if (cost < bestCost[end]) {
          bestCost[end] = cost;
          backptr[end] = [start, candidate];
        }
      }

      if (!hasScoredExtension) {
        const single = units[start];
        const byteCount = Math.max(encodeTextAsByteTokens(single).length, 1);
        const cost = bestCost[start] + OOV_GRAPHEME_COST * byteCount;
        if (cost < bestCost[start + 1]) {
          bestCost[start + 1] = cost;
          backptr[start + 1] = [start, single];
        }
      }
    }

    if (!Number.isFinite(bestCost[length])) return [];

    const segments = [];
    let index = length;
    while (index > 0) {
      const step = backptr[index];
      if (!step) return [];
      segments.push(step[1]);
      index = step[0];
    }
    segments.reverse();
    return segments;
  }

  function resolvePieceSpans(piece, pieceStart, tokenToId, spmScores) {
    const spans = [];
    let localOffset = 0;

    for (const segment of segmentPieceByVocab(piece, tokenToId, spmScores)) {
      const localStart = piece.indexOf(segment, localOffset);
      const segLocalStart = localStart >= 0 ? localStart : localOffset;
      const absStart = pieceStart + segLocalStart;
      const absEnd = absStart + segment.length;

      if (Object.prototype.hasOwnProperty.call(tokenToId, segment)) {
        spans.push({
          id: tokenToId[segment],
          token: segment,
          start: absStart,
          end: absEnd,
          synthetic: false,
        });
      } else {
        for (const byteTokenValue of encodeTextAsByteTokens(segment)) {
          if (Object.prototype.hasOwnProperty.call(tokenToId, byteTokenValue)) {
            spans.push({
              id: tokenToId[byteTokenValue],
              token: byteTokenValue,
              start: absStart,
              end: absEnd,
              synthetic: false,
            });
          }
        }
      }
      localOffset = segLocalStart + segment.length;
    }
    return spans;
  }

  function resolvePieceTokens(piece, tokenToId, spmScores) {
    const tokens = [];
    for (const segment of segmentPieceByVocab(piece, tokenToId, spmScores)) {
      if (Object.prototype.hasOwnProperty.call(tokenToId, segment)) {
        tokens.push({ id: tokenToId[segment], token: segment });
        continue;
      }
      for (const byteTokenValue of encodeTextAsByteTokens(segment)) {
        if (Object.prototype.hasOwnProperty.call(tokenToId, byteTokenValue)) {
          tokens.push({ id: tokenToId[byteTokenValue], token: byteTokenValue });
        }
      }
    }
    return tokens;
  }

  function pretokenize(text, pattern) {
    const re = compilePretokenPattern(pattern);
    const matches = [];
    let match;
    while ((match = re.exec(text)) !== null) {
      matches.push({
        text: match[0],
        start: match.index,
        end: match.index + match[0].length,
      });
    }
    return matches;
  }

  function encodeGapSpans(text, start, end, tokenToId, spmScores) {
    if (start >= end) return [];
    const spans = [];
    const gapText = text.slice(start, end);
    let offset = start;
    for (const grapheme of graphemeClusters(gapText)) {
      spans.push(...resolvePieceSpans(grapheme, offset, tokenToId, spmScores));
      offset += grapheme.length;
    }
    return spans;
  }

  function encodeMergedWithSpans(text, runtime) {
    const tokenToId = runtime.token_to_id || {};
    const spmScores = runtime.te_spm_scores || null;
    const pattern = runtime.pretoken_pattern;
    if (!pattern) return [];

    const re = compilePretokenPattern(pattern);
    const spans = [];
    let lastEnd = 0;
    let match;

    while ((match = re.exec(text)) !== null) {
      const matchStart = match.index;
      const matchEnd = match.index + match[0].length;
      spans.push(
        ...encodeGapSpans(text, lastEnd, matchStart, tokenToId, spmScores)
      );
      spans.push(
        ...resolvePieceSpans(match[0], matchStart, tokenToId, spmScores)
      );
      lastEnd = matchEnd;
    }

    spans.push(
      ...encodeGapSpans(text, lastEnd, text.length, tokenToId, spmScores)
    );
    return spans;
  }

  function displaySpans(text, runtime) {
    const raw = encodeWithSpans(text, runtime).filter(span => !span.synthetic);
    const display = [];
    let index = 0;

    while (index < raw.length) {
      const span = raw[index];
      const isByteToken = String(span.token).startsWith(BYTE_TOKEN_PREFIX);

      if (!isByteToken) {
        display.push(span);
        index += 1;
        continue;
      }

      const group = [span];
      let next = index + 1;
      while (
        next < raw.length &&
        String(raw[next].token).startsWith(BYTE_TOKEN_PREFIX) &&
        raw[next].start === span.start &&
        raw[next].end === span.end
      ) {
        group.push(raw[next]);
        next += 1;
      }

      if (group.length === 1) {
        display.push(span);
      } else {
        display.push({
          id: group[0].id,
          token: text.slice(span.start, span.end),
          ids: group.map(item => item.id),
          tokens: group.map(item => item.token),
          start: span.start,
          end: span.end,
          synthetic: false,
          collapsed: true,
        });
      }
      index = next;
    }

    return display;
  }

  function encodeWithSpans(text, runtime) {
    if (!runtime || !runtime.token_to_id) return [];
    return encodeMergedWithSpans(text, runtime);
  }

  function encodeIds(text, runtime) {
    return encodeWithSpans(text, runtime).map(span => span.id);
  }

  function countPretokenWords(text, pattern) {
    return pretokenize(text, pattern).length;
  }

  function countFertilityTokens(text, runtime) {
    if (!runtime || !runtime.token_to_id) return 0;
    const tokenToId = runtime.token_to_id;
    const spmScores = runtime.te_spm_scores || null;
    const pattern = runtime.pretoken_pattern;
    const pretokens = pretokenize(text, pattern);
    let count = 0;

    for (const pretoken of pretokens) {
      count += resolvePieceTokens(pretoken.text, tokenToId, spmScores).length;
    }
    return count;
  }

  globalObject.S2Encode = {
    pretokenize,
    graphemeClusters,
    resolvePieceTokens,
    resolvePieceSpans,
    encodeWithSpans,
    displaySpans,
    encodeIds,
    countPretokenWords,
    countFertilityTokens,
  };
})();
