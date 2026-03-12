/**
 * metrics/index.js
 *
 * Retrieval and answer quality metrics.
 *
 * Retrieval metrics:
 *   - Precision@K, Recall@K, F1@K
 *   - NDCG@K (Normalized Discounted Cumulative Gain)
 *   - MRR (Mean Reciprocal Rank)
 *
 * Answer metrics:
 *   - ROUGE-L (longest common subsequence F1)
 *   - Citation precision/recall
 *   - Hallucination rate (heuristic)
 */

// ─── Retrieval metrics ────────────────────────────────────────────────────────

/**
 * Precision@K: fraction of top-K results that are relevant.
 * @param {string[]} retrieved   - ordered list of doc IDs
 * @param {Set<string>} relevant - set of relevant doc IDs
 * @param {number} k
 */
export function precisionAtK(retrieved, relevant, k) {
  const topK = retrieved.slice(0, k);
  const hits = topK.filter((id) => relevant.has(id)).length;
  return hits / k;
}

/**
 * Recall@K: fraction of relevant docs found in top-K.
 */
export function recallAtK(retrieved, relevant, k) {
  if (relevant.size === 0) return 0;
  const topK = retrieved.slice(0, k);
  const hits = topK.filter((id) => relevant.has(id)).length;
  return hits / relevant.size;
}

/**
 * F1@K: harmonic mean of Precision@K and Recall@K.
 */
export function f1AtK(retrieved, relevant, k) {
  const p = precisionAtK(retrieved, relevant, k);
  const r = recallAtK(retrieved, relevant, k);
  if (p + r === 0) return 0;
  return (2 * p * r) / (p + r);
}

/**
 * NDCG@K: Normalized Discounted Cumulative Gain.
 * Rewards relevant results appearing earlier in the list.
 * Uses binary relevance (relevant=1, not=0).
 *
 * @param {string[]} retrieved
 * @param {Set<string>} relevant
 * @param {number} k
 */
export function ndcgAtK(retrieved, relevant, k) {
  const topK = retrieved.slice(0, k);

  // DCG
  let dcg = 0;
  for (let i = 0; i < topK.length; i++) {
    if (relevant.has(topK[i])) {
      dcg += 1 / Math.log2(i + 2); // +2 because log2(1)=0
    }
  }

  // Ideal DCG: relevant docs placed at top positions
  let idcg = 0;
  const nRelevant = Math.min(relevant.size, k);
  for (let i = 0; i < nRelevant; i++) {
    idcg += 1 / Math.log2(i + 2);
  }

  return idcg === 0 ? 0 : dcg / idcg;
}

/**
 * MRR: Mean Reciprocal Rank.
 * Average of 1/rank of first relevant result per query.
 *
 * @param {Array<{retrieved: string[], relevant: Set<string>}>} queryResults
 */
export function mrr(queryResults) {
  let total = 0;
  for (const { retrieved, relevant } of queryResults) {
    const rank = retrieved.findIndex((id) => relevant.has(id));
    if (rank >= 0) total += 1 / (rank + 1);
  }
  return total / Math.max(queryResults.length, 1);
}

// ─── Answer metrics ───────────────────────────────────────────────────────────

/**
 * ROUGE-L: F1 based on Longest Common Subsequence.
 * @param {string} hypothesis - generated answer
 * @param {string} reference  - gold answer
 * @returns {number} F1 score in [0,1]
 */
export function rougeL(hypothesis, reference) {
  const hTokens = hypothesis.toLowerCase().split(/\s+/).filter(Boolean);
  const rTokens = reference.toLowerCase().split(/\s+/).filter(Boolean);

  if (!hTokens.length || !rTokens.length) return 0;

  // LCS length via DP
  const m = hTokens.length;
  const n = rTokens.length;
  const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      dp[i][j] =
        hTokens[i - 1] === rTokens[j - 1]
          ? dp[i - 1][j - 1] + 1
          : Math.max(dp[i - 1][j], dp[i][j - 1]);
    }
  }

  const lcs = dp[m][n];
  const precision = lcs / m;
  const recall = lcs / n;
  if (precision + recall === 0) return 0;
  return parseFloat(((2 * precision * recall) / (precision + recall)).toFixed(4));
}

/**
 * Citation precision: fraction of used citations that are relevant.
 * @param {number[]} usedCitations    - 1-based indices cited in answer
 * @param {Set<number>} relevantCitations - 1-based indices of actually relevant sources
 */
export function citationPrecision(usedCitations, relevantCitations) {
  if (!usedCitations.length) return 0;
  const correct = usedCitations.filter((c) => relevantCitations.has(c)).length;
  return correct / usedCitations.length;
}

/**
 * Citation recall: fraction of relevant sources that were cited.
 */
export function citationRecall(usedCitations, relevantCitations) {
  if (!relevantCitations.size) return 1; // vacuously perfect
  const usedSet = new Set(usedCitations);
  const found = [...relevantCitations].filter((c) => usedSet.has(c)).length;
  return found / relevantCitations.size;
}

/**
 * Aggregate metrics across multiple eval examples.
 * @param {number[]} scores  - per-example scores in [0,1] or [1,5]
 */
export function aggregateScores(scores) {
  if (!scores.length) return { mean: 0, std: 0, min: 0, max: 0, p25: 0, p50: 0, p75: 0 };
  const sorted = [...scores].sort((a, b) => a - b);
  const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
  const variance = scores.reduce((a, b) => a + (b - mean) ** 2, 0) / scores.length;
  const pct = (p) => sorted[Math.floor((p / 100) * sorted.length)];
  return {
    mean: parseFloat(mean.toFixed(3)),
    std: parseFloat(Math.sqrt(variance).toFixed(3)),
    min: sorted[0],
    max: sorted[sorted.length - 1],
    p25: pct(25),
    p50: pct(50),
    p75: pct(75),
  };
}
