/**
 * tests/metrics.test.js — pplx-eval-harness
 */

import {
  precisionAtK, recallAtK, f1AtK, ndcgAtK, mrr,
  rougeL, citationPrecision, citationRecall, aggregateScores
} from "../src/metrics/index.js";

import { heuristicJudge } from "../src/judges/llm_judge.js";

// ── Precision@K ───────────────────────────────────────────────────────────────

test("precisionAtK: all relevant → 1.0", () => {
  const retrieved = ["a", "b", "c"];
  const relevant = new Set(["a", "b", "c", "d"]);
  expect(precisionAtK(retrieved, relevant, 3)).toBe(1.0);
});

test("precisionAtK: none relevant → 0.0", () => {
  const retrieved = ["x", "y", "z"];
  const relevant = new Set(["a", "b"]);
  expect(precisionAtK(retrieved, relevant, 3)).toBe(0.0);
});

test("precisionAtK: 2 of 4 relevant → 0.5", () => {
  const retrieved = ["a", "x", "b", "y"];
  const relevant = new Set(["a", "b"]);
  expect(precisionAtK(retrieved, relevant, 4)).toBe(0.5);
});

test("precisionAtK: K truncates results", () => {
  const retrieved = ["a", "b", "x", "y"];
  const relevant = new Set(["a", "b"]);
  // P@2 should be 1.0, P@4 should be 0.5
  expect(precisionAtK(retrieved, relevant, 2)).toBe(1.0);
  expect(precisionAtK(retrieved, relevant, 4)).toBe(0.5);
});

// ── Recall@K ──────────────────────────────────────────────────────────────────

test("recallAtK: all relevant in top-K → 1.0", () => {
  const retrieved = ["a", "b"];
  const relevant = new Set(["a", "b"]);
  expect(recallAtK(retrieved, relevant, 2)).toBe(1.0);
});

test("recallAtK: none retrieved → 0.0", () => {
  const retrieved = ["x", "y"];
  const relevant = new Set(["a", "b"]);
  expect(recallAtK(retrieved, relevant, 2)).toBe(0.0);
});

test("recallAtK: empty relevant set → 0", () => {
  expect(recallAtK(["a", "b"], new Set(), 2)).toBe(0);
});

// ── F1@K ─────────────────────────────────────────────────────────────────────

test("f1AtK: perfect P and R → 1.0", () => {
  const retrieved = ["a", "b"];
  const relevant = new Set(["a", "b"]);
  expect(f1AtK(retrieved, relevant, 2)).toBe(1.0);
});

test("f1AtK: zero P or R → 0", () => {
  expect(f1AtK(["x"], new Set(["a"]), 1)).toBe(0);
});

// ── NDCG@K ────────────────────────────────────────────────────────────────────

test("ndcgAtK: perfect ranking → 1.0", () => {
  const retrieved = ["a", "b", "c"];
  const relevant = new Set(["a", "b", "c"]);
  expect(ndcgAtK(retrieved, relevant, 3)).toBe(1.0);
});

test("ndcgAtK: all irrelevant → 0.0", () => {
  expect(ndcgAtK(["x", "y", "z"], new Set(["a"]), 3)).toBe(0.0);
});

test("ndcgAtK: relevant at position 2 < relevant at position 1", () => {
  const relevant = new Set(["a"]);
  const firstPos = ndcgAtK(["a", "x"], relevant, 2);
  const secondPos = ndcgAtK(["x", "a"], relevant, 2);
  expect(firstPos).toBeGreaterThan(secondPos);
});

test("ndcgAtK: no relevant items in corpus → 0", () => {
  expect(ndcgAtK(["a", "b"], new Set(), 2)).toBe(0);
});

// ── MRR ───────────────────────────────────────────────────────────────────────

test("mrr: first result always relevant → 1.0", () => {
  const queries = [
    { retrieved: ["a", "b"], relevant: new Set(["a"]) },
    { retrieved: ["c", "d"], relevant: new Set(["c"]) },
  ];
  expect(mrr(queries)).toBe(1.0);
});

test("mrr: relevant at rank 2 → 0.5", () => {
  const queries = [{ retrieved: ["x", "a"], relevant: new Set(["a"]) }];
  expect(mrr(queries)).toBe(0.5);
});

test("mrr: no relevant found → 0.0", () => {
  const queries = [{ retrieved: ["x", "y"], relevant: new Set(["z"]) }];
  expect(mrr(queries)).toBe(0.0);
});

test("mrr: empty queries → 0.0", () => {
  expect(mrr([])).toBe(0.0);
});

// ── ROUGE-L ───────────────────────────────────────────────────────────────────

test("rougeL: identical strings → 1.0", () => {
  const text = "the quick brown fox jumps over the lazy dog";
  expect(rougeL(text, text)).toBe(1.0);
});

test("rougeL: completely different → low score", () => {
  const hyp = "apple banana cherry";
  const ref = "delta epsilon zeta";
  expect(rougeL(hyp, ref)).toBe(0.0);
});

test("rougeL: partial overlap", () => {
  const hyp = "the quick brown fox";
  const ref = "the slow brown cat";
  const score = rougeL(hyp, ref);
  expect(score).toBeGreaterThan(0);
  expect(score).toBeLessThan(1);
});

test("rougeL: empty hypothesis → 0", () => {
  expect(rougeL("", "reference text")).toBe(0);
});

test("rougeL: empty reference → 0", () => {
  expect(rougeL("hypothesis text", "")).toBe(0);
});

test("rougeL: symmetric-ish (not exact due to P/R)", () => {
  const a = "the cat sat on the mat";
  const b = "the mat sat on the cat";
  // Should be non-zero both ways
  expect(rougeL(a, b)).toBeGreaterThan(0);
  expect(rougeL(b, a)).toBeGreaterThan(0);
});

// ── Citation precision/recall ─────────────────────────────────────────────────

test("citationPrecision: all cited are relevant → 1.0", () => {
  expect(citationPrecision([1, 2, 3], new Set([1, 2, 3, 4]))).toBe(1.0);
});

test("citationPrecision: none cited are relevant → 0.0", () => {
  expect(citationPrecision([5, 6], new Set([1, 2]))).toBe(0.0);
});

test("citationPrecision: empty citations → 0", () => {
  expect(citationPrecision([], new Set([1, 2]))).toBe(0);
});

test("citationRecall: all relevant cited → 1.0", () => {
  expect(citationRecall([1, 2, 3], new Set([1, 2]))).toBe(1.0);
});

test("citationRecall: no relevant cited → 0.0", () => {
  expect(citationRecall([3, 4], new Set([1, 2]))).toBe(0.0);
});

test("citationRecall: empty relevant set → 1.0 (vacuously true)", () => {
  expect(citationRecall([1, 2], new Set())).toBe(1.0);
});

// ── aggregateScores ───────────────────────────────────────────────────────────

test("aggregateScores: single value", () => {
  const r = aggregateScores([3.5]);
  expect(r.mean).toBe(3.5);
  expect(r.std).toBe(0);
  expect(r.min).toBe(3.5);
  expect(r.max).toBe(3.5);
});

test("aggregateScores: known values", () => {
  const r = aggregateScores([1, 2, 3, 4, 5]);
  expect(r.mean).toBe(3.0);
  expect(r.min).toBe(1);
  expect(r.max).toBe(5);
  expect(r.p50).toBe(3);
});

test("aggregateScores: empty array → zeros", () => {
  const r = aggregateScores([]);
  expect(r.mean).toBe(0);
});

// ── Heuristic judge ───────────────────────────────────────────────────────────

const SOURCES = [
  { id: "s1", text: "Source one content here." },
  { id: "s2", text: "Source two content here." },
];

test("heuristicJudge: well-cited answer scores well", () => {
  const answer =
    "The first fact is true [[1]]. The second fact is also supported by research [[2]]. " +
    "Together they demonstrate that this topic is well documented [[1]] across multiple sources.";
  const r = heuristicJudge(answer, SOURCES);
  expect(r.heuristicScore).toBeGreaterThan(0.5);
  expect(r.citationsInRange).toBe(true);
});

test("heuristicJudge: out-of-range citation detected", () => {
  const answer = "This claims something [[99]] that is out of range.";
  const r = heuristicJudge(answer, SOURCES);
  expect(r.citationsInRange).toBe(false);
});

test("heuristicJudge: no citations → coverage=0", () => {
  const answer =
    "This is a long answer with multiple sentences. Each one makes claims. But none are cited.";
  const r = heuristicJudge(answer, SOURCES);
  expect(r.citationCoverage).toBe(0);
});

test("heuristicJudge: returns expected fields", () => {
  const r = heuristicJudge("Some answer [[1]].", SOURCES);
  expect(r).toHaveProperty("wordCount");
  expect(r).toHaveProperty("citationCount");
  expect(r).toHaveProperty("citationCoverage");
  expect(r).toHaveProperty("lengthOk");
  expect(r).toHaveProperty("citationsInRange");
  expect(r).toHaveProperty("heuristicScore");
});
