/**
 * judges/llm_judge.js
 *
 * LLM-as-judge evaluation for answer quality.
 *
 * Evaluates answers on five dimensions:
 *   1. Factual Accuracy   — are the claims correct?
 *   2. Citation Grounding — do citations support the claims?
 *   3. Completeness       — does it answer the full question?
 *   4. Conciseness        — is there unnecessary verbosity?
 *   5. Coherence          — is it logically structured and readable?
 *
 * Each dimension: 1-5 scale.
 * Overall score: weighted average.
 *
 * Weights (tuned on human preference data):
 *   factual_accuracy:    0.35
 *   citation_grounding:  0.25
 *   completeness:        0.20
 *   conciseness:         0.10
 *   coherence:           0.10
 */

import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

const WEIGHTS = {
  factual_accuracy: 0.35,
  citation_grounding: 0.25,
  completeness: 0.20,
  conciseness: 0.10,
  coherence: 0.10,
};

const JUDGE_SYSTEM = `You are an expert evaluator for AI-generated search answers.
Your job is to rigorously assess answer quality on specific dimensions.
Be critical. Distinguish between excellent (5), good (4), adequate (3), poor (2), and unacceptable (1) answers.
Output ONLY valid JSON, no preamble or explanation.`;

function buildJudgePrompt(question, answer, sources) {
  const sourceBlock = sources
    .map((s, i) => `[${i + 1}] ${s.title ?? s.id}: ${s.text.slice(0, 200)}`)
    .join("\n");

  return `QUESTION: ${question}

ANSWER TO EVALUATE:
${answer}

SOURCES AVAILABLE TO THE SYSTEM:
${sourceBlock}

Evaluate the answer on these dimensions. Score 1-5 for each.

Return ONLY this JSON:
{
  "factual_accuracy": { "score": 1-5, "reasoning": "brief explanation" },
  "citation_grounding": { "score": 1-5, "reasoning": "brief explanation" },
  "completeness": { "score": 1-5, "reasoning": "brief explanation" },
  "conciseness": { "score": 1-5, "reasoning": "brief explanation" },
  "coherence": { "score": 1-5, "reasoning": "brief explanation" },
  "critical_issues": ["list any critical errors or hallucinations"]
}`;
}

/**
 * Judge an answer using an LLM.
 *
 * @param {Object} params
 * @param {string} params.question
 * @param {string} params.answer
 * @param {Array}  params.sources
 * @param {string} [params.model]
 * @returns {Promise<EvalResult>}
 */
export async function judgeAnswer({ question, answer, sources, model = "claude-sonnet-4-6" }) {
  const prompt = buildJudgePrompt(question, answer, sources);

  const response = await client.messages.create({
    model,
    max_tokens: 1024,
    system: JUDGE_SYSTEM,
    messages: [{ role: "user", content: prompt }],
  });

  let raw;
  try {
    raw = JSON.parse(response.content[0].text);
  } catch {
    throw new Error(`Judge returned invalid JSON: ${response.content[0].text.slice(0, 100)}`);
  }

  // Compute weighted overall score
  let overall = 0;
  for (const [dim, weight] of Object.entries(WEIGHTS)) {
    overall += (raw[dim]?.score ?? 0) * weight;
  }

  return {
    question,
    scores: {
      factual_accuracy: raw.factual_accuracy,
      citation_grounding: raw.citation_grounding,
      completeness: raw.completeness,
      conciseness: raw.conciseness,
      coherence: raw.coherence,
    },
    overall: parseFloat(overall.toFixed(2)),
    criticalIssues: raw.critical_issues ?? [],
    usage: response.usage,
  };
}

/**
 * Quick heuristic judge (no API call) — useful for fast pre-screening.
 * Scores based on observable signals without semantic understanding.
 *
 * @param {string} answer
 * @param {Array}  sources
 * @returns {HeuristicResult}
 */
export function heuristicJudge(answer, sources) {
  const wordCount = answer.split(/\s+/).length;
  const citationCount = (answer.match(/\[\[\d+\]\]/g) ?? []).length;
  const sentences = answer.split(/[.!?]+/).filter((s) => s.trim().length > 10);
  const citedSentences = sentences.filter((s) => /\[\[\d+\]\]/.test(s)).length;

  // Citation coverage
  const citationCoverage = sentences.length
    ? citedSentences / sentences.length
    : 0;

  // Length penalty: too short (<50w) or too long (>600w)
  const lengthOk = wordCount >= 50 && wordCount <= 600;

  // Out-of-range citation check
  const maxCitationIdx = Math.max(
    ...(answer.match(/\[\[(\d+)\]\]/g) ?? ["[[0]]"]).map((m) =>
      parseInt(m.replace(/[[\]]/g, ""), 10)
    )
  );
  const citationsInRange = maxCitationIdx <= sources.length;

  return {
    wordCount,
    citationCount,
    citationCoverage: parseFloat(citationCoverage.toFixed(2)),
    lengthOk,
    citationsInRange,
    heuristicScore: parseFloat(
      (
        citationCoverage * 0.4 +
        (lengthOk ? 0.3 : 0) +
        (citationsInRange ? 0.3 : 0)
      ).toFixed(2)
    ),
  };
}
