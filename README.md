# pplx-eval-harness

LLM-as-judge evaluation harness for Perplexity answer quality.

## Evaluation dimensions

| Dimension | Weight | Description |
|---|---|---|
| Factual accuracy | 0.35 | Are the claims correct? |
| Citation grounding | 0.25 | Do citations support the claims? |
| Completeness | 0.20 | Does it fully answer the question? |
| Conciseness | 0.10 | Is there unnecessary verbosity? |
| Coherence | 0.10 | Is it logically structured? |

Overall score = weighted average across all dimensions (1–5 scale).

## Retrieval metrics

Full suite: Precision@K, Recall@K, F1@K, NDCG@K, MRR, ROUGE-L, citation precision/recall.

```js
import { ndcgAtK, mrr, rougeL, aggregateScores } from './src/metrics/index.js';

const ndcg = ndcgAtK(retrievedIds, relevantSet, 10);
const mrrScore = mrr(queryResults);
const rouge = rougeL(generatedAnswer, referenceAnswer);
```

## Heuristic judge (no API cost)

Fast pre-screening without an LLM call:

```js
import { heuristicJudge } from './src/judges/llm_judge.js';
const r = heuristicJudge(answer, sources);
// { citationCoverage: 0.8, citationsInRange: true, heuristicScore: 0.74, ... }
```

## Tests

```bash
npm test   # 36 tests
```
