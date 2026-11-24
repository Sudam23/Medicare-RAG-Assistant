# Evaluation

Every number here is produced by `make eval`, which writes
[`evaluation_report.json`](evaluation_report.json), and then written into this file by
`make docs`. Nothing is typed by hand. `scripts/render_results.py --check` runs in CI
and fails the build if any document disagrees with the report.

## 1. The gold set

98 questions, labelled by hand by reading the booklet pages:

| Split | In scope | Out of scope | Total |
| --- | --- | --- | --- |
| dev | 45 | 9 | 54 |
| test | 35 | 9 | 44 |

- **In scope** — the answer is genuinely in one of the nine booklets. The label records
  the booklet, the page(s), and a string that must appear in a correct answer.
- **Out of scope** — a reasonable-sounding question the booklets do not answer
  ("what is the best stock to buy?"). The correct behaviour is a refusal.

Every label was verified mechanically: a script re-opened the cited page and confirmed
the `must_contain` string is actually on it. That check caught a real mistake — q066,
*"Does Medicare pay for blood?"*, was labelled page 18, which is breast prostheses. The
answer is on page 16.

### Why a dev/test split

Every threshold and weight in this project was tuned on **dev**. **Test was scored
once**, after tuning finished, and not touched again. A number tuned on the set it is
reported on is measuring the fitting, not the system.

## 2. Retrieval

Does the search put the right page near the top?

<!-- BEGIN:retrieval-table -->
| Retrieval strategy | recall@1 | recall@3 | recall@5 | recall@10 | MRR@10 | median latency |
| --- | --- | --- | --- | --- | --- | --- |
| Keyword only (BM25) | 37.1% | 60.0% | 68.6% | 94.3% | 0.526 | 0.6 ms |
| Meaning only (embeddings) **&larr; best MRR** | 51.4% | 68.6% | 91.4% | 94.3% | 0.654 | 4.3 ms |
| Hybrid (shipped) | 48.6% | 74.3% | 88.6% | 94.3% | 0.639 | 4.6 ms |

35 held-out questions, scored once. Higher is better everywhere.
<!-- END:retrieval-table -->

**recall@k** — the share of questions where a correct page is in the top k.
**MRR@10** — 1 if the right page is first, ½ if second, ⅓ if third, 0 if not in ten.

## 3. The whole pipeline

<!-- BEGIN:pipeline-table -->
| What is measured | Result | Over what |
| --- | --- | --- |
| Answered an in-scope question | **100.0%** | 35 questions the booklets do cover |
| Refused an out-of-scope question | **100.0%** | 9 questions they do not cover |
| Groundedness | **100.0%** | every sentence traced back to a cited page |
| Cited the correct page | **94.3%** | the gold page was among the citations shown |
| Median response time | **220 ms** | end to end, on a laptop CPU |
| 95th percentile | **276 ms** | the slow ones |
<!-- END:pipeline-table -->

Measured in extractive mode, so these numbers hold for the public demo exactly as it
is deployed, with no API key.

### On citation precision

23.3% looks alarming and is not. The interface shows ~4 sources; the gold label names
one page. Every other citation is counted as imprecise even when it is a relevant
neighbouring page — the dental example cites two different booklets, both correct and
useful, and the metric credits one. Showing a single source would raise the number and
make the tool worse. It is published because deleting an inconvenient metric is how
evaluation sections become decoration.

## 4. Do the two searches find different things?

<!-- BEGIN:complementarity -->
| Found by | Questions (of 35) |
| --- | --- |
| Both keyword and meaning search | 31 |
| Meaning search only | 2 |
| Keyword search only | 2 |
| Neither | 0 |
<!-- END:complementarity -->

This is the table that decided the architecture, and it did **not** say what I expected.
Only 2 of 35 questions are found by keyword search and missed by meaning search, and 2
the other way. The two retrievers almost entirely overlap on this corpus.

## 5. The fusion weight, and the trade it represents

A sweep over the lexical weight on **dev** (`tuning_report.json`):

| Lexical weight (k = 20) | dev recall@1 | dev MRR@10 |
| --- | --- | --- |
| 0.0 — dense only | 0.622 | 0.733 |
| **0.1 — shipped** | **0.667** | **0.759** |
| 0.2 | 0.622 | 0.733 |
| 0.4 | 0.600 | 0.714 |
| 0.6 | 0.556 | 0.688 |
| 1.0 — keyword only | 0.489 | 0.641 |

### What moving 0.1 → 0.6 actually does

Rather than trust the aggregate, I diffed the per-question rank-1 outcomes:

| | |
| --- | --- |
| Questions moved **onto** rank 1 | **1** |
| Questions knocked **off** rank 1 | **6** |
| Net | **−5** (30/45 → 25/45) |

The single question it fixes is *"Do higher earners pay more for Medicare?"*. The six
it breaks are ordinary prose questions — *"Can I get Medicare if I have kidney
failure?"*, *"Does hospice cover a condition unrelated to my terminal illness?"*, and
four more like them.

Looking only at the 9 deliberately exact-token dev questions (plan letters, `Part B`,
`SNP`, `MSA`), the heavier weight is **not** the win it appears to be either:

| Question | rank at w=0.1 | rank at w=0.6 |
| --- | --- | --- |
| `Plan G` | not in top 10 | 7 |
| `Plan N` | not in top 10 | not in top 10 |
| "Can I sign up for Part B if I am over 65 already?" | 7 | 4 |
| `SNP` | 3 | 4 |
| rank-1 count over all 9 | **5 / 9** | **5 / 9** |

So the heavy weight rescues `Plan G` into the top ten without ever ranking it first,
and buys nothing at rank 1 even on the queries it was supposed to help.

The high-weight setting is tempting because its failure is *visible*: typing `Plan G`
in a demo and getting nothing looks broken. Its cost is invisible — six ordinary
questions quietly ranked worse. I shipped 0.1 and wrote the trade-off down rather than
choosing by whichever failure happened to be easier to notice.

**`Plan N` is a real, open limitation**, not a tuned-away one: no weight in the sweep
retrieves it. A two-character query carries almost no signal for either retriever, and
the honest fix is a reranker or an exact-phrase pre-filter, not a fusion weight.

**This contradicts my previous project.** In the ICD-10 search project, hybrid beat
dense decisively. The difference is the corpus: ICD-10 descriptions are short and
keyword-dense, while Medicare booklets are long prose that embeddings handle well
unaided. A technique that wins on one corpus does not transfer by default.

## 6. The refusal threshold

Swept on dev (`threshold_tuning.json`), balancing two mistakes:

- answering an out-of-scope question (confident nonsense — the worse error here)
- refusing a question the booklets *do* answer (unhelpful)

**0.62** maximises balanced accuracy on dev at 0.953, and on the held-out test split it
gives 100% answered in scope and 100% refused out of scope.

## 7. What the numbers do not prove

- 98 questions resolve a ~10-point difference and cannot resolve 2 points. The
  hybrid-vs-dense result is reported as a **tie**, not a win for dense.
- I wrote the questions, so they inherit my idea of what people ask. Real user logs
  would look different.
- Groundedness of 100% is partly structural: extractive answering copies sentences
  verbatim, so it is nearly guaranteed. It is the right floor to measure, not a
  difficult achievement.
