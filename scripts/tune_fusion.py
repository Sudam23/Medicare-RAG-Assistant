"""Sweep the fusion weights on the DEV split only.

The first evaluation showed hybrid retrieval scoring *below* the semantic arm
on its own, which is the usual symptom of a fusion weight that lets a weak arm
drag a strong one down. This sweep finds the weight that fixes it.

It runs on the dev split and never touches test. Choosing a setting by looking
at the test numbers is how a held-out set quietly stops being held out.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medicare_rag.evaluation.goldset import load_gold
from medicare_rag.evaluation.metrics import mrr_at, recall_at
from medicare_rag.evaluation.run import evaluate_retrieval
from medicare_rag.rag import retrieve as retrieve_module
from medicare_rag.rag.retrieve import Retriever

WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
RRF_KS = [5, 10, 20, 60]


def main() -> None:
    questions = load_gold(split="dev")
    retriever = Retriever.load()

    baseline = evaluate_retrieval(retriever, questions, "dense")
    print(
        f"  dense-only baseline: recall@1={recall_at(baseline, 1):.3f} "
        f"recall@5={recall_at(baseline, 5):.3f} mrr@10={mrr_at(baseline, 10):.3f}\n"
    )

    rows = []
    original_weight = retrieve_module.RRF_WEIGHT_LEXICAL
    original_k = retrieve_module.RRF_K
    try:
        for weight, rrf_k in itertools.product(WEIGHTS, RRF_KS):
            retrieve_module.RRF_WEIGHT_LEXICAL = weight
            retrieve_module.RRF_K = rrf_k
            outcomes = evaluate_retrieval(retriever, questions, "hybrid")
            rows.append(
                {
                    "lexical_weight": weight,
                    "rrf_k": rrf_k,
                    "recall@1": round(recall_at(outcomes, 1), 4),
                    "recall@3": round(recall_at(outcomes, 3), 4),
                    "recall@5": round(recall_at(outcomes, 5), 4),
                    "recall@10": round(recall_at(outcomes, 10), 4),
                    "mrr@10": round(mrr_at(outcomes, 10), 4),
                }
            )
    finally:
        retrieve_module.RRF_WEIGHT_LEXICAL = original_weight
        retrieve_module.RRF_K = original_k

    # Rank by MRR, which rewards putting the right page high rather than merely
    # somewhere in the list - the answer is built from the top few chunks.
    rows.sort(key=lambda r: (-r["mrr@10"], -r["recall@1"]))
    print(f"  {'weight':>7}{'rrf_k':>7}{'r@1':>8}{'r@3':>8}{'r@5':>8}{'r@10':>8}{'mrr@10':>9}")
    print("  " + "-" * 55)
    for row in rows[:12]:
        print(
            f"  {row['lexical_weight']:>7}{row['rrf_k']:>7}{row['recall@1']:>8.3f}"
            f"{row['recall@3']:>8.3f}{row['recall@5']:>8.3f}{row['recall@10']:>8.3f}{row['mrr@10']:>9.3f}"
        )

    best = rows[0]
    print(
        f"\n  best on dev: lexical_weight={best['lexical_weight']} rrf_k={best['rrf_k']} "
        f"-> mrr@10={best['mrr@10']:.3f} recall@1={best['recall@1']:.3f}"
    )
    out = Path(__file__).resolve().parents[1] / "docs" / "tuning_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "split": "dev",
                "dense_baseline": {
                    "recall@1": round(recall_at(baseline, 1), 4),
                    "recall@5": round(recall_at(baseline, 5), 4),
                    "mrr@10": round(mrr_at(baseline, 10), 4),
                },
                "grid": rows,
                "best": best,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"  written to {out}")


if __name__ == "__main__":
    main()
