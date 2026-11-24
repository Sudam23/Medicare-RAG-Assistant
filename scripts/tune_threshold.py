"""Sweep the refusal threshold on the DEV split.

``MIN_RELEVANCE_SCORE`` sets the trade every question-answering system has to
make: refuse too eagerly and real questions go unanswered, refuse too rarely
and the system confidently answers things it has no source for. One number
cannot be read without the other, so both are printed for every candidate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medicare_rag.evaluation.goldset import load_gold
from medicare_rag.evaluation.metrics import answer_rate, refusal_rate
from medicare_rag.evaluation.run import evaluate_pipeline
from medicare_rag.rag import graph as graph_module
from medicare_rag.rag.graph import RagPipeline
from medicare_rag.rag.retrieve import Retriever

THRESHOLDS = [0.50, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.70]


def main() -> None:
    questions = load_gold(split="dev")
    pipeline = RagPipeline(Retriever.load())
    rows = []
    original = graph_module.MIN_RELEVANCE_SCORE
    try:
        for threshold in THRESHOLDS:
            graph_module.MIN_RELEVANCE_SCORE = threshold
            outcomes = evaluate_pipeline(pipeline, questions)
            answered = answer_rate(outcomes)
            refused = refusal_rate(outcomes)
            # Both halves matter equally, so rank by their harmonic mean: a
            # setting that aces one and fails the other scores badly.
            balanced = (
                0.0 if (answered + refused) == 0 else 2 * answered * refused / (answered + refused)
            )
            rows.append(
                {
                    "threshold": threshold,
                    "answered_in_scope": round(answered, 4),
                    "refused_out_of_scope": round(refused, 4),
                    "balanced": round(balanced, 4),
                }
            )
            print(
                f"  {threshold:.2f}  answered_in_scope={answered:.3f}  "
                f"refused_out_of_scope={refused:.3f}  balanced={balanced:.3f}"
            )
    finally:
        graph_module.MIN_RELEVANCE_SCORE = original

    best = max(rows, key=lambda r: (r["balanced"], r["threshold"]))
    print(f"\n  best on dev: threshold={best['threshold']} balanced={best['balanced']:.3f}")
    out = Path(__file__).resolve().parents[1] / "docs" / "threshold_tuning.json"
    out.write_text(
        json.dumps({"split": "dev", "grid": rows, "best": best}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"  written to {out}")


if __name__ == "__main__":
    main()
