"""Ranking metrics for daily top-K lists; definitions follow section 2.1 of the design doc.

Ties in a score are resolved in expectation, as if the order inside a tie were random.
Row order must never break ties: in the source CSV it is related to the label.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray = npt.NDArray[np.float64]


def roc_auc(score: pd.Series, label: pd.Series) -> float:
    """ROC-AUC of a single score via the Mann-Whitney U statistic (ties get average ranks)."""
    ranks = score.rank()
    n_pos = int(label.sum())
    n_neg = len(label) - n_pos
    return float((ranks[label == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(score: FloatArray, label: FloatArray) -> float:
    """Average precision with one step per distinct score, as in scikit-learn."""
    order = np.argsort(-score, kind="mergesort")
    ordered_score, ordered_label = score[order], label[order]
    last_of_tie = np.r_[np.flatnonzero(np.diff(ordered_score)), len(ordered_score) - 1]
    true_positives = np.cumsum(ordered_label)[last_of_tie]
    precision = true_positives / (last_of_tie + 1)
    recall = true_positives / ordered_label.sum()
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def expected_top_k_hits(score: FloatArray, label: FloatArray, k: int) -> float:
    """Expected number of positives among the k highest scores under random tie-breaking."""
    if len(score) <= k:
        return float(label.sum())
    threshold = np.partition(score, len(score) - k)[len(score) - k]
    above = score > threshold
    tied = score == threshold
    free_slots = k - int(above.sum())
    return float(label[above].sum() + free_slots * label[tied].mean())


def daily_top_k(frame: pd.DataFrame, score: str, ks: Sequence[int]) -> pd.DataFrame:
    """Candidates, positives and expected top-k hits for every (hotel, snapshot day)."""
    rows = []
    for (hotel, day), group in frame.groupby(["hotel", "snapshot_date"], sort=True):
        values = group[score].to_numpy(dtype=np.float64)
        labels = group["label"].to_numpy(dtype=np.float64)
        row = {
            "hotel": hotel,
            "snapshot_date": day,
            "candidates": len(labels),
            "positives": labels.sum(),
        }
        row.update({f"hits_{k}": expected_top_k_hits(values, labels, k) for k in ks})
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_top_k(daily: pd.DataFrame, ks: Sequence[int]) -> dict[str, float]:
    """Macro Precision@K, Recall@K and Lift@K over days and the perfect-ranking ceiling.

    Lift@K is the ratio of macro precision to the mean daily positive rate, not a mean of
    daily lifts. Recall@K averages only days with at least one positive.
    """
    listed = daily[daily["candidates"] > 0]
    candidates = listed["candidates"].to_numpy(dtype=np.float64)
    positives = listed["positives"].to_numpy(dtype=np.float64)
    has_positives = positives > 0
    base_rate = float(np.mean(positives / candidates))
    result = {
        "days": float(len(listed)),
        "days_without_positives": float(np.sum(~has_positives)),
        "mean_candidates": float(np.mean(candidates)),
        "mean_positives": float(np.mean(positives)),
        "mean_daily_rate": base_rate,
    }
    for k in ks:
        hits = listed[f"hits_{k}"].to_numpy(dtype=np.float64)
        budget = np.minimum(float(k), candidates)
        precision = float(np.mean(hits / budget))
        recall = float(np.mean(hits[has_positives] / positives[has_positives]))
        ceiling = float(np.mean(np.minimum(budget, positives) / budget))
        found = positives[has_positives]
        ceiling_recall = float(np.mean(np.minimum(float(k), found) / found))
        result |= {
            f"precision_at_{k}": precision,
            f"recall_at_{k}": recall,
            f"lift_at_{k}": precision / base_rate,
            f"ceiling_precision_at_{k}": ceiling,
            f"ceiling_recall_at_{k}": ceiling_recall,
            f"ceiling_lift_at_{k}": ceiling / base_rate,
        }
    return result
