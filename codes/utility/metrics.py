"""
metrics.py — Recommendation Evaluation Metrics
================================================

Implements the standard top-K evaluation metrics used in recommendation
system research:

    Precision@K  — fraction of recommended items that are relevant
    Recall@K     — fraction of relevant items that are recommended
    NDCG@K       — normalised discounted cumulative gain (position-aware)
    Hit@K        — whether at least one relevant item appears in top-K
    F1           — harmonic mean of precision and recall
    AUC          — area under the ROC curve (ranking quality over all items)

All functions expect a binary relevance vector ``r`` where:
    r[i] = 1  if the item at rank i is relevant (ground truth)
    r[i] = 0  otherwise

The relevance vector is typically produced by ranklist_by_heapq() or
ranklist_by_sorted() in batch_test.py.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import numpy.typing as npt
from sklearn.metrics import roc_auc_score


def recall(rank: list[int], ground_truth: list[int], N: int) -> float:
    """
    Recall@N: proportion of ground-truth items found in the top-N ranking.

    Args:
        rank         : ranked item IDs
        ground_truth : ground-truth relevant item IDs
        N            : cut-off position

    Returns:
        float in [0, 1]
    """
    return len(set(rank[:N]) & set(ground_truth)) / float(len(set(ground_truth)))


def precision_at_k(r: Sequence[int], k: int) -> float:
    """
    Precision@K: fraction of the top-K items that are relevant.

    Args:
        r : binary relevance vector (1 = relevant, 0 = not)
        k : cut-off position

    Returns:
        float in [0, 1]

    Raises:
        AssertionError if k < 1
    """
    assert k >= 1
    r_arr: npt.NDArray[np.floating] = np.asarray(r)[:k]
    return float(np.mean(r_arr))


def average_precision(r: Sequence[int], cut: int) -> float:
    """
    Average Precision (AP): area under the precision-recall curve,
    truncated at position ``cut``.

    Used as a building block for MAP (Mean Average Precision).

    Args:
        r   : binary relevance vector
        cut : maximum rank position to consider

    Returns:
        AP score
    """
    r_arr: npt.NDArray[np.floating] = np.asarray(r)
    out: list[float] = [precision_at_k(r_arr, k + 1) for k in range(cut) if r_arr[k]]
    if not out:
        return 0.
    return float(np.sum(out) / float(min(cut, np.sum(r_arr))))


def mean_average_precision(rs: list[Sequence[int]]) -> float:
    """
    Mean Average Precision (MAP): average of AP scores across multiple users.

    Args:
        rs : list of relevance vectors (one per user)

    Returns:
        MAP score
    """
    return float(np.mean([average_precision(r) for r in rs]))


def dcg_at_k(r: Sequence[Union[int, float]], k: int, method: int = 1) -> float:
    """
    Discounted Cumulative Gain (DCG@K).

    Measures the quality of a ranking by giving higher credit to relevant
    items appearing at earlier (higher) positions.

    Two methods:
        method 0: DCG = r[0] + Σ_{i=1}^{K-1} r[i] / log2(i+1)
        method 1: DCG = Σ_{i=0}^{K-1} r[i] / log2(i+2)     ← default

    Args:
        r      : relevance scores (binary or graded)
        k      : cut-off position
        method : 0 or 1 (formula variant)

    Returns:
        DCG score
    """
    r_arr: npt.NDArray[np.floating] = np.asfarray(r)[:k]
    if r_arr.size:
        if method == 0:
            return float(r_arr[0] + np.sum(r_arr[1:] / np.log2(np.arange(2, r_arr.size + 1))))
        elif method == 1:
            return float(np.sum(r_arr / np.log2(np.arange(2, r_arr.size + 2))))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r: Sequence[Union[int, float]], k: int, method: int = 1) -> float:
    """
    Normalised Discounted Cumulative Gain (NDCG@K).

    Normalises DCG by the Ideal DCG (IDCG) — the DCG achieved by the
    perfect ranking (all relevant items first).

    NDCG@K = DCG@K / IDCG@K

    A value of 1.0 means perfect ranking; 0.0 means no relevant items
    in the top-K.

    Args:
        r      : binary relevance vector
        k      : cut-off position
        method : DCG formula variant (default 1)

    Returns:
        float in [0, 1]
    """
    dcg_max: float = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def recall_at_k(r: Sequence[Union[int, float]], k: int, all_pos_num: int) -> float:
    """
    Recall@K: fraction of all positive items that appear in the top-K.

    Args:
        r           : binary relevance vector
        k           : cut-off position
        all_pos_num : total number of positive (relevant) items

    Returns:
        float in [0, 1]
    """
    r_arr: npt.NDArray[np.floating] = np.asfarray(r)[:k]
    if all_pos_num == 0:
        return 0.
    else:
        return float(np.sum(r_arr) / all_pos_num)


def hit_at_k(r: Sequence[Union[int, float]], k: int) -> float:
    """
    Hit@K: 1 if at least one relevant item appears in top-K, else 0.

    This is a binary indicator — useful for measuring whether the model
    can place *any* relevant item in the recommendation list.

    Args:
        r : binary relevance vector
        k : cut-off position

    Returns:
        1.0 or 0.0
    """
    r_arr: npt.NDArray[np.integer] = np.array(r)[:k]
    if np.sum(r_arr) > 0:
        return 1.
    else:
        return 0.


def F1(pre: float, rec: float) -> float:
    """
    F1 score: harmonic mean of precision and recall.

    Args:
        pre : precision
        rec : recall

    Returns:
        float in [0, 1]
    """
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.


def auc(ground_truth: list[int], prediction: list[float]) -> float:
    """
    AUC (Area Under ROC Curve): measures ranking quality over all items.

    Uses sklearn's roc_auc_score.  Returns 0.0 if computation fails
    (e.g. only one class present in ground_truth).

    Args:
        ground_truth : binary labels (1 = relevant)
        prediction   : predicted scores (higher = more relevant)

    Returns:
        float in [0, 1]
    """
    try:
        res: float = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        res = 0.
    return res
