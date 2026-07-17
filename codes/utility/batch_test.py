"""
batch_test.py — Evaluation Pipeline
=====================================

Handles the evaluation of trained MMHCL models on validation or test sets.

Key responsibilities:
  1. Create the global ``data_generator`` (Data object) at import time.
     This is shared by main.py via ``from utility.batch_test import *``.
  2. For each user, compute predicted scores for all items via dot product,
     then rank them and compute recommendation metrics.
  3. Use Python multiprocessing to parallelise per-user metric computation.

Evaluation flow:
    test_torch()                         ← called from Trainer.test()
     ├── For each user batch:
     │    ├── Compute scores = user_emb · item_emb^T
     │    ├── Move to CPU (numpy)
     │    └── pool.map(test_one_user, ...)   ← parallel per-user eval
     │         ├── Remove training items from candidates
     │         ├── Rank items by predicted score
     │         └── Compute Precision, Recall, NDCG, Hit, AUC
     └── Average metrics across all test users

Two ranking strategies:
  - 'part' mode (default): uses heapq.nlargest — O(N log K), faster,
    but does not compute AUC.
  - 'full' mode: full sort — O(N log N), slower, but also computes AUC.
"""

from __future__ import annotations

from typing import Any

import utility.metrics as metrics
from utility.parser import parse_args
from utility.load_data import Data
import multiprocessing
import argparse
import heapq
import torch
import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
#  Module-level initialisation (runs at import time)
# ---------------------------------------------------------------------------
# Use 1/5 of available CPU cores for the multiprocessing pool
cores: int = multiprocessing.cpu_count() // 5

args: argparse.Namespace = parse_args()
Ks: list[int] = eval(args.Ks)  # e.g. [10, 20]

# Create the global data generator — this loads the dataset from disk
# and is shared by main.py for training sampling and graph construction.
data_generator: Data = Data(
    path=args.data_path + args.dataset, batch_size=args.batch_size
)
USR_NUM: int = data_generator.n_users
ITEM_NUM: int = data_generator.n_items
N_TRAIN: int = data_generator.n_train
N_TEST: int = data_generator.n_test
BATCH_SIZE: int = args.batch_size


# Type alias for the result dictionary returned by evaluation functions
MetricsDict = dict[str, Any]


# ===========================================================================
#  Ranking Functions
# ===========================================================================
def ranklist_by_heapq(
    user_pos_test: list[int],
    test_items: list[int],
    rating: npt.NDArray[np.floating],
    Ks: list[int],
) -> tuple[list[int], float]:
    """
    Rank items using a max-heap and return the top-K relevance vector.

    Uses heapq.nlargest() which is O(N log K) — much faster than full sort
    when K << N.  Does NOT compute AUC (returns 0).

    Args:
        user_pos_test : ground-truth positive items for this user
        test_items    : candidate items (all items minus training items)
        rating        : predicted scores for all items
        Ks            : K values, e.g. [10, 20]

    Returns:
        Tuple of (r, auc) where r is a binary relevance vector and auc is 0.0.
    """
    item_score: dict[int, float] = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max: int = max(Ks)
    # Get the top-K_max items by predicted score
    K_max_item_score: list[int] = heapq.nlargest(K_max, item_score, key=item_score.get)

    # Build binary relevance vector: 1 if the item is in ground truth, else 0
    r: list[int] = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc: float = 0.
    return r, auc


def get_auc(
    item_score: dict[int, float], user_pos_test: list[int]
) -> float:
    """
    Compute AUC (Area Under ROC Curve) for a single user.

    Sorts all items by predicted score and uses sklearn's roc_auc_score.
    """
    sorted_items: list[tuple[int, float]] = sorted(
        item_score.items(), key=lambda kv: kv[1]
    )
    sorted_items.reverse()
    item_sort: list[int] = [x[0] for x in sorted_items]
    posterior: list[float] = [x[1] for x in sorted_items]

    r: list[int] = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc: float = metrics.auc(ground_truth=r, prediction=posterior)
    return auc


def ranklist_by_sorted(
    user_pos_test: list[int],
    test_items: list[int],
    rating: npt.NDArray[np.floating],
    Ks: list[int],
) -> tuple[list[int], float]:
    """
    Rank items using full sort.  Slower than heapq but also computes AUC.

    Used when --test_flag full is specified.
    """
    item_score: dict[int, float] = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max: int = max(Ks)
    K_max_item_score: list[int] = heapq.nlargest(K_max, item_score, key=item_score.get)

    r: list[int] = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc: float = get_auc(item_score, user_pos_test)
    return r, auc


# ===========================================================================
#  Per-User Metric Computation
# ===========================================================================
def get_performance(
    user_pos_test: list[int],
    r: list[int],
    auc: float,
    Ks: list[int],
) -> MetricsDict:
    """
    Compute all recommendation metrics for a single user.

    Returns:
        dict with keys: 'recall', 'precision', 'ndcg', 'hit_ratio', 'auc'
        Each metric (except 'auc') is a numpy array of length len(Ks).
    """
    precision: list[float] = []
    recall: list[float] = []
    ndcg: list[float] = []
    hit_ratio: list[float] = []

    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))
        hit_ratio.append(metrics.hit_at_k(r, K))

    return {
        'recall': np.array(recall),
        'precision': np.array(precision),
        'ndcg': np.array(ndcg),
        'hit_ratio': np.array(hit_ratio),
        'auc': auc,
    }


def test_one_user(x: tuple[npt.NDArray[np.floating], int, bool]) -> MetricsDict:
    """
    Evaluate a single user — designed to be called by multiprocessing.Pool.map().

    Input ``x`` is a tuple: (rating_vector, user_id, is_val_flag)

    Steps:
        1. Look up the user's training items (to exclude from candidates).
        2. Look up ground-truth items (from val or test set).
        3. Candidate items = all items - training items.
        4. Rank candidates by predicted score and compute metrics.

    Returns:
        dict of metrics for this user (same format as get_performance).
    """
    is_val: bool = x[-1]
    rating: npt.NDArray[np.floating] = x[0]   # predicted scores for ALL items
    u: int = x[1]                              # user ID

    # Items this user interacted with during training (must be excluded)
    training_items: list[int]
    try:
        training_items = data_generator.train_items[u]
    except Exception:
        training_items = []

    # Ground-truth positive items (from val or test set)
    user_pos_test: list[int]
    if is_val:
        user_pos_test = data_generator.val_set[u]
    else:
        user_pos_test = data_generator.test_set[u]

    # Candidate items = all items minus training items
    all_items: set[int] = set(range(ITEM_NUM))
    test_items: list[int] = list(all_items - set(training_items))

    # Rank and compute metrics
    r: list[int]
    auc: float
    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, auc, Ks)


# ===========================================================================
#  Main Evaluation Function
# ===========================================================================
def test_torch(
    ua_embeddings: torch.Tensor,
    ia_embeddings: torch.Tensor,
    users_to_test: list[int],
    is_val: bool,
    drop_flag: bool = False,
    batch_test_flag: bool = False,
) -> MetricsDict:
    """
    Evaluate the model on a set of users using GPU-accelerated scoring
    and multiprocessing for per-user metric computation.

    Args:
        ua_embeddings   : (n_users, emb_dim) tensor — all user embeddings (GPU)
        ia_embeddings   : (n_items, emb_dim) tensor — all item embeddings (GPU)
        users_to_test   : user IDs to evaluate
        is_val          : True for validation, False for test
        drop_flag       : unused (kept for API compatibility)
        batch_test_flag : if True, compute user-item scores in batches

    Returns:
        dict with averaged metrics: 'precision', 'recall', 'ndcg',
        'hit_ratio', 'auc' — each is a numpy array of shape (len(Ks),).
    """
    # Initialise accumulators for averaging
    result: MetricsDict = {
        'precision': np.zeros(len(Ks)),
        'recall': np.zeros(len(Ks)),
        'ndcg': np.zeros(len(Ks)),
        'hit_ratio': np.zeros(len(Ks)),
        'auc': 0.,
    }
    pool: multiprocessing.pool.Pool = multiprocessing.Pool(cores)

    # Use 2× batch_size for user batches (scoring is cheaper than training)
    u_batch_size: int = BATCH_SIZE * 2
    i_batch_size: int = BATCH_SIZE

    test_users: list[int] = users_to_test
    n_test_users: int = len(test_users)
    n_user_batchs: int = n_test_users // u_batch_size + 1
    count: int = 0

    for u_batch_id in range(n_user_batchs):
        start: int = u_batch_id * u_batch_size
        end: int = (u_batch_id + 1) * u_batch_size
        user_batch: list[int] = test_users[start: end]

        rate_batch: npt.NDArray[np.floating] | torch.Tensor

        if batch_test_flag:
            # ----- Batched scoring: compute scores in item-chunks -----
            n_item_batchs: int = ITEM_NUM // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), ITEM_NUM))

            i_count: int = 0
            for i_batch_id in range(n_item_batchs):
                i_start: int = i_batch_id * i_batch_size
                i_end: int = min((i_batch_id + 1) * i_batch_size, ITEM_NUM)

                item_batch: range = range(i_start, i_end)
                u_g_embeddings: torch.Tensor = ua_embeddings[user_batch]
                i_g_embeddings: torch.Tensor = ia_embeddings[item_batch]
                # score(u, i) = u_emb^T · i_emb
                i_rate_batch: torch.Tensor = torch.matmul(
                    u_g_embeddings, torch.transpose(i_g_embeddings, 0, 1)
                )

                rate_batch[:, i_start: i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == ITEM_NUM

        else:
            # ----- Full scoring: compute all user-item scores at once -----
            item_batch = range(ITEM_NUM)
            u_g_embeddings = ua_embeddings[user_batch]
            i_g_embeddings = ia_embeddings[item_batch]
            # score matrix: (|user_batch|, n_items)
            rate_batch = torch.matmul(
                u_g_embeddings, torch.transpose(i_g_embeddings, 0, 1)
            )

        # Move scores to CPU for per-user evaluation
        rate_batch = rate_batch.detach().cpu().numpy()

        # Zip scores with user IDs and val/test flag for multiprocessing
        user_batch_rating_uid = zip(
            rate_batch, user_batch, [is_val] * len(user_batch)
        )

        # Parallel per-user metric computation
        batch_result: list[MetricsDict] = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)

        # Accumulate (running average: each user contributes 1/n_test_users)
        for re in batch_result:
            result['precision'] += re['precision'] / n_test_users
            result['recall'] += re['recall'] / n_test_users
            result['ndcg'] += re['ndcg'] / n_test_users
            result['hit_ratio'] += re['hit_ratio'] / n_test_users
            result['auc'] += re['auc'] / n_test_users

    assert count == n_test_users
    pool.close()
    return result
