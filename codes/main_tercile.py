"""main_tercile.py

Runs the stock MMHCL training loop (main.py) with three additions:

  1. On every validation evaluation, computes Recall@20 restricted to
     items in the Head / Mid / Tail popularity tercile (by training frequency).
  2. Whenever validation improves under the same rule as main.py's early
     stopping, snapshots those tercile recalls as the running best
     (independent of WandB).
  3. When main.py logs the improved-checkpoint metrics (``best_recall`` /
     ``best_ndcg``), also logs:
         best_recall@20_Head
         best_recall@20_Mid
         best_recall@20_Tail
     and writes the same three values into the WandB run summary.

Usage (drop-in for main.py):
    python main_tercile.py --dataset Clothing --seed <s> [ ... ]
"""
from __future__ import annotations

import math
import os
import numpy as np
import torch

# --- 1. Import main.py -- runs argparse + data_generator + Trainer defs,
#        but NOT training (guarded by `if __name__ == "__main__"`).
import main
from utility.batch_test import Ks as _BT_KS  # noqa: F401

args = main.args
data_generator = main.data_generator
wandb = main.wandb

# --- 2. Item -> tercile assignment (once, at import time) ------------------
_n_items = data_generator.n_items
_item_freq = np.zeros(_n_items, dtype=np.int64)
for _u, _items in data_generator.train_items.items():
    for _i in _items:
        _item_freq[_i] += 1
_order = np.argsort(_item_freq, kind="stable")   # ascending -> tail first
_t1 = _n_items // 3
_t2 = 2 * _n_items // 3
TAIL_IDS: set[int] = set(_order[:_t1].tolist())
MID_IDS:  set[int] = set(_order[_t1:_t2].tolist())
HEAD_IDS: set[int] = set(_order[_t2:].tolist())
print(
    f"[tercile] n_items={_n_items}  "
    f"|Tail|={len(TAIL_IDS)}  |Mid|={len(MID_IDS)}  |Head|={len(HEAD_IDS)}  "
    f"tail-freq<={int(_item_freq[_order[_t1 - 1]])}  "
    f"head-freq>={int(_item_freq[_order[_t2]])}",
    flush=True,
)

# --- 3. GPU-batched tercile recall evaluator (no multiprocessing) ---------
@torch.no_grad()
def compute_tercile_recall(
    ua: torch.Tensor,
    ia: torch.Tensor,
    users_to_test: list[int],
    ground_truth: dict[int, list[int]],
    K: int = 20,
) -> dict[str, float]:
    """Recall@K restricted to each popularity tercile.

    Per-user tercile recall = |hits in top-K that fall in tercile AND
    are ground-truth| / |ground-truth positives that fall in tercile|.
    Users with zero tercile-positives are skipped for that tercile
    (matches Milogradskii et al. 2024, Krichene & Rendle 2020).
    """
    head_scores: list[float] = []
    mid_scores:  list[float] = []
    tail_scores: list[float] = []
    ubs = 2048  # user-batch size for scoring
    for start in range(0, len(users_to_test), ubs):
        batch = users_to_test[start : start + ubs]
        ub = ua[batch]                                # (B, d)
        rate = (ub @ ia.T).cpu().numpy()              # (B, n_items)
        for row, u in enumerate(batch):
            scores = rate[row].copy()
            for ti in data_generator.train_items.get(u, []):
                scores[ti] = -1e9                     # exclude trained items
            ground = ground_truth.get(u, [])
            if not ground:
                continue
            top = np.argpartition(-scores, K)[:K]
            top = top[np.argsort(-scores[top])].tolist()
            gset = set(int(x) for x in ground)
            for tercile, out in (
                (HEAD_IDS, head_scores),
                (MID_IDS,  mid_scores),
                (TAIL_IDS, tail_scores),
            ):
                gt_in_tercile = gset & tercile
                if not gt_in_tercile:
                    continue
                hits = sum(
                    1 for it in top if int(it) in tercile and int(it) in gt_in_tercile
                )
                out.append(hits / len(gt_in_tercile))
    _m = lambda xs: float(np.mean(xs)) if xs else float("nan")
    return {"head": _m(head_scores), "mid": _m(mid_scores), "tail": _m(tail_scores)}


# --- 4. Monkey-patch Trainer.test to stash validation terciles ------------
_orig_test = main.Trainer.test

# Most recent validation-tercile snapshot (updated every val eval epoch).
_last_val_tercile: dict[str, float] = {
    "head": float("nan"),
    "mid": float("nan"),
    "tail": float("nan"),
}
# Snapshot at the same checkpoint that updates best_recall / best_ndcg.
# Updated here (mirroring main.py early-stopping), NOT only via wandb.log.
_best_val_tercile: dict[str, float] = {
    "head": float("nan"),
    "mid": float("nan"),
    "tail": float("nan"),
}
_best_monitor: dict[str, float] = {"recall": 0.0, "ndcg": 0.0}
_have_val_tercile = [False]


def _test_with_terciles(self, users_to_test, is_val):
    result = _orig_test(self, users_to_test, is_val)
    if is_val:
        self.model.eval()
        with torch.no_grad():
            ua, ia, _ii, _uu = self.model(
                self.UI_mat, self.Item_mat, self.User_mat
            )
        ter = compute_tercile_recall(
            ua, ia, users_to_test, data_generator.val_set
        )
        _last_val_tercile.update(ter)
        _have_val_tercile[0] = True

        # Mirror main.py early-stopping "improved" rule so best Head/Mid/Tail
        # are recorded even when WandB is disabled or wandb.log is skipped.
        rec = float(result["recall"][1])
        ndcg = float(result["ndcg"][1])
        min_delta = float(args.early_stopping_min_delta)
        improved = (
            rec > _best_monitor["recall"] + min_delta
            or ndcg > _best_monitor["ndcg"] + min_delta
        )
        if improved:
            if rec > _best_monitor["recall"]:
                _best_monitor["recall"] = rec
            if ndcg > _best_monitor["ndcg"]:
                _best_monitor["ndcg"] = ndcg
            _best_val_tercile.update(ter)
            print(
                f"[tercile] val@best: "
                f"head={ter['head']:.6f} mid={ter['mid']:.6f} tail={ter['tail']:.6f} "
                f"(val_recall@20={rec:.6f})",
                flush=True,
            )
        else:
            print(
                f"[tercile] val: "
                f"head={ter['head']:.6f} mid={ter['mid']:.6f} tail={ter['tail']:.6f}",
                flush=True,
            )
    return result


main.Trainer.test = _test_with_terciles

# --- 5. Monkey-patch wandb.log / wandb.finish so the UI sees the metrics --
if args.use_wandb and wandb is not None:
    _orig_log = wandb.log

    def _patched_log(d, *a, **kw):
        if isinstance(d, dict) and _have_val_tercile[0]:
            d = dict(d)
            if "val/recall@20" in d:
                d["val/recall@20_Head"] = _last_val_tercile["head"]
                d["val/recall@20_Mid"] = _last_val_tercile["mid"]
                d["val/recall@20_Tail"] = _last_val_tercile["tail"]
            if "best_recall" in d:
                # Prefer the already-snapshotted best; fall back to last val.
                src = _best_val_tercile
                if math.isnan(src["head"]):
                    src = _last_val_tercile
                    _best_val_tercile.update(src)
                d["best_recall@20_Head"] = src["head"]
                d["best_recall@20_Mid"] = src["mid"]
                d["best_recall@20_Tail"] = src["tail"]
        return _orig_log(d, *a, **kw)

    wandb.log = _patched_log

    _orig_finish = wandb.finish

    def _patched_finish(*a, **kw):
        try:
            if _have_val_tercile[0] and wandb.run is not None:
                src = _best_val_tercile
                if math.isnan(src["head"]):
                    src = _last_val_tercile
                wandb.summary["best_recall@20_Head"] = src["head"]
                wandb.summary["best_recall@20_Mid"] = src["mid"]
                wandb.summary["best_recall@20_Tail"] = src["tail"]
        except Exception as _e:
            print(f"[tercile] wandb-summary write skipped: {_e}", flush=True)
        return _orig_finish(*a, **kw)

    wandb.finish = _patched_finish


def _fmt(x: float) -> str:
    """Format floats for the notebook regex (never emit bare 'nan')."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    return f"{float(x):.8f}"


# --- 6. Replicate main.py's __main__ block exactly ------------------------
if __name__ == "__main__":
    from typing import Any as _Any
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    main.set_seed(args.seed)

    config: dict[str, _Any] = {
        "n_users": data_generator.n_users,
        "n_items": data_generator.n_items,
    }
    config["UI_mat"]   = data_generator.get_UI_mat()
    config["User_mat"] = data_generator.get_U2U_mat()
    if args.dataset == "Tiktok":
        config["Item_mat"] = data_generator.get_tiktok_I2I_Hypergraph_mul_mat()
    else:
        config["Item_mat"] = data_generator.get_I2I_Hypergraph_mul_mat()

    trainer = main.Trainer(data_config=config)
    trainer.train()

    final = dict(_best_val_tercile)
    if math.isnan(final["head"]) and _have_val_tercile[0]:
        final = dict(_last_val_tercile)

    # Notebook parser reads this line; keep the key format stable.
    print(
        "[tercile-final] "
        f"BEST_Recall@20_Head={_fmt(final['head'])} "
        f"BEST_Recall@20_Mid={_fmt(final['mid'])} "
        f"BEST_Recall@20_Tail={_fmt(final['tail'])}",
        flush=True,
    )
