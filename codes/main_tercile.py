"""main_tercile.py

Runs the stock MMHCL training loop (main.py) with three additions:

  1. Per improved-epoch, computes Recall@20 restricted to items in the
     Head / Mid / Tail popularity tercile (by training frequency).
  2. Injects these three values into every wandb.log(...) call that
     already carries 'test/recall@20', so they appear as time-series
     curves in the WandB UI.
  3. On wandb.finish(...), writes Best_Recall@20 Head/Mid/Tail into
     the run summary -- these are the tercile recalls at the SAME
     checkpoint whose best_test_recall@20 is already reported by main.py
     (i.e., last improved epoch under early-stopping-restore-best=1).

Usage (drop-in for main.py):
    python main_tercile.py --dataset Clothing --seed <s> [ ... ]
"""
from __future__ import annotations

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
            ground = data_generator.test_set.get(u, [])
            if not ground:
                continue
            top = np.argpartition(-scores, K)[:K]
            top = top[np.argsort(-scores[top])].tolist()
            gset = set(ground)
            for tercile, out in (
                (HEAD_IDS, head_scores),
                (MID_IDS,  mid_scores),
                (TAIL_IDS, tail_scores),
            ):
                gt_in_tercile = gset & tercile
                if not gt_in_tercile:
                    continue
                hits = sum(
                    1 for it in top if it in tercile and it in gt_in_tercile
                )
                out.append(hits / len(gt_in_tercile))
    _m = lambda xs: float(np.mean(xs)) if xs else float("nan")
    return {"head": _m(head_scores), "mid": _m(mid_scores), "tail": _m(tail_scores)}


# --- 4. Monkey-patch Trainer.test to also stash the tercile snapshot ------
_orig_test = main.Trainer.test
_last_improved_tercile: dict[str, float] = {"head": float("nan"),
                                             "mid":  float("nan"),
                                             "tail": float("nan")}
_have_improved = [False]


def _test_with_terciles(self, users_to_test, is_val):
    result = _orig_test(self, users_to_test, is_val)
    if not is_val:  # test-set evaluation triggered by improvement in val
        self.model.eval()
        with torch.no_grad():
            ua, ia, _ii, _uu = self.model(
                self.UI_mat, self.Item_mat, self.User_mat
            )
        ter = compute_tercile_recall(ua, ia, users_to_test)
        _last_improved_tercile.update(ter)
        _have_improved[0] = True
        print(
            f"[tercile] test @ improved epoch: "
            f"head={ter['head']:.6f} mid={ter['mid']:.6f} tail={ter['tail']:.6f}",
            flush=True,
        )
    return result


main.Trainer.test = _test_with_terciles

# --- 5. Monkey-patch wandb.log / wandb.finish so the UI sees the metrics --
if args.use_wandb and wandb is not None:
    _orig_log = wandb.log

    def _patched_log(d, *a, **kw):
        if isinstance(d, dict) and "test/recall@20" in d and _have_improved[0]:
            d = dict(d)
            d["Recall@20 Head"] = _last_improved_tercile["head"]
            d["Recall@20 Mid"]  = _last_improved_tercile["mid"]
            d["Recall@20 Tail"] = _last_improved_tercile["tail"]
        return _orig_log(d, *a, **kw)

    wandb.log = _patched_log

    _orig_finish = wandb.finish

    def _patched_finish(*a, **kw):
        try:
            if _have_improved[0] and wandb.run is not None:
                wandb.summary["Best_Recall@20 Head"] = _last_improved_tercile["head"]
                wandb.summary["Best_Recall@20 Mid"]  = _last_improved_tercile["mid"]
                wandb.summary["Best_Recall@20 Tail"] = _last_improved_tercile["tail"]
        except Exception as _e:
            print(f"[tercile] wandb-summary write skipped: {_e}", flush=True)
        return _orig_finish(*a, **kw)

    wandb.finish = _patched_finish

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

    # After training + wandb.finish, also emit the tercile summary to stdout
    # so the notebook parser (below) can pick it up even if the WandB API
    # call is rate-limited.
    print(
        "[tercile-final] "
        f"BEST_Recall@20_Head={_last_improved_tercile['head']:.8f} "
        f"BEST_Recall@20_Mid={_last_improved_tercile['mid']:.8f} "
        f"BEST_Recall@20_Tail={_last_improved_tercile['tail']:.8f}",
        flush=True,
    )
