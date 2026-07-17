"""
main.py — Training Orchestrator for MMHCL
==========================================

This is the **entry point** of the MMHCL (Multi-Modal Hypergraph Contrastive
Learning) recommendation system.  It wires together every other module:

    parser.py  ──>  args          (hyperparameters)
    load_data.py ──>  Data        (user-item interactions + multi-modal graphs)
    Models.py  ──>  MMHCL         (the neural network)
    batch_test.py ──>  test_torch (evaluation on val / test splits)
    logging.py ──>  Logger        (file + console logging)

Overall training pipeline (per epoch):
    1.  Sample (user, positive_item, negative_item) triplets  — BPR paradigm
    2.  Forward pass through MMHCL to get embeddings
    3.  Compute total loss  =  BPR loss  +  embedding regularisation
                              +  item contrastive loss  +  user contrastive loss
    4.  Back-propagate and update parameters
    5.  Every ``verbose`` epochs, evaluate on the validation set
    6.  If validation metrics improve, evaluate on the test set and save the best
        model state; otherwise increment the early-stopping counter

Reference:
    "MMHCL: Multi-Modal Hypergraph Contrastive Learning for Recommendation"
    (ACM, 2024)
"""

from __future__ import annotations

from datetime import datetime
import math
import os
import random
import sys
from time import time
from tqdm import tqdm
import json
import copy
import argparse
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.sparse as sparse
import scipy.sparse as sp

from utility.parser import parse_args

from Models import MMHCL

from utility.batch_test import *          # also creates `data_generator` (global)
from utility.logging import Logger
import pathlib

# Type alias for metric result dictionaries
MetricsDict = dict[str, Any]

# ---------------------------------------------------------------------------
# 1.  Parse command-line arguments (see utility/parser.py for all defaults)
# ---------------------------------------------------------------------------
args: argparse.Namespace = parse_args()

# ---------------------------------------------------------------------------
# 2.  Optional Weights & Biases (W&B) integration for experiment tracking
#     Set --use_wandb 1 to enable; requires ``pip install wandb``.
# ---------------------------------------------------------------------------
wandb: Any = None
if args.use_wandb:
    try:
        import wandb as _wandb
        wandb = _wandb
    except ImportError:
        print("[WARNING] wandb not installed. Disabling W&B logging.")
        args.use_wandb = 0


# ---------------------------------------------------------------------------
# 3.  Build a unique experiment directory name from key hyperparameters
#     This ensures every (seed, hyperparameter) combination gets its own
#     folder for logs and model checkpoints.
#
#     Example path_name:
#       "uu_ii=3_2_0.03_0.07_topk=5_t=0.6_regs=0.001_dim=64_seed=42_"
# ---------------------------------------------------------------------------
path_name: str = (
    f"uu_ii={args.User_layers}_{args.Item_layers}"
    f"_{args.user_loss_ratio}_{args.item_loss_ratio}"
    f"_topk={args.topk}_t={args.temperature}"
    f"_regs={args.regs}_dim={args.embed_size}"
    f"_seed={args.seed}_{args.ablation_target}"
)
path: str = f"../{args.dataset}/{path_name}/"          # per-run logs + weights
record_path: str = f"../{args.dataset}/MM/"             # aggregated ablation records
pathlib.Path(f"{path}").mkdir(parents=True, exist_ok=True)
pathlib.Path(f"{record_path}").mkdir(parents=True, exist_ok=True)


# ===========================================================================
#  Trainer — the main class that owns the model, optimizer, and training loop
# ===========================================================================
class Trainer(object):
    """
    Encapsulates:
        - Model initialisation (MMHCL on GPU)
        - Adam optimiser + LambdaLR exponential-decay scheduler
        - Optional ReduceLROnPlateau scheduler
        - BPR (Bayesian Personalised Ranking) loss computation
        - The full train-evaluate loop with early stopping
    """

    def __init__(self, data_config: dict[str, Any]) -> None:
        """
        Args:
            data_config: must contain keys
                'n_users', 'n_items',
                'UI_mat'   — normalised user-item bipartite adjacency (sparse, GPU),
                'User_mat' — user-user co-interaction hypergraph      (sparse, GPU),
                'Item_mat' — item-item multi-modal hypergraph          (sparse, GPU).
        """
        # --- dataset dimensions ---
        self.n_users: int = data_config['n_users']
        self.n_items: int = data_config['n_items']

        # --- logger (writes to both per-run and aggregated directories) ---
        self.logger: Logger = Logger(
            path, is_debug=args.debug, target=path_name,
            path2=record_path, ablation_target=args.ablation_target
        )
        self.logger.logging("PID: %d" % os.getpid())
        self.logger.logging(str(args))

        # --- store frequently-used hyperparameters ---
        self.lr: float = args.lr                           # initial learning rate
        self.emb_dim: int = args.embed_size                # embedding dimensionality
        self.batch_size: int = args.batch_size             # BPR training batch size
        self.weight_size: list[int] = eval(args.weight_size)  # per-layer sizes
        self.n_layers: int = len(self.weight_size)         # number of GNN layers
        self.regs: float = args.regs                       # L2 regularisation coefficient
        self.decay: float = self.regs                      # alias used in bpr_loss()

        # --- move the three adjacency matrices to GPU ---
        self.UI_mat: torch.Tensor = data_config['UI_mat'].cuda()
        self.User_mat: torch.Tensor = data_config['User_mat'].cuda()
        self.Item_mat: torch.Tensor = data_config['Item_mat'].cuda()

        # --- instantiate the MMHCL model and move to GPU ---
        self.model: MMHCL = MMHCL(self.n_users, self.n_items, self.emb_dim)
        self.model = self.model.cuda()

        # --- optimizer and learning-rate schedulers ---
        self.optimizer: optim.Adam = optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler: optim.lr_scheduler.LambdaLR = self.set_lr_scheduler()
        self.reduce_lr_scheduler: Optional[optim.lr_scheduler.ReduceLROnPlateau] = (
            self.set_reduce_lr_scheduler() if args.use_reduce_lr else None
        )

    # -----------------------------------------------------------------------
    #  Learning-rate schedulers
    # -----------------------------------------------------------------------
    def set_lr_scheduler(self) -> optim.lr_scheduler.LambdaLR:
        """
        Exponential decay scheduler:  lr(epoch) = lr_0 * 0.96^(epoch / 50)

        This provides a smooth, gradual decay — after 50 epochs the LR is
        multiplied by 0.96, after 100 epochs by 0.96^2 ≈ 0.922, etc.
        """
        fac = lambda epoch: 0.96 ** (epoch / 50)
        scheduler: optim.lr_scheduler.LambdaLR = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=fac
        )
        return scheduler

    def set_reduce_lr_scheduler(self) -> optim.lr_scheduler.ReduceLROnPlateau:
        """
        ReduceLROnPlateau: automatically halves the LR when the monitored
        validation metric stops improving for ``reduce_lr_patience`` evaluation
        cycles.  This is an *additional* scheduler layered on top of the
        exponential decay.
        """
        scheduler: optim.lr_scheduler.ReduceLROnPlateau = (
            optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=args.early_stopping_mode,
                factor=args.reduce_lr_factor,
                patience=args.reduce_lr_patience,
                min_lr=args.reduce_lr_min,
            )
        )
        return scheduler

    # -----------------------------------------------------------------------
    #  Evaluation helper
    # -----------------------------------------------------------------------
    def test(self, users_to_test: list[int], is_val: bool) -> MetricsDict:
        """
        Run a full evaluation pass (no gradient computation).

        Args:
            users_to_test: list of user IDs to evaluate.
            is_val: True → use validation set; False → use test set.

        Returns:
            dict with keys 'recall', 'precision', 'ndcg', 'hit_ratio', 'auc'.
        """
        self.model.eval()
        with torch.no_grad():
            ua_embeddings: torch.Tensor
            ia_embeddings: torch.Tensor
            ii: torch.Tensor
            uu: torch.Tensor
            ua_embeddings, ia_embeddings, ii, uu = self.model(
                self.UI_mat, self.Item_mat, self.User_mat
            )
        result: MetricsDict = test_torch(
            ua_embeddings, ia_embeddings, users_to_test, is_val
        )
        return result

    # -----------------------------------------------------------------------
    #  Main training loop
    # -----------------------------------------------------------------------
    def train(self) -> None:
        """
        Full training procedure with early stopping and optional W&B logging.
        """
        training_time_list: list[float] = []
        loss_loger: list[float] = []
        pre_loger: list[npt.NDArray[np.floating]] = []
        rec_loger: list[npt.NDArray[np.floating]] = []
        ndcg_loger: list[npt.NDArray[np.floating]] = []
        hit_loger: list[npt.NDArray[np.floating]] = []
        stopping_step: int = 0
        should_stop: bool = False
        cur_best_pre_0: float = 0.

        # n_batch = how many mini-batches per epoch
        n_batch: int = data_generator.n_train // args.batch_size + 1
        best_recall: float = 0.         # best validation recall@20 seen so far
        best_ndcg: float = 0.           # best validation ndcg@20 seen so far
        best_model_state: Optional[dict[str, Any]] = None
        test_ret: Union[str, MetricsDict] = ""

        # ===== Initialise W&B run (if enabled) =====
        if args.use_wandb and wandb is not None:
            wandb_config: dict[str, Any] = vars(args)
            wandb_config['path_name'] = path_name
            wandb_init_kwargs: dict[str, Any] = {
                'project': args.wandb_project,
                'config': wandb_config,
                'reinit': True,
            }
            if args.wandb_entity:
                wandb_init_kwargs['entity'] = args.wandb_entity
            if args.wandb_run_name:
                wandb_init_kwargs['name'] = args.wandb_run_name
            wandb.init(**wandb_init_kwargs)
            self.logger.logging(f"W&B run initialized: {wandb.run.name}")

        # ===== Epoch loop =====
        for epoch in range(args.epoch):
            t1: float = time()
            # Accumulators for epoch-level loss components
            loss: float = 0.
            mf_loss: float = 0.
            emb_loss: float = 0.
            reg_loss: float = 0.
            contrastive_loss: float = 0.
            n_batch = data_generator.n_train // args.batch_size + 1
            sample_time: float = 0.

            # ----- Mini-batch loop within one epoch -----
            for idx in range(n_batch):
                self.model.train()
                self.optimizer.zero_grad()

                # (1) Sample a BPR triplet batch
                sample_t1: float = time()
                users: list[int]
                pos_items: list[int]
                neg_items: list[int]
                users, pos_items, neg_items = data_generator.sample()
                sample_time += time() - sample_t1

                # (2) Forward pass — get all user/item embeddings + hypergraph views
                ua_embeddings: torch.Tensor
                ia_embeddings: torch.Tensor
                ii: torch.Tensor
                uu: torch.Tensor
                ua_embeddings, ia_embeddings, ii, uu = self.model(
                    self.UI_mat, self.Item_mat, self.User_mat
                )

                # Look up embeddings for this batch's users and items
                u_g_embeddings: torch.Tensor = ua_embeddings[users]
                pos_i_g_embeddings: torch.Tensor = ia_embeddings[pos_items]
                neg_i_g_embeddings: torch.Tensor = ia_embeddings[neg_items]

                # (3) BPR loss
                batch_mf_loss: torch.Tensor
                batch_emb_loss: torch.Tensor
                batch_reg_loss: float
                batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(
                    u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
                )

                # (4) Contrastive losses (InfoNCE)
                batch_contrastive_loss1: torch.Tensor = (
                    self.model.batched_contrastive_loss(ia_embeddings, ii)
                )
                batch_contrastive_loss1 *= args.item_loss_ratio

                batch_contrastive_loss2: torch.Tensor = (
                    self.model.batched_contrastive_loss(ua_embeddings, uu)
                )
                batch_contrastive_loss2 *= args.user_loss_ratio

                batch_contrastive_loss: torch.Tensor = (
                    batch_contrastive_loss1 + batch_contrastive_loss2
                )

                # (5) Total loss = BPR + regularisation + contrastive
                batch_loss: torch.Tensor = (
                    batch_mf_loss + batch_emb_loss
                    + batch_reg_loss + batch_contrastive_loss
                )

                # (6) Back-propagation and parameter update
                batch_loss.backward(retain_graph=False)
                self.optimizer.step()

                # Accumulate for epoch-level logging
                loss += float(batch_loss)
                mf_loss += float(batch_mf_loss)
                emb_loss += float(batch_emb_loss)
                reg_loss += float(batch_reg_loss)
                contrastive_loss += float(batch_contrastive_loss)

            # Step the exponential LR decay scheduler (once per epoch)
            self.lr_scheduler.step()

            # Free GPU memory for embeddings computed during training
            del ua_embeddings, ia_embeddings, u_g_embeddings
            del neg_i_g_embeddings, pos_i_g_embeddings, ii, uu

            # ----- NaN guard -----
            if math.isnan(loss):
                self.logger.logging('ERROR: loss is nan.')
                if args.use_wandb and wandb is not None:
                    wandb.finish(exit_code=1)
                sys.exit()

            # ----- W&B: log training loss every epoch -----
            if args.use_wandb and wandb is not None:
                wandb.log({
                    'epoch': epoch,
                    'train/loss': loss,
                    'train/mf_loss': mf_loss,
                    'train/emb_loss': emb_loss,
                    'train/contrastive_loss': contrastive_loss,
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                })

            # ----- Non-evaluation epoch: just log training loss and move on -----
            if (epoch + 1) % args.verbose != 0:
                perf_str: str = (
                    'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (
                        epoch, time() - t1, loss, mf_loss, emb_loss, contrastive_loss
                    )
                )
                training_time_list.append(time() - t1)
                self.logger.logging(perf_str)
                continue

            # ===== Evaluation epoch (every ``verbose`` epochs) =====
            t2: float = time()
            users_to_test: list[int] = list(data_generator.test_set.keys())
            users_to_val: list[int] = list(data_generator.val_set.keys())
            ret: MetricsDict = self.test(users_to_val, is_val=True)
            training_time_list.append(t2 - t1)

            t3: float = time()

            # Store validation metrics for later analysis
            loss_loger.append(loss)
            rec_loger.append(ret['recall'])
            pre_loger.append(ret['precision'])
            ndcg_loger.append(ret['ndcg'])
            hit_loger.append(ret['hit_ratio'])

            # Pretty-print epoch summary
            if args.verbose > 0:
                perf_str = (
                    'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f], '
                    'recall=[%.5f, %.5f], precision=[%.5f, %.5f], '
                    'hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % (
                        epoch, t2 - t1, t3 - t2,
                        loss, mf_loss, emb_loss, contrastive_loss,
                        ret['recall'][0], ret['recall'][-1],
                        ret['precision'][0], ret['precision'][-1],
                        ret['hit_ratio'][0], ret['hit_ratio'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1],
                    )
                )
                self.logger.logging(perf_str)

            # ----- W&B: log validation metrics -----
            if args.use_wandb and wandb is not None:
                wandb.log({
                    'epoch': epoch,
                    'val/recall@10': ret['recall'][0],
                    'val/recall@20': ret['recall'][-1],
                    'val/precision@10': ret['precision'][0],
                    'val/precision@20': ret['precision'][-1],
                    'val/ndcg@10': ret['ndcg'][0],
                    'val/ndcg@20': ret['ndcg'][-1],
                    'val/hit@10': ret['hit_ratio'][0],
                    'val/hit@20': ret['hit_ratio'][-1],
                })

            # ----- ReduceLROnPlateau step (monitors val recall@20) -----
            if self.reduce_lr_scheduler is not None:
                self.reduce_lr_scheduler.step(ret['recall'][-1])

            # ===== Early-stopping logic =====
            improved: bool = False
            if (ret['recall'][1] > best_recall + args.early_stopping_min_delta or
                    ret['ndcg'][1] > best_ndcg + args.early_stopping_min_delta):
                if ret['recall'][1] > best_recall:
                    best_recall = ret['recall'][1]
                if ret['ndcg'][1] > best_ndcg:
                    best_ndcg = ret['ndcg'][1]
                improved = True

            if improved:
                # ---- Improvement found → evaluate on the TEST set ----
                test_ret = self.test(users_to_test, is_val=False)

                self.logger.logging(
                    "Test_Recall@%d: %.8f   Test_Precision@%d: %.8f   "
                    "Test_NDCG@%d: %.8f" % (
                        eval(args.Ks)[1], test_ret['recall'][1],
                        eval(args.Ks)[1], test_ret['precision'][1],
                        eval(args.Ks)[1], test_ret['ndcg'][1],
                    )
                )

                # W&B: log test metrics at this checkpoint
                if args.use_wandb and wandb is not None:
                    wandb.log({
                        'epoch': epoch,
                        'test/recall@20': test_ret['recall'][1],
                        'test/precision@20': test_ret['precision'][1],
                        'test/ndcg@20': test_ret['ndcg'][1],
                        'best_recall': best_recall,
                        'best_ndcg': best_ndcg,
                    })

                stopping_step = 0

                if args.early_stopping_restore_best:
                    best_model_state = copy.deepcopy(self.model.state_dict())

            elif epoch + 1 >= args.early_stopping_min_epochs:
                if stopping_step < args.early_stopping_patience:
                    stopping_step += 1
                    self.logger.logging(
                        '#####Early stopping steps: %d #####' % stopping_step
                    )
                else:
                    self.logger.logging('#####Early stop! #####')
                    if args.early_stopping_restore_best and best_model_state is not None:
                        self.model.load_state_dict(best_model_state)
                        self.logger.logging('Restored best model weights.')
                    fname: str = f'Model.epoch={epoch}.pth'
                    torch.save(self.model.state_dict(), os.path.join(path, fname))
                    break
            else:
                self.logger.logging(
                    f'Epoch {epoch}: no improvement, but '
                    f'min_epochs={args.early_stopping_min_epochs} not reached yet'
                )

        # ===== Post-training: log the BEST test results =====
        if isinstance(test_ret, dict):
            Ks_list: list[int] = eval(args.Ks)
            self.logger.logging(
                "BEST_Test_Recall@%d: %.8f" % (Ks_list[1], test_ret['recall'][1])
            )
            self.logger.logging(
                "BEST_Test_Precision@%d: %.8f" % (Ks_list[1], test_ret['precision'][1])
            )
            self.logger.logging(
                "BEST_Test_NDCG@%d: %.8f" % (Ks_list[1], test_ret['ndcg'][1])
            )

            if args.use_wandb and wandb is not None:
                wandb.summary['best_test_recall@20'] = test_ret['recall'][1]
                wandb.summary['best_test_precision@20'] = test_ret['precision'][1]
                wandb.summary['best_test_ndcg@20'] = test_ret['ndcg'][1]

        self.logger.logging(str(test_ret))
        self.logger.logging_sum(f"{path_name}:{str(test_ret)}")

        if args.use_wandb and wandb is not None:
            wandb.finish()

    # -----------------------------------------------------------------------
    #  BPR (Bayesian Personalised Ranking) Loss
    # -----------------------------------------------------------------------
    def bpr_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """
        Compute the BPR pairwise ranking loss.

        Loss = -mean( log σ(score_pos - score_neg) )  +  λ * L2_regulariser

        Args:
            users     : (batch, emb_dim) — user embeddings for this batch
            pos_items : (batch, emb_dim) — positive item embeddings
            neg_items : (batch, emb_dim) — negative item embeddings

        Returns:
            Tuple of (mf_loss, emb_loss, reg_loss).
        """
        # Predicted scores via dot product: score(u, i) = u^T · i
        pos_scores: torch.Tensor = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores: torch.Tensor = torch.sum(torch.mul(users, neg_items), dim=1)

        # L2 regularisation: penalise large embedding norms
        regularizer: torch.Tensor = (
            1. / 2 * (users ** 2).sum()
            + 1. / 2 * (pos_items ** 2).sum()
            + 1. / 2 * (neg_items ** 2).sum()
        )
        regularizer = regularizer / self.batch_size

        # BPR loss: -log( sigmoid(pos_score - neg_score) )
        maxi: torch.Tensor = F.logsigmoid(pos_scores - neg_scores)
        mf_loss: torch.Tensor = -torch.mean(maxi)

        emb_loss: torch.Tensor = self.decay * regularizer
        reg_loss: float = 0.0
        return mf_loss, emb_loss, reg_loss

    # -----------------------------------------------------------------------
    #  Utility: scipy sparse → torch sparse
    # -----------------------------------------------------------------------
    def sparse_mx_to_torch_sparse_tensor(
        self, sparse_mx: sp.spmatrix
    ) -> torch.Tensor:
        """Convert a scipy sparse matrix to a torch sparse COO tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices: torch.Tensor = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        )
        values: torch.Tensor = torch.from_numpy(sparse_mx.data)
        shape: torch.Size = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape)


# ===========================================================================
#  Reproducibility: fix all random seeds
# ===========================================================================
def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch (CPU + all GPUs)."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)          # CPU
    torch.cuda.manual_seed_all(seed)  # all GPUs


# ===========================================================================
#  Entry point
# ===========================================================================
if __name__ == '__main__':
    # Select the GPU specified by --gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    set_seed(args.seed)

    # --- Prepare the configuration dict for the Trainer ---
    config: dict[str, Any] = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    # User-Item bipartite interaction graph  (symmetrically normalised)
    UI_mat: torch.Tensor = data_generator.get_UI_mat()

    # User-User co-interaction graph:  U2U = R · R^T  (row-normalised)
    User_mat: torch.Tensor = data_generator.get_U2U_mat()

    # --- Item-Item multi-modal hypergraph ---
    Item_mat: torch.Tensor
    if args.dataset == "Tiktok":
        Item_mat = data_generator.get_tiktok_I2I_Hypergraph_mul_mat()
    elif args.dataset in ["Clothing", "Sports"]:
        Item_mat = data_generator.get_I2I_Hypergraph_mul_mat()

    config['UI_mat'] = UI_mat
    config['User_mat'] = User_mat
    config['Item_mat'] = Item_mat

    # --- (Commented out) Single-modality ablation experiment ---
    # Image_item_mat, Text_item_mat, Audio_item_mat = data_generator.get_I2I_single_mat()
    # config['Item_mat'] = Text_item_mat
    # config['Item_mat'] = Audio_item_mat

    # --- Create the Trainer and start training ---
    trainer: Trainer = Trainer(data_config=config)
    trainer.train()
