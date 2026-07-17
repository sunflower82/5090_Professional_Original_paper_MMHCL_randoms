"""
parser.py — Command-Line Argument Definitions
===============================================

Centralises every hyperparameter and configuration flag used by MMHCL.
All other modules import `parse_args()` to access these values.

Usage examples:
    # Train on Clothing with default settings
    python main.py --dataset Clothing

    # Train with a specific seed and W&B logging
    python main.py --dataset Clothing --seed 42 --use_wandb 1

    # Override key hyperparameters
    python main.py --dataset Sports --batch_size 512 --epoch 500 --lr 0.001
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Parse and return all MMHCL command-line arguments."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="MMHCL: Multi-Modal Hypergraph Contrastive Learning "
                    "for Recommendation"
    )

    # =====================================================================
    #  General / Data
    # =====================================================================
    parser.add_argument('--data_path', nargs='?', default='../data/',
                        help='Root path to all dataset folders.')
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed for reproducibility (Python, NumPy, PyTorch).')
    parser.add_argument('--dataset', nargs='?', default='Tiktok',
                        help='Dataset name: {Tiktok, Sports, Clothing}. '
                             'Must match a folder under data_path.')

    # =====================================================================
    #  Training Schedule
    # =====================================================================
    parser.add_argument('--verbose', type=int, default=5,
                        help='Run validation every N epochs (e.g. 5 → evaluate '
                             'at epoch 4, 9, 14, …).')
    parser.add_argument('--epoch', type=int, default=250,
                        help='Maximum number of training epochs '
                             '(250 fits with original paper).')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Number of BPR triplets per mini-batch '
                             '(1024 fits with original paper).')
    parser.add_argument('--regs', type=float, default=1e-3,
                        help='L2 regularisation coefficient for BPR embedding loss.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Initial learning rate for Adam optimiser.')
    parser.add_argument('--train_dir', default='train')

    # =====================================================================
    #  Model Architecture
    # =====================================================================
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Dimensionality of user/item embeddings.')
    parser.add_argument('--weight_size', nargs='?', default='[64,64,64]',
                        help='Output sizes of each GNN layer (as a Python list string). '
                             'Only used by NGCF backbone.')
    parser.add_argument('--core', type=int, default=5,
                        help='K-core filtering threshold. '
                             '5 = warm-start (remove users/items with < 5 interactions); '
                             '0 = cold-start (keep everything).')
    parser.add_argument('--topk', type=int, default=5,
                        help='Number of nearest neighbours to keep in the k-NN '
                             'sparsification of the modality similarity graphs.')
    parser.add_argument('--cf_model', nargs='?', default='LightGCN',
                        help='Collaborative filtering backbone: '
                             '{MF, NGCF, LightGCN}. LightGCN is recommended.')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                        help='Number of consecutive non-improving evaluation '
                             'cycles before stopping training.')

    parser.add_argument('--sparse', type=int, default=0,
                        help='1 = use sparse adjacency matrices for the UI graph; '
                             '0 = use dense (default).')
    parser.add_argument('--debug', default="True",
                        help='If "True", enable file logging in addition to console output.')

    parser.add_argument('--norm_type', nargs='?', default='sym',
                        help='Adjacency matrix normalisation type: '
                             '{sym, rw, origin}. '
                             'sym = symmetric (D^-½ A D^-½); '
                             'rw  = random-walk (D^-1 A).')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='Index of the GPU to use (for multi-GPU machines).')

    # =====================================================================
    #  Evaluation
    # =====================================================================
    parser.add_argument('--Ks', nargs='?', default='[10,20]',
                        help='Values of K for Recall@K, Precision@K, NDCG@K, Hit@K '
                             '(as a Python list string, e.g. "[10,20]").')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='{part, full}. '
                             'part = mini-batch evaluation (faster, uses heapq for top-K); '
                             'full = full-sort evaluation (slower, also computes AUC).')

    # =====================================================================
    #  MMHCL-specific: Hypergraph & Contrastive Learning
    # =====================================================================
    parser.add_argument('--UI_layers', type=int, default=2,
                        help='Number of GNN layers on the user-item bipartite graph '
                             '(LightGCN / NGCF message-passing depth).')
    parser.add_argument('--User_layers', type=int, default=3,
                        help='Number of GNN layers on the user-user co-interaction '
                             'hypergraph.')
    parser.add_argument('--Item_layers', type=int, default=2,
                        help='Number of GNN layers on the item-item multi-modal '
                             'hypergraph.')
    parser.add_argument('--user_loss_ratio', type=float, default=0.03,
                        help='Weight (λ_user) for the user-side contrastive loss. '
                             'Set to 0 to disable user hypergraph branch entirely.')
    parser.add_argument('--item_loss_ratio', type=float, default=0.07,
                        help='Weight (λ_item) for the item-side contrastive loss. '
                             'Set to 0 to disable item hypergraph branch entirely.')
    parser.add_argument('--temperature', type=float, default=0.6,
                        help='Temperature (τ) for InfoNCE contrastive loss. '
                             'Lower values produce sharper similarity distributions.')

    parser.add_argument('--ablation_target', type=str, default="",
                        help='Tag for ablation experiments (appears in log folder name). '
                             'Leave empty for standard training.')

    # =====================================================================
    #  Enhanced Early Stopping
    # =====================================================================
    parser.add_argument('--early_stopping_min_epochs', type=int, default=0,
                        help='Minimum epochs before early stopping can trigger. '
                             'During warm-up the model may not improve every cycle.')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.0001,
                        help='Minimum metric improvement to count as progress. '
                             'Prevents premature stopping on tiny fluctuations.')
    parser.add_argument('--early_stopping_monitor', type=str, default='val_recall@20',
                        help='Metric name to monitor (informational; the actual '
                             'logic in main.py checks recall@20 and ndcg@20).')
    parser.add_argument('--early_stopping_mode', type=str, default='max',
                        help='{max, min}. "max" means higher is better (for recall/ndcg).')
    parser.add_argument('--early_stopping_restore_best', type=int, default=1,
                        help='1 = restore the best model weights when early stopping '
                             'triggers; 0 = keep the last model weights.')
    parser.add_argument('--adaptive_patience', type=int, default=0,
                        help='1 = enable adaptive patience (scales patience by dataset '
                             'size); 0 = disabled.')

    # =====================================================================
    #  ReduceLROnPlateau Scheduler
    # =====================================================================
    parser.add_argument('--use_reduce_lr', type=int, default=0,
                        help='1 = enable ReduceLROnPlateau (in addition to the '
                             'exponential decay scheduler); 0 = disabled.')
    parser.add_argument('--reduce_lr_factor', type=float, default=0.5,
                        help='Factor by which LR is multiplied when plateau is detected.')
    parser.add_argument('--reduce_lr_patience', type=int, default=3,
                        help='Number of non-improving evaluation cycles before reducing LR.')
    parser.add_argument('--reduce_lr_min', type=float, default=1e-6,
                        help='Minimum learning rate (floor for the scheduler).')

    # =====================================================================
    #  Weights & Biases (W&B) Experiment Tracking
    # =====================================================================
    parser.add_argument('--use_wandb', type=int, default=0,
                        help='1 = enable W&B logging; 0 = disabled. '
                             'Requires `pip install wandb` and `wandb login`.')
    parser.add_argument('--wandb_project', type=str, default='mmhcl',
                        help='W&B project name (creates one if it does not exist).')
    parser.add_argument('--wandb_entity', type=str, default='',
                        help='W&B entity (team or username). '
                             'Leave empty to use your default entity.')
    parser.add_argument('--wandb_run_name', type=str, default='',
                        help='Custom name for this W&B run. '
                             'Leave empty for an auto-generated name.')

    return parser.parse_args()
