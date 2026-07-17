"""
Models.py — Neural Network Definitions for MMHCL
==================================================

This file defines the two key model architectures:

1. **LightGCN** — a lightweight Graph Convolutional Network for collaborative
   filtering.  It propagates embeddings on the user-item bipartite graph and
   averages across layers (no feature transformation, no activation).

2. **MMHCL** — Multi-Modal Hypergraph Contrastive Learning model.
   This is the main model that combines:
     - A collaborative filtering backbone (LightGCN / NGCF / MF) operating
       on the user-item interaction graph.
     - Item-item hypergraph convolution (multi-modal similarity graph).
     - User-user hypergraph convolution (co-interaction graph).
     - InfoNCE contrastive loss to align the CF embeddings with the
       hypergraph-derived embeddings.

Architecture overview (MMHCL forward pass):
    ┌──────────────────────────────────────────────────────────────┐
    │  User-Item Bipartite Graph (UI_mat)                          │
    │     → LightGCN layers → u_ui_emb, i_ui_emb                  │
    ├──────────────────────────────────────────────────────────────┤
    │  Item-Item Multi-Modal Hypergraph (I2I_mat)                  │
    │     → GNN layers on ii_embedding → ii_emb                    │
    ├──────────────────────────────────────────────────────────────┤
    │  User-User Co-Interaction Hypergraph (U2U_mat)               │
    │     → GNN layers on uu_embedding → uu_emb                    │
    ├──────────────────────────────────────────────────────────────┤
    │  Fusion:                                                     │
    │     final_item = i_ui_emb + L2_normalise(ii_emb)             │
    │     final_user = u_ui_emb + L2_normalise(uu_emb)             │
    └──────────────────────────────────────────────────────────────┘

    Contrastive alignment (computed in main.py during training):
        InfoNCE(ia_embeddings, ii_emb)  ×  λ_item
        InfoNCE(ua_embeddings, uu_emb)  ×  λ_user

Reference:
    "MMHCL: Multi-Modal Hypergraph Contrastive Learning for Recommendation"
"""

from __future__ import annotations

import os
import argparse
from typing import Callable

import numpy as np
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

from utility.parser import parse_args
from utility.norm import build_sim, build_knn_normalized_graph

args: argparse.Namespace = parse_args()


# ===========================================================================
#  LightGCN — Lightweight Graph Convolutional Network
# ===========================================================================
class LightGCN(nn.Module):
    """
    LightGCN (He et al., SIGIR 2020): simplifies GCN for collaborative
    filtering by removing feature transformation and non-linear activation.

    Forward pass:
        1.  Concatenate user and item ID embeddings into one vector.
        2.  For each layer, perform sparse matrix multiplication with the
            normalised adjacency matrix (message passing).
        3.  Average the embeddings from all layers (including layer 0 = input).
        4.  Split back into user and item embeddings.

    Note: This class is not used directly by MMHCL — the same LightGCN logic
    is inlined inside MMHCL.forward() for the 'LightGCN' branch.  This class
    is kept for standalone usage or ablation experiments.
    """

    def __init__(self, n_users: int, n_items: int, embedding_dim: int) -> None:
        super().__init__()
        self.n_users: int = n_users
        self.n_items: int = n_items
        self.embedding_dim: int = embedding_dim

        # Learnable ID embeddings, initialised with Xavier uniform
        self.user_embedding: nn.Embedding = nn.Embedding(n_users, embedding_dim)
        self.item_id_embedding: nn.Embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

    def forward(self, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            adj: normalised user-item bipartite adjacency matrix
                 shape (n_users + n_items, n_users + n_items), sparse.

        Returns:
            Tuple of (u_g_embeddings, i_g_embeddings):
                u_g_embeddings: (n_users, embedding_dim) — refined user embeddings
                i_g_embeddings: (n_items, embedding_dim) — refined item embeddings
        """
        # Layer 0: raw ID embeddings [users; items]
        ego_embeddings: torch.Tensor = torch.cat(
            (self.user_embedding.weight, self.item_id_embedding.weight), dim=0
        )
        all_embeddings: list[torch.Tensor] = [ego_embeddings]

        # Message-passing layers: each layer is a single sparse mat-mul
        # adj · E  =  neighbourhood aggregation (no weights, no activation)
        for i in range(args.UI_layers):
            side_embeddings: torch.Tensor = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]

        # Mean pooling across all layers (0, 1, ..., L)
        all_embeddings_stacked: torch.Tensor = torch.stack(all_embeddings, dim=1)
        all_embeddings_mean: torch.Tensor = all_embeddings_stacked.mean(
            dim=1, keepdim=False
        )

        # Split the concatenated embedding back into user / item parts
        u_g_embeddings: torch.Tensor
        i_g_embeddings: torch.Tensor
        u_g_embeddings, i_g_embeddings = torch.split(
            all_embeddings_mean, [self.n_users, self.n_items], dim=0
        )
        return u_g_embeddings, i_g_embeddings


# ===========================================================================
#  MMHCL — Multi-Modal Hypergraph Contrastive Learning
# ===========================================================================
class MMHCL(nn.Module):
    """
    The main MMHCL model.

    It maintains FOUR sets of learnable embeddings:
        user_ui_embedding  — user embeddings for the UI bipartite graph
        item_ui_embedding  — item embeddings for the UI bipartite graph
        uu_embedding       — user embeddings for the U2U hypergraph
        ii_embedding       — item embeddings for the I2I multi-modal hypergraph

    The forward pass produces two "views" of each entity:
        - CF view:         from LightGCN / NGCF / MF on the UI graph
        - Hypergraph view: from GNN propagation on U2U / I2I graphs

    These two views are fused (added) and also used separately in the
    contrastive loss to encourage cross-view agreement.
    """

    def __init__(self, n_users: int, n_items: int, embedding_dim: int) -> None:
        super(MMHCL, self).__init__()
        self.n_users: int = n_users
        self.n_items: int = n_items
        self.embeddings_dim: int = embedding_dim

        # ---- CF branch embeddings (for the user-item bipartite graph) ----
        self.user_ui_embedding: nn.Embedding = nn.Embedding(n_users, self.embeddings_dim)
        self.item_ui_embedding: nn.Embedding = nn.Embedding(n_items, self.embeddings_dim)

        # ---- Hypergraph branch embeddings (separate from CF embeddings) ----
        # uu_embedding : propagated on the user-user co-interaction graph
        # ii_embedding : propagated on the item-item multi-modal hypergraph
        self.uu_embedding: nn.Embedding = nn.Embedding(n_users, self.embeddings_dim)
        self.ii_embedding: nn.Embedding = nn.Embedding(n_items, self.embeddings_dim)

        # ---- Optional NGCF layers (only used if cf_model == 'NGCF') ----
        # NGCF adds linear transformations + bi-linear interactions per layer
        if args.cf_model == 'NGCF':
            self.GC_Linear_list: nn.ModuleList = nn.ModuleList()
            self.Bi_Linear_list: nn.ModuleList = nn.ModuleList()
            self.dropout_list: nn.ModuleList = nn.ModuleList()
            weight_sizes: list[int] = eval(args.weight_size)
            for i in range(args.UI_layers):
                self.GC_Linear_list.append(
                    nn.Linear(weight_sizes[i], weight_sizes[i + 1])
                )
                self.Bi_Linear_list.append(
                    nn.Linear(weight_sizes[i], weight_sizes[i + 1])
                )
                self.dropout_list.append(nn.Dropout(0.1))

        # Xavier uniform initialisation for all embedding tables
        nn.init.xavier_uniform_(self.user_ui_embedding.weight)
        nn.init.xavier_uniform_(self.item_ui_embedding.weight)
        nn.init.xavier_uniform_(self.uu_embedding.weight)
        nn.init.xavier_uniform_(self.ii_embedding.weight)

        # InfoNCE temperature parameter (τ) — controls the sharpness of
        # the softmax in the contrastive loss.  Lower τ → sharper distribution.
        self.tau: float = args.temperature

    def forward(
        self,
        UI_mat: torch.Tensor,
        I2I_mat: torch.Tensor,
        U2U_mat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass producing both CF and hypergraph embeddings.

        Args:
            UI_mat  : sparse (n_users+n_items, n_users+n_items) — user-item graph
            I2I_mat : sparse (n_items, n_items) — item-item multi-modal hypergraph
            U2U_mat : sparse (n_users, n_users) — user-user co-interaction graph

        Returns:
            Tuple of (u_ui_emb, i_ui_emb, ii_emb, uu_emb):
                u_ui_emb : (n_users, emb_dim) — final user embeddings (CF + hypergraph)
                i_ui_emb : (n_items, emb_dim) — final item embeddings (CF + hypergraph)
                ii_emb   : (n_items, emb_dim) — item hypergraph-only embeddings
                uu_emb   : (n_users, emb_dim) — user hypergraph-only embeddings

        The ii_emb and uu_emb are used in the contrastive loss (in main.py).
        """
        # =====================================================================
        # (A) Hypergraph branch: propagate through I2I and U2U graphs
        # =====================================================================
        ii_emb: torch.Tensor = self.ii_embedding.weight   # (n_items, emb_dim)
        uu_emb: torch.Tensor = self.uu_embedding.weight   # (n_users, emb_dim)

        # Item hypergraph propagation: stack `Item_layers` GNN layers
        if args.item_loss_ratio != 0:
            for i in range(args.Item_layers):
                ii_emb = torch.sparse.mm(I2I_mat, ii_emb)

        # User hypergraph propagation: stack `User_layers` GNN layers
        if args.user_loss_ratio != 0:
            for i in range(args.User_layers):
                uu_emb = torch.sparse.mm(U2U_mat, uu_emb)

        # =====================================================================
        # (B) CF branch: operate on the user-item bipartite graph
        # =====================================================================
        u_ui_emb: torch.Tensor
        i_ui_emb: torch.Tensor

        if args.cf_model == 'LightGCN':
            # ----- LightGCN: no weights, no activation -----
            ego_embeddings: torch.Tensor = torch.cat(
                (self.user_ui_embedding.weight, self.item_ui_embedding.weight), dim=0
            )
            all_embeddings: list[torch.Tensor] = [ego_embeddings]

            for i in range(args.UI_layers):
                side_embeddings: torch.Tensor = torch.sparse.mm(UI_mat, ego_embeddings)
                ego_embeddings = side_embeddings
                all_embeddings += [ego_embeddings]

            all_embeddings_stacked: torch.Tensor = torch.stack(all_embeddings, dim=1)
            all_embeddings_mean: torch.Tensor = all_embeddings_stacked.mean(
                dim=1, keepdim=False
            )
            u_ui_emb, i_ui_emb = torch.split(
                all_embeddings_mean, [self.n_users, self.n_items], dim=0
            )

        elif args.cf_model == 'NGCF':
            # ----- NGCF: adds feature transformation + bi-linear interaction -----
            ego_embeddings = torch.cat(
                (self.user_ui_embedding.weight, self.item_ui_embedding.weight), dim=0
            )
            all_embeddings = [ego_embeddings]

            for i in range(args.UI_layers):
                side_embeddings = torch.sparse.mm(UI_mat, ego_embeddings)
                sum_embeddings: torch.Tensor = F.leaky_relu(
                    self.GC_Linear_list[i](side_embeddings)
                )
                bi_embeddings: torch.Tensor = torch.mul(ego_embeddings, side_embeddings)
                bi_embeddings = F.leaky_relu(self.Bi_Linear_list[i](bi_embeddings))
                ego_embeddings = sum_embeddings + bi_embeddings
                ego_embeddings = self.dropout_list[i](ego_embeddings)
                norm_embeddings: torch.Tensor = F.normalize(ego_embeddings, p=2, dim=1)
                all_embeddings += [norm_embeddings]

            all_embeddings_stacked = torch.stack(all_embeddings, dim=1)
            all_embeddings_mean = all_embeddings_stacked.mean(dim=1, keepdim=False)
            u_ui_emb, i_ui_emb = torch.split(
                all_embeddings_mean, [self.n_users, self.n_items], dim=0
            )

        elif args.cf_model == 'MF':
            # ----- Matrix Factorisation baseline: no graph propagation -----
            u_ui_emb = self.user_ui_embedding.weight
            i_ui_emb = self.item_ui_embedding.weight

        # =====================================================================
        # (C) Fusion: add L2-normalised hypergraph embeddings to CF embeddings
        # =====================================================================
        if args.item_loss_ratio != 0:
            i_ui_emb = i_ui_emb + F.normalize(ii_emb, p=2, dim=1)

        if args.user_loss_ratio != 0:
            u_ui_emb = u_ui_emb + F.normalize(uu_emb, p=2, dim=1)

        return u_ui_emb, i_ui_emb, ii_emb, uu_emb

    # -----------------------------------------------------------------------
    #  Contrastive Loss (InfoNCE)
    # -----------------------------------------------------------------------
    def batched_contrastive_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        batch_size: int = 4096,
    ) -> torch.Tensor:
        """
        Compute the InfoNCE contrastive loss between two embedding matrices.

        Args:
            z1         : (N, emb_dim) — embeddings from view 1 (e.g. CF branch)
            z2         : (N, emb_dim) — embeddings from view 2 (e.g. hypergraph branch)
            batch_size : number of rows to process at once (default 4096)

        Returns:
            Scalar mean InfoNCE loss (torch.Tensor).
        """
        device: torch.device = z1.device
        num_nodes: int = z1.size(0)
        num_batches: int = (num_nodes - 1) // batch_size + 1

        # Temperature-scaled exponential: f(x) = exp(x / τ)
        f: Callable[[torch.Tensor], torch.Tensor] = lambda x: torch.exp(x / self.tau)
        indices: torch.Tensor = torch.arange(0, num_nodes).to(device)
        losses: list[torch.Tensor] = []

        for i in range(num_batches):
            # Select a batch of anchor nodes
            mask: torch.Tensor = indices[i * batch_size:(i + 1) * batch_size]

            # refl_sim[a, b] = exp( sim(z1[anchor_a], z1[b]) / τ )
            refl_sim: torch.Tensor = f(self.sim(z1[mask], z1))      # (B, N)

            # between_sim[a, b] = exp( sim(z1[anchor_a], z2[b]) / τ )
            between_sim: torch.Tensor = f(self.sim(z1[mask], z2))    # (B, N)

            # InfoNCE: positive pair is the diagonal (same index in z2)
            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())
            ))

        loss_vec: torch.Tensor = torch.cat(losses)
        return loss_vec.mean()

    def sim(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Cosine similarity between two sets of vectors.

        Args:
            z1 : (B, D) — first set of vectors
            z2 : (N, D) — second set of vectors

        Returns:
            (B, N) matrix of cosine similarities.
        """
        z1 = F.normalize(z1)   # L2-normalise each row
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())
