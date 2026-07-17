"""
load_data.py — Data Loading & Graph Construction
==================================================

This module is responsible for:
  1. Reading train / val / test JSON split files.
  2. Building the sparse user-item interaction matrix  R  (n_users × n_items).
  3. Constructing the three graph structures required by MMHCL:
       • UI_mat   : user-item bipartite graph      (symmetrically normalised)
       • User_mat : user-user co-interaction graph  (row / random-walk normalised)
       • Item_mat : item-item multi-modal hypergraph (H · H^T, sym-normalised)
  4. Providing the BPR sampling function ``sample()`` used during training.

Data directory layout expected:
    data/<dataset>/
        5-core/
            train.json   — {uid: [iid, ...], ...}
            val.json
            test.json
        image_feat.npy   — (n_items, image_dim)  e.g. 4096-d from ResNet
        text_feat.npy    — (n_items, text_dim)   e.g. 768-d  from BERT
        audio_feat.npy   — (n_items, audio_dim)  [Tiktok only]

Graph construction pipeline for the Item-Item Hypergraph:
    For each modality m ∈ {image, text, [audio]}:
        1. Load feature matrix  F_m  ∈ R^{n_items × d_m}
        2. Compute cosine similarity:  S_m = norm(F_m) · norm(F_m)^T
        3. k-NN sparsification: keep only top-k values per row → A_m
    Concatenate:    H = [A_image | A_text | ...]      (the incidence matrix)
    Hypergraph:     Item_mat = H · H^T                (normalised by D^{-½})
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np
import numpy.typing as npt
import random as rd
import scipy.sparse as sp
from time import time
import json
from utility.parser import parse_args
import argparse

args: argparse.Namespace = parse_args()

import torch


def _torch_load(path: str) -> Any:
    """
    Compatibility wrapper for ``torch.load()``.

    PyTorch 2.6+ defaults to ``weights_only=True``, which breaks loading of
    sparse tensors and other pickled objects.  We explicitly set
    ``weights_only=False`` to maintain backward compatibility.
    """
    return torch.load(path, weights_only=False)


# ===========================================================================
#  Data — dataset loading, graph building, and BPR sampling
# ===========================================================================
class Data(object):
    """
    Loads a recommendation dataset and builds all required adjacency matrices.

    On first run, matrices are computed from raw features and cached as .pth
    files.  Subsequent runs load the cached versions for faster startup.

    Attributes:
        n_users     : dataset user count
        n_items     : dataset item count
        n_train     : number of training interactions
        n_test      : number of test interactions
        n_val       : number of validation interactions
        R           : scipy.sparse.dok_matrix — user-item interaction matrix (binary)
        train_items : training interactions per user
        test_set    : test interactions per user
        val_set     : validation interactions per user
        exist_users : user IDs that have at least one training interaction
    """

    def __init__(self, path: str, batch_size: int) -> None:
        """
        Args:
            path       : root path to the dataset (e.g. '../data/Clothing')
            batch_size : number of users to sample per training batch
        """
        self.path: str = path + '/%d-core' % args.core
        self.batch_size: int = batch_size

        train_file: str = path + '/%d-core/train.json' % (args.core)
        val_file: str = path + '/%d-core/val.json' % (args.core)
        test_file: str = path + '/%d-core/test.json' % (args.core)

        # --- Count users, items, and interactions from JSON files ---
        self.n_users: int = 0
        self.n_items: int = 0
        self.n_train: int = 0
        self.n_test: int = 0
        self.n_val: int = 0
        self.neg_pools: dict[int, list[int]] = {}

        self.exist_users: list[int] = []   # users with ≥ 1 training interaction

        train: dict[str, list[int]] = json.load(open(train_file))
        test: dict[str, list[int]] = json.load(open(test_file))
        val: dict[str, list[int]] = json.load(open(val_file))

        # Scan training data to determine n_users, n_items, and n_train
        for uid_str, items in train.items():
            if len(items) == 0:
                continue
            uid: int = int(uid_str)
            self.exist_users.append(uid)
            self.n_items = max(self.n_items, max(items))
            self.n_users = max(self.n_users, uid)
            self.n_train += len(items)

        # Scan test data
        for uid_str, items in test.items():
            uid = int(uid_str)
            try:
                self.n_items = max(self.n_items, max(items))
                self.n_test += len(items)
            except:
                continue

        # Scan validation data
        for uid_str, items in val.items():
            uid = int(uid_str)
            try:
                self.n_items = max(self.n_items, max(items))
                self.n_val += len(items)
            except:
                continue

        # +1 because IDs are 0-indexed
        self.n_items += 1
        self.n_users += 1

        self.print_statistics()

        # --- Build the binary user-item interaction matrix R ---
        # R[u, i] = 1 if user u interacted with item i in training
        self.R: sp.dok_matrix = sp.dok_matrix(
            (self.n_users, self.n_items), dtype=np.float32
        )
        self.R_Item_Interacts: sp.dok_matrix = sp.dok_matrix(
            (self.n_items, self.n_items), dtype=np.float32
        )

        self.train_items: dict[int, list[int]] = {}
        self.test_set: dict[int, list[int]] = {}
        self.val_set: dict[int, list[int]] = {}

        # Populate R and train_items from training data
        for uid_str, train_items_list in train.items():
            if len(train_items_list) == 0:
                continue
            uid = int(uid_str)
            for idx, i in enumerate(train_items_list):
                self.R[uid, i] = 1.
            self.train_items[uid] = train_items_list

        # Populate test_set
        for uid_str, test_items_list in test.items():
            uid = int(uid_str)
            if len(test_items_list) == 0:
                continue
            try:
                self.test_set[uid] = test_items_list
            except:
                continue

        # Populate val_set
        for uid_str, val_items_list in val.items():
            uid = int(uid_str)
            if len(val_items_list) == 0:
                continue
            try:
                self.val_set[uid] = val_items_list
            except:
                continue

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

    def print_statistics(self) -> None:
        """Print basic dataset statistics to the console."""
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (
            self.n_train, self.n_test,
            (self.n_train + self.n_test) / (self.n_users * self.n_items)
        ))

    # ===================================================================
    #  BPR Sampling
    # ===================================================================
    def sample(self) -> tuple[list[int], list[int], list[int]]:
        """
        Sample a batch of BPR (Bayesian Personalised Ranking) triplets.

        For each sampled user:
          - Draw 1 positive item (an item the user interacted with)
          - Draw 1 negative item (a random item the user did NOT interact with)

        Returns:
            Tuple of (users, pos_items, neg_items), each a list of length batch_size.
        """
        # Sample batch_size users (without replacement if possible)
        users: list[int]
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u: int, num: int) -> list[int]:
            """Randomly sample ``num`` distinct positive items for user u."""
            pos_items: list[int] = self.train_items[u]
            n_pos_items: int = len(pos_items)
            pos_batch: list[int] = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id: int = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id: int = pos_items[pos_id]
                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u: int, num: int) -> list[int]:
            """Randomly sample ``num`` distinct negative items for user u."""
            neg_items: list[int] = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id: int = np.random.randint(low=0, high=self.n_items, size=1)[0]
                # Ensure the negative item is truly negative (not in training set)
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u: int, num: int) -> list[int]:
            """Sample negatives from a pre-computed pool (unused by default)."""
            neg_items: list[int] = list(
                set(self.neg_pools[u]) - set(self.train_items[u])
            )
            return rd.sample(neg_items, num)

        pos_items: list[int] = []
        neg_items: list[int] = []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)    # 1 positive per user
            neg_items += sample_neg_items_for_u(u, 1)    # 1 negative per user
        return users, pos_items, neg_items

    # ===================================================================
    #  Adjacency Matrix Construction (legacy format from LightGCN/LATTICE)
    # ===================================================================
    def get_adj_mat(self) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
        """
        Load or create the user-item adjacency matrix in three forms:
          - adj_mat      : raw bipartite adjacency (with self-loops)
          - norm_adj_mat : row-normalised (D^{-1} · (A + I))
          - mean_adj_mat : row-normalised (D^{-1} · A)

        These are cached as .npz files for fast reloading.
        Note: MMHCL uses ``get_UI_mat()`` instead; this method is kept for
        compatibility with the original LATTICE codebase.
        """
        try:
            t1: float = time()
            adj_mat: sp.spmatrix = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat: sp.spmatrix = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat: sp.spmatrix = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)
        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
        return adj_mat, norm_adj_mat, mean_adj_mat

    def create_adj_mat(self) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
        """
        Build the (n_users + n_items) × (n_users + n_items) bipartite adjacency:
            A = [ 0   R  ]
                [ R^T 0  ]
        Then normalise it with D^{-1} (row normalisation).
        """
        t1: float = time()
        adj_mat: sp.spmatrix = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items),
            dtype=np.float32
        )
        adj_mat = adj_mat.tolil()
        R: sp.lil_matrix = self.R.tolil()

        # Fill in the bipartite structure
        adj_mat[:self.n_users, self.n_users:] = R       # user → item
        adj_mat[self.n_users:, :self.n_users] = R.T     # item → user
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2: float = time()

        def normalized_adj_single(adj: sp.spmatrix) -> sp.coo_matrix:
            """Row-normalise: D^{-1} · A"""
            rowsum: npt.NDArray[np.floating] = np.array(adj.sum(1))
            d_inv: npt.NDArray[np.floating] = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv: sp.dia_matrix = sp.diags(d_inv)
            norm_adj: sp.spmatrix = d_mat_inv.dot(adj)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def get_D_inv(adj: sp.spmatrix) -> sp.dia_matrix:
            """Compute the inverse degree matrix D^{-1}."""
            rowsum: npt.NDArray[np.floating] = np.array(adj.sum(1))
            d_inv: npt.NDArray[np.floating] = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv: sp.dia_matrix = sp.diags(d_inv)
            return d_mat_inv

        def check_adj_if_equal(adj: sp.spmatrix) -> npt.NDArray[np.floating]:
            """Debug helper: verify normalisation by brute-force dense computation."""
            dense_A: npt.NDArray[np.floating] = np.array(adj.todense())
            degree: npt.NDArray[np.floating] = np.sum(dense_A, axis=1, keepdims=False)
            temp: npt.NDArray[np.floating] = np.dot(
                np.diag(np.power(degree, -1)), dense_A
            )
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        # A + I  (self-loops) then normalise
        norm_adj_mat: sp.coo_matrix = normalized_adj_single(
            adj_mat + sp.eye(adj_mat.shape[0])
        )
        # A without self-loops, normalised
        mean_adj_mat: sp.coo_matrix = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    # ===================================================================
    #  Dense Normalisation Utility
    # ===================================================================
    def norm_dense(
        self, adj: torch.Tensor, normalization: str = 'origin'
    ) -> torch.Tensor:
        """
        Normalise a dense adjacency matrix.

        Args:
            adj           : square adjacency matrix
            normalization :
                'sym'    : symmetric normalisation   D^{-½} A D^{-½}
                '2sym'   : asymmetric sym normalisation D_row^{-½} A D_col^{-½}
                'rw'     : random-walk normalisation D^{-1} A
                'origin' : no normalisation (return as-is)

        Returns:
            Normalised adjacency matrix (dense torch.Tensor).
        """
        L_norm: torch.Tensor

        if normalization == 'sym':
            # Symmetric normalisation: D^{-1/2} · A · D^{-1/2}
            rowsum: torch.Tensor = torch.sum(adj, -1)
            d_inv_sqrt: torch.Tensor = torch.pow(rowsum, -0.5)
            d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt: torch.Tensor = torch.diagflat(d_inv_sqrt)
            L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

        elif normalization == "2sym":
            # Asymmetric symmetric normalisation (for non-square or directed):
            # D_row^{-1/2} · A · D_col^{-1/2}
            rowsum = torch.sum(adj, -1)
            d_row_inv_sqrt: torch.Tensor = torch.pow(rowsum, -0.5)
            d_row_inv_sqrt[torch.isinf(d_row_inv_sqrt)] = 0.
            d_row_mat_inv_sqrt: torch.Tensor = torch.diagflat(d_row_inv_sqrt)

            colsum: torch.Tensor = torch.sum(adj, -2)
            d_col_inv_sqrt: torch.Tensor = torch.pow(colsum, -0.5)
            d_col_inv_sqrt[torch.isinf(d_col_inv_sqrt)] = 0.
            d_col_mat_inv_sqrt: torch.Tensor = torch.diagflat(d_col_inv_sqrt)

            L_norm = torch.mm(
                torch.mm(d_row_mat_inv_sqrt, adj), d_col_mat_inv_sqrt
            )

        elif normalization == 'rw':
            # Random-walk normalisation: D^{-1} · A
            rowsum = torch.sum(adj, -1)
            d_inv: torch.Tensor = torch.pow(rowsum, -1)
            d_inv[torch.isinf(d_inv)] = 0.
            d_mat_inv: torch.Tensor = torch.diagflat(d_inv)
            L_norm = torch.mm(d_mat_inv, adj)

        elif normalization == 'origin':
            # No normalisation
            L_norm = adj

        return L_norm

    # ===================================================================
    #  User-Item Bipartite Graph
    # ===================================================================
    def get_UI_mat(self, norm_type: str = 'sym') -> torch.Tensor:
        """
        Build or load the user-item bipartite adjacency matrix.

        Shape: (n_users + n_items) × (n_users + n_items), sparse.
        """
        print("Loading UI_mat:(" + norm_type + ")")
        t: float = time()
        UI_mat: torch.Tensor
        try:
            UI_mat = _torch_load(self.path + '/UI_mat_' + norm_type + ".pth")
        except Exception:
            # First run: build from scratch
            adj_mat_sp: sp.spmatrix = sp.dok_matrix(
                (self.n_users + self.n_items, self.n_users + self.n_items),
                dtype=np.float32
            )
            adj_mat_lil: sp.lil_matrix = adj_mat_sp.tolil()
            R: sp.lil_matrix = self.R.tolil()
            adj_mat_lil[:self.n_users, self.n_users:] = R       # user → item
            adj_mat_lil[self.n_users:, :self.n_users] = R.T     # item → user
            adj_mat_dense: npt.NDArray[np.float32] = np.asarray(adj_mat_lil.todense())
            UI_mat = torch.from_numpy(adj_mat_dense).float()
            UI_mat = self.norm_dense(UI_mat, norm_type)
            UI_mat = UI_mat.to_sparse()
            torch.save(UI_mat, self.path + '/UI_mat_' + norm_type + ".pth")
        print("End Load UI_mat:[%.1fs](" % (time() - t) + norm_type + ")")
        return UI_mat

    def get_UI_single_mat(self, norm_type: str = '2sym') -> torch.Tensor:
        """
        Build or load the raw user-item interaction matrix R (NOT bipartite).

        Shape: (n_users × n_items), sparse.
        """
        print("Loading UI_single_mat:(" + norm_type + ")")
        t: float = time()
        UI_mat: torch.Tensor
        try:
            UI_mat = _torch_load(self.path + '/UI_single_mat_' + norm_type + ".pth")
        except Exception:
            adj_mat_dense: npt.NDArray[np.float32] = np.asarray(self.R.todense())
            UI_mat = torch.from_numpy(adj_mat_dense).float()
            UI_mat = self.norm_dense(UI_mat, norm_type)
            UI_mat = UI_mat.to_sparse()
            torch.save(UI_mat, self.path + '/UI_single_mat_' + norm_type + ".pth")
        print("End Load UI_single_mat:[%.1fs](" % (time() - t) + norm_type + ")")
        return UI_mat

    # ===================================================================
    #  User-User Co-Interaction Graph
    # ===================================================================
    def get_U2U_mat(self, norm_type: str = 'rw') -> torch.Tensor:
        """
        Build or load the user-user co-interaction graph.

        Construction: User_mat = R · R^T,  diagonal zeroed out.
        Shape: (n_users × n_users), sparse.
        """
        print("Loading User_mat:(" + norm_type + ")")
        t: float = time()
        User_mat: torch.Tensor
        try:
            User_mat = _torch_load(self.path + '/User_mat_' + norm_type + ".pth")
        except Exception:
            R: torch.Tensor = torch.from_numpy(
                np.asarray(self.R.todense())
            ).float()
            # Co-interaction: User_mat[u1, u2] = number of shared items
            User_mat = R @ R.T
            n_user: int = User_mat.size()[0]
            mask: torch.Tensor = torch.eye(n_user)
            User_mat[mask > 0] = 0   # Remove self-connections
            User_mat = self.norm_dense(User_mat, norm_type)
            User_mat = User_mat.to_sparse()
            torch.save(User_mat, self.path + '/User_mat_' + norm_type + ".pth")
        print("End Load User_mat:[%.1fs](" % (time() - t) + norm_type + ")")
        return User_mat

    # ===================================================================
    #  Item-Item Single-Modality Graphs (for ablation experiments)
    # ===================================================================
    def get_I2I_single_mat(
        self, norm_type: str = "sym"
    ) -> tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, str]]:
        """
        Build or load individual per-modality item-item similarity graphs.

        Returns:
            Tuple of (image_adj, text_adj, audio_adj_or_empty_string).
        """
        print("Loading I2I media-specific mat:(" + norm_type + ")")
        t: float = time()
        image_adj: torch.Tensor
        text_adj: torch.Tensor
        audio_adj: torch.Tensor
        try:
            image_adj = _torch_load(self.path + '/Image_mat_' + norm_type + ".pth")
            text_adj = _torch_load(self.path + '/Text_mat_' + norm_type + ".pth")
            if args.dataset == "tiktok":
                audio_adj = _torch_load(self.path + '/Audio_mat_' + norm_type + ".pth")
        except Exception:
            # Load raw multi-modal features
            image_feats: torch.Tensor = torch.tensor(
                np.load('../data/{}/image_feat.npy'.format(args.dataset))
            ).float()
            text_feats: torch.Tensor = torch.tensor(
                np.load('../data/{}/text_feat.npy'.format(args.dataset))
            ).float()
            if args.dataset == "tiktok":
                audio_feats: torch.Tensor = torch.tensor(
                    np.load('../data/{}/audio_feat.npy'.format(args.dataset))
                ).float()

            # Cosine similarity → k-NN → normalise for each modality
            image_adj = self.build_sim(image_feats)
            image_adj = self.build_knn_normalized_graph(image_adj, topk=args.topk)

            text_adj = self.build_sim(text_feats)
            text_adj = self.build_knn_normalized_graph(text_adj, topk=args.topk)
            if args.dataset == "tiktok":
                audio_adj = self.build_sim(audio_feats)
                audio_adj = self.build_knn_normalized_graph(audio_adj, topk=args.topk)

            image_adj = self.norm_dense(image_adj, norm_type)
            text_adj = self.norm_dense(text_adj, norm_type)
            if args.dataset == "tiktok":
                audio_adj = self.norm_dense(audio_adj, norm_type)

            # Convert to sparse and cache
            image_adj = image_adj.to_sparse()
            text_adj = text_adj.to_sparse()
            if args.dataset == "tiktok":
                audio_adj = audio_adj.to_sparse()

            torch.save(image_adj, self.path + '/Image_mat_' + norm_type + ".pth")
            torch.save(text_adj, self.path + '/Text_mat_' + norm_type + ".pth")
            if args.dataset == "tiktok":
                torch.save(audio_adj, self.path + '/Audio_mat_' + norm_type + ".pth")

        print("End Load I2I media-specific mat:[%.1fs](" % (time() - t) + norm_type + ")")
        if args.dataset == "tiktok":
            return image_adj, text_adj, audio_adj
        else:
            return image_adj, text_adj, ""

    # ===================================================================
    #  Item-Item Multi-Modal Hypergraph (2-modality: Clothing, Sports)
    # ===================================================================
    def get_I2I_Hypergrah_mat(self, norm_type: str = "origin") -> torch.Tensor:
        """
        Build the multi-modal hypergraph incidence matrix H (2 modalities).

        Returns:
            Sparse tensor of shape (n_items, 2 × n_items).
        """
        print(f"Loading I2I multi-media Hypergraph mat:({norm_type})_topk:{str(args.topk)}")
        t: float = time()
        Hypergraph: torch.Tensor
        try:
            Hypergraph = _torch_load(
                f"{self.path}/hypergraph_mat_{norm_type}_topk_{str(args.topk)}.pth"
            )
        except Exception:
            # Load features for both modalities
            image_feats: torch.Tensor = torch.tensor(
                np.load('../data/{}/image_feat.npy'.format(args.dataset))
            ).float()
            text_feats: torch.Tensor = torch.tensor(
                np.load('../data/{}/text_feat.npy'.format(args.dataset))
            ).float()

            # Build per-modality k-NN similarity graphs
            image_adj: torch.Tensor = self.build_sim(image_feats)
            image_adj = self.build_knn_normalized_graph(image_adj, topk=args.topk)

            text_adj: torch.Tensor = self.build_sim(text_feats)
            text_adj = self.build_knn_normalized_graph(text_adj, topk=args.topk)

            # Concatenate along columns to form the hypergraph incidence matrix
            Hypergraph = torch.cat((image_adj, text_adj), dim=1)
            Hypergraph = self.norm_dense(Hypergraph, norm_type)
            Hypergraph = Hypergraph.to_sparse()
            torch.save(
                Hypergraph,
                f"{self.path}/hypergraph_mat_{norm_type}_topk_{str(args.topk)}.pth"
            )
        print("End Load I2I multi-media Hypergraph mat:[%.1fs](" % (time() - t) + norm_type + ")")
        return Hypergraph

    def get_I2I_Hypergraph_mul_mat(self, norm_type: str = "sym") -> torch.Tensor:
        """
        Build the final item-item multi-modal hypergraph:  H · H^T

        Shape: (n_items × n_items), sparse.
        """
        print(f"Loading I2I multi-media Hypergraph mul mat*mat.T:({norm_type})_topk:{str(args.topk)}")
        t: float = time()
        Hypergraph_mul: torch.Tensor
        try:
            Hypergraph_mul = _torch_load(
                f"{self.path}/hypergraph_mat_mul_{norm_type}_topk_{str(args.topk)}.pth"
            )
        except Exception:
            Hypergraph: torch.Tensor = self.get_I2I_Hypergrah_mat("origin")
            Hypergraph_mul = torch.sparse.mm(Hypergraph, Hypergraph.to_dense().T)
            Hypergraph_mul = self.norm_dense(Hypergraph_mul, norm_type)
            Hypergraph_mul = Hypergraph_mul.to_sparse()
            torch.save(
                Hypergraph_mul,
                f"{self.path}/hypergraph_mat_mul_{norm_type}_topk_{str(args.topk)}.pth"
            )
        print("End Load I2I multi-media Hypergraph mul mat*mat.T:[%.1fs](" % (time() - t) + norm_type + ")")
        return Hypergraph_mul

    # ===================================================================
    #  PyTorch (.pt) feature variants (for datasets stored as .pt files)
    # ===================================================================
    def get_I2I_Hypergrah_mat_pt(self, norm_type: str = "origin") -> torch.Tensor:
        """Same as get_I2I_Hypergrah_mat() but loads features from .pt files."""
        print("Loading I2I multi-media Hypergraph mat:(" + norm_type + ")")
        t: float = time()
        Hypergraph: torch.Tensor
        try:
            Hypergraph = _torch_load(self.path + '/hypergraph_mat_' + norm_type + ".pth")
        except Exception:
            image_feats: torch.Tensor = _torch_load("../data/{}/img_feat.pt".format(args.dataset))
            text_feats: torch.Tensor = _torch_load("../data/{}/text_feat.pt".format(args.dataset))

            image_adj: torch.Tensor = self.build_sim(image_feats)
            image_adj = self.build_knn_normalized_graph(image_adj, topk=args.topk)

            text_adj: torch.Tensor = self.build_sim(text_feats)
            text_adj = self.build_knn_normalized_graph(text_adj, topk=args.topk)

            Hypergraph = torch.cat((image_adj, text_adj), dim=1)
            Hypergraph = self.norm_dense(Hypergraph, norm_type)
            Hypergraph = Hypergraph.to_sparse()
            torch.save(Hypergraph, self.path + '/hypergraph_mat_' + norm_type + ".pth")
        print("End Load I2I multi-media Hypergraph mat:[%.1fs](" % (time() - t) + norm_type + ")")
        return Hypergraph

    def get_I2I_Hypergraph_mul_mat_pt(self, norm_type: str = "sym") -> torch.Tensor:
        """Same as get_I2I_Hypergraph_mul_mat() but uses .pt features."""
        print("Loading I2I multi-media Hypergraph mul mat*mat.T pytorch:(" + norm_type + ")")
        t: float = time()
        Hypergraph_mul: torch.Tensor
        try:
            Hypergraph_mul = _torch_load(self.path + '/hypergraph_mat_mul' + norm_type + ".pth")
        except Exception:
            Hypergraph: torch.Tensor = self.get_I2I_Hypergrah_mat_pt("origin")
            Hypergraph_mul = torch.sparse.mm(Hypergraph, Hypergraph.to_dense().T)
            Hypergraph_mul = self.norm_dense(Hypergraph_mul, norm_type)
            Hypergraph_mul = Hypergraph_mul.to_sparse()
            torch.save(Hypergraph_mul, self.path + '/hypergraph_mat_mul' + norm_type + ".pth")
        print("End Load I2I multi-media Hypergraph mul mat*mat.T pytorch:[%.1fs](" % (time() - t) + norm_type + ")")
        return Hypergraph_mul

    # ===================================================================
    #  Item-Item Multi-Modal Hypergraph (3-modality: Tiktok)
    # ===================================================================
    def get_tiktok_I2I_Hypergrah_mat(self, norm_type: str = "origin") -> torch.Tensor:
        """
        Build the multi-modal hypergraph incidence matrix H for Tiktok
        (3 modalities: image, text, audio).

        Returns:
            Sparse tensor of shape (n_items, 3 × n_items).
        """
        print(f"Loading I2I multi-media Hypergraph mat:({ norm_type })_topk:{str(args.topk)}")
        t: float = time()
        Hypergraph: torch.Tensor
        try:
            Hypergraph = _torch_load(
                f"{self.path}/hypergraph_mat_{norm_type}_topk_{str(args.topk)}.pth"
            )
        except Exception:
            # Load all three modality features
            image_feats: torch.Tensor = torch.tensor(
                np.load('../data/{}/image_feat.npy'.format(args.dataset))
            ).float()
            text_feats: torch.Tensor = torch.tensor(
                np.load('../data/{}/text_feat.npy'.format(args.dataset))
            ).float()
            audio_feats: torch.Tensor = torch.tensor(
                np.load('../data/{}/audio_feat.npy'.format(args.dataset))
            ).float()

            # Build k-NN similarity graph for each modality
            image_adj: torch.Tensor = self.build_sim(image_feats)
            image_adj = self.build_knn_normalized_graph(image_adj, topk=args.topk)

            text_adj: torch.Tensor = self.build_sim(text_feats)
            text_adj = self.build_knn_normalized_graph(text_adj, topk=args.topk)

            audio_adj: torch.Tensor = self.build_sim(audio_feats)
            audio_adj = self.build_knn_normalized_graph(audio_adj, topk=args.topk)

            # Concatenate all three modalities into the incidence matrix
            Hypergraph = torch.cat(
                (torch.cat((image_adj, text_adj), dim=1), audio_adj), dim=1
            )
            Hypergraph = self.norm_dense(Hypergraph, norm_type)
            Hypergraph = Hypergraph.to_sparse()
            torch.save(
                Hypergraph,
                f"{self.path}/hypergraph_mat_{norm_type}_topk_{str(args.topk)}.pth"
            )
        print("End Load I2I multi-media Hypergraph mat:[%.1fs](" % (time() - t) + norm_type + ")")
        return Hypergraph

    def get_tiktok_I2I_Hypergraph_mul_mat(self, norm_type: str = "sym") -> torch.Tensor:
        """
        Build  H · H^T  for the 3-modality Tiktok hypergraph.

        Shape: (n_items × n_items), sparse.
        """
        print(f"Loading I2I multi-media Hypergraph mul mat*mat.T:({norm_type})_topk:{str(args.topk)}")
        t: float = time()
        Hypergraph_mul: torch.Tensor
        try:
            Hypergraph_mul = _torch_load(
                f"{self.path}/hypergraph_mat_mul_{norm_type}_topk_{str(args.topk)}.pth"
            )
        except Exception:
            Hypergraph: torch.Tensor = self.get_tiktok_I2I_Hypergrah_mat("origin")
            Hypergraph_mul = torch.sparse.mm(Hypergraph, Hypergraph.to_dense().T)
            Hypergraph_mul = self.norm_dense(Hypergraph_mul, norm_type)
            Hypergraph_mul = Hypergraph_mul.to_sparse()
            torch.save(
                Hypergraph_mul,
                f"{self.path}/hypergraph_mat_mul_{norm_type}_topk_{str(args.topk)}.pth"
            )
        print("End Load I2I multi-media Hypergraph mul mat*mat.T:[%.1fs](" % (time() - t) + norm_type + ")")
        return Hypergraph_mul

    # ===================================================================
    #  Similarity & k-NN Graph Building Utilities
    # ===================================================================
    def build_sim(self, context: torch.Tensor) -> torch.Tensor:
        """
        Compute a cosine similarity matrix from a feature matrix.

        Args:
            context : (N, D) tensor — feature vectors for N entities

        Returns:
            (N, N) tensor — pairwise cosine similarities, values in [-1, 1].
        """
        context_norm: torch.Tensor = context.div(
            torch.norm(context, p=2, dim=-1, keepdim=True)
        )
        sim: torch.Tensor = torch.mm(context_norm, context_norm.transpose(1, 0))
        return sim

    def build_sim_feature_nan(self, context: torch.Tensor) -> torch.Tensor:
        """
        Cosine similarity with NaN-safe handling.

        Some feature extractors produce zero-vectors for missing/corrupted
        items (e.g. broken image URLs).  Normalising a zero-vector produces
        NaN, so we replace NaNs with 0 before computing similarity.
        """
        context_norm: torch.Tensor = context.div(
            torch.norm(context, p=2, dim=-1, keepdim=True)
        )
        context_norm[context_norm.isnan()] = 0
        sim: torch.Tensor = torch.mm(context, context.transpose(1, 0))
        return sim

    def build_knn_normalized_graph(
        self, adj: torch.Tensor, topk: int
    ) -> torch.Tensor:
        """
        k-NN sparsification: keep only the top-k values per row.

        Args:
            adj  : (N, N) — dense similarity matrix
            topk : number of neighbours to keep

        Returns:
            (N, N) — sparse binary adjacency matrix with at most topk
                     non-zero entries per row.
        """
        knn_val: torch.Tensor
        knn_ind: torch.Tensor
        knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
        # Scatter top-k values into a zeroed matrix, then binarise
        adj = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
        adj[adj > 0] = 1.
        return adj
