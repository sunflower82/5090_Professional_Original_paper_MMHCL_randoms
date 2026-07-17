"""
norm.py — Graph Normalisation Utilities
=========================================

Provides functions for building similarity graphs and normalising
adjacency matrices (both sparse and dense formats).

These are used by Models.py (imported at the top) but NOT directly called
during training — load_data.py has its own inline versions.  This module
exists for the LATTICE-style model variants that use ``build_knn_normalized_graph``
with sparse output and Laplacian normalisation.

Normalisation types:
    'sym'  : Symmetric normalisation    D^{-½} A D^{-½}
             Preserves symmetry; standard for undirected graphs (GCN-style).
    'rw'   : Random-walk normalisation  D^{-1} A
             Rows sum to 1; equivalent to a transition probability matrix.
    'none' : No normalisation (return A as-is).

Mathematical background:
    For an adjacency matrix A with degree matrix D = diag(Σ_j A_{ij}):
    - Symmetric:   L_sym = D^{-½} A D^{-½}
    - Random-walk: L_rw  = D^{-1} A
"""

from __future__ import annotations

import torch


def build_sim(context: torch.Tensor) -> torch.Tensor:
    """
    Compute a pairwise cosine similarity matrix.

    Args:
        context : (N, D) tensor — feature vectors for N entities

    Returns:
        (N, N) tensor — cosine similarity matrix, values in [-1, 1].

    Steps:
        1. L2-normalise each row:  context_norm = F / ||F||_2
        2. Compute dot products:   sim = context_norm · context_norm^T
    """
    context_norm: torch.Tensor = context.div(
        torch.norm(context, p=2, dim=-1, keepdim=True)
    )
    sim: torch.Tensor = torch.mm(context_norm, context_norm.transpose(1, 0))
    return sim


def build_knn_normalized_graph(
    adj: torch.Tensor,
    topk: int,
    is_sparse: bool,
    norm_type: str,
) -> torch.Tensor:
    """
    Build a k-NN graph from a similarity matrix with optional normalisation.

    For each row (entity), keep only the ``topk`` largest values and set the
    rest to zero.  Then normalise the resulting adjacency.

    Args:
        adj       : (N, N) tensor — pairwise similarity matrix
        topk      : number of neighbours to retain per entity
        is_sparse : if True, return a sparse COO tensor;
                    if False, return a dense tensor
        norm_type : normalisation type: 'sym', 'rw', or 'none'

    Returns:
        Normalised k-NN adjacency matrix (sparse or dense).
    """
    device: torch.device = adj.device
    knn_val: torch.Tensor
    knn_ind: torch.Tensor
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)

    if is_sparse:
        # --- Sparse output path ---
        # Build edge list from the top-k indices
        tuple_list: list[list[int]] = [
            [row, int(col)]
            for row in range(len(knn_ind))
            for col in knn_ind[row]
        ]
        row_idx: list[int] = [i[0] for i in tuple_list]
        col_idx: list[int] = [i[1] for i in tuple_list]
        i: torch.Tensor = torch.LongTensor([row_idx, col_idx]).to(device)
        v: torch.Tensor = knn_val.flatten()

        # Apply Laplacian normalisation on the sparse edge weights
        edge_index: torch.Tensor
        edge_weight: torch.Tensor
        edge_index, edge_weight = get_sparse_laplacian(
            i, v, normalization=norm_type, num_nodes=adj.shape[0]
        )
        return torch.sparse_coo_tensor(edge_index, edge_weight, adj.shape)

    else:
        # --- Dense output path ---
        # Scatter top-k values into a zero matrix to form the adjacency
        weighted_adjacency_matrix: torch.Tensor = (
            (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
        )
        return get_dense_laplacian(weighted_adjacency_matrix, normalization=norm_type)


def get_sparse_laplacian(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
    normalization: str = 'none',
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Normalise edge weights of a sparse graph (given as edge_index + edge_weight).

    Args:
        edge_index   : (2, E) LongTensor — [source_nodes; target_nodes]
        edge_weight  : (E,) FloatTensor  — edge weights
        num_nodes    : total number of nodes
        normalization: 'sym', 'rw', or 'none'

    Returns:
        Tuple of (edge_index, edge_weight) — with normalised weights

    Normalisation formulas:
        'sym': w'_{ij} = w_{ij} / sqrt(deg_i * deg_j)
        'rw' : w'_{ij} = w_{ij} / deg_i
    """
    row: torch.Tensor = edge_index[0]
    col: torch.Tensor = edge_index[1]

    # Compute node degrees: deg[i] = Σ_j w_{ij}
    # Try torch_scatter first; fall back to built-in scatter_add_
    deg: torch.Tensor
    try:
        from torch_scatter import scatter_add
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    except ImportError:
        deg = torch.zeros(num_nodes, device=edge_weight.device, dtype=edge_weight.dtype)
        deg.scatter_add_(0, row, edge_weight)

    if normalization == 'sym':
        # Symmetric: D^{-½} · w · D^{-½}
        deg_inv_sqrt: torch.Tensor = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    elif normalization == 'rw':
        # Random-walk: D^{-1} · w
        deg_inv: torch.Tensor = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = deg_inv[row] * edge_weight

    return edge_index, edge_weight


def get_dense_laplacian(
    adj: torch.Tensor,
    normalization: str = 'none',
) -> torch.Tensor:
    """
    Normalise a dense adjacency matrix.

    Args:
        adj           : (N, N) tensor — weighted adjacency matrix
        normalization : 'sym', 'rw', or 'none'

    Returns:
        (N, N) tensor — normalised adjacency matrix

    Formulas:
        'sym'  :  D^{-½} A D^{-½}    (symmetric normalisation)
        'rw'   :  D^{-1} A            (random-walk / row normalisation)
        'none' :  A                   (unchanged)
    """
    L_norm: torch.Tensor

    if normalization == 'sym':
        rowsum: torch.Tensor = torch.sum(adj, -1)
        d_inv_sqrt: torch.Tensor = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt: torch.Tensor = torch.diagflat(d_inv_sqrt)
        L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

    elif normalization == 'rw':
        rowsum = torch.sum(adj, -1)
        d_inv: torch.Tensor = torch.pow(rowsum, -1)
        d_inv[torch.isinf(d_inv)] = 0.
        d_mat_inv: torch.Tensor = torch.diagflat(d_inv)
        L_norm = torch.mm(d_mat_inv, adj)

    elif normalization == 'none':
        L_norm = adj

    return L_norm
