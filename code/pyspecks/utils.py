from typing import Tuple

from numpy import ndarray
import numpy as np
from torch import Tensor
import itertools


def score_clustering(
    pred: ndarray,
    targ: ndarray,
    basic: bool = True
) -> Tuple[float, ndarray]:
    assert pred.shape == targ.shape
    shape = pred.shape
    pred = pred.copy().reshape(-1)
    targ = targ.copy().reshape(-1)
    if basic:
        mask = (targ <= 3)
    else:
        mask = np.ones(targ.size, dtype=bool)
    pred_labels = np.unique(pred[mask])

    best_perm = pred.copy()
    for group_id in pred_labels:
        group_id_mask = ((pred == group_id) & mask)
        new_group_id = np.bincount(targ[group_id_mask]).argmax()
        best_perm[group_id_mask] = new_group_id

    best_score = np.mean(best_perm[mask] == targ[mask])

    targ[~mask] = 0
    best_perm[~mask] = 0
    targ = targ.reshape(shape)
    best_perm = best_perm.reshape(shape)

    return best_score, best_perm, targ


def compute_smoothness(pred: ndarray) -> float:
    vert_score = np.abs(pred == np.roll(pred, shift=1, axis=0)).mean()
    horz_score = np.abs(pred == np.roll(pred, shift=1, axis=0)).mean()
    return np.mean([vert_score, horz_score])


def sparse_dim_multiply(
    A: Tensor,
    x: Tensor,
    dim: int
) -> Tensor:
    """Multiply a sparse Tensor by a vector along a particular dimension.
    """
    idx = A._indices()[dim]
    vals = A._values()
    vals *= x[idx]
    return A


def sparse_add_identity(
    A: Tensor
) -> Tensor:
    """Add identity matrix to a sparse Tensor.
    """
    idx1, idx2 = A._indices()
    vals = A._values()
    vals[idx1 == idx2] += 1
    return A
