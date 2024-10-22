from typing import Union

import numpy as np
import scipy as sp
import torch


def stack_four(A: torch.Tensor, B: torch.Tensor,
               C: torch.Tensor, D: torch.Tensor):
    top = torch.cat([A, B], dim=1)
    bottom = torch.cat([C, D], dim=1)
    return torch.cat([top, bottom], dim=0)


def scale(matrix: np.ndarray):
    return matrix / np.max(np.abs(matrix))


def compute_steady_dist(P: np.array) -> np.ndarray:
    '''
    P: transition probability matrix
    '''

    n = P.shape[0]
    null_vec = sp.linalg.null_space(np.eye(n) - P.T)

    return (null_vec / np.sum(null_vec)).flatten()


def solve_msve_weight(steady_dist: np.ndarray,
                      X: np.ndarray,
                      v: np.ndarray) -> np.ndarray:
    '''
    P: transition probability matrix
    X: feature matrix
    v: true value
    returns weight minimizing MSVE
    '''
    D = np.diag(steady_dist)
    return np.linalg.inv(X.T @ D @ X) @ X.T @ D @ v


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def compare_P(P_tf: np.ndarray, d: int):
    '''
    P_tf: P matrix from transformer
    P_true: hardcoded P matrix that implements TD
    '''
    bottom_right = P_tf[-1, -1]
    avg_abs_all_others = 1/((2*d+1)**2 - 1) * \
        (np.sum(np.abs(P_tf)) - np.abs(P_tf[-1, -1]))
    return bottom_right, avg_abs_all_others


def compare_Q(Q_tf: np.ndarray, d: int):
    '''
    Q_tf: Q matrix from transformer
    Q_true: hardcoded Q matrix that implements TD
    d: feature dimension
    '''
    upper_left_block_trace = np.trace(Q_tf[:d, :d])
    upper_right_block_trace = np.trace(Q_tf[:d, d:2*d])
    # average of absolute values of all other elements
    # (we have 2d+1 x 2d+1 matrix and we are excluding the diagonal entries of the two upper dxd blocks)
    avg_abs_all_others = 1/((2*d+1)**2 - 2*d)*(np.sum(np.abs(Q_tf)) -
                                               upper_right_block_trace - upper_left_block_trace)
    return upper_left_block_trace, upper_right_block_trace, avg_abs_all_others


# Ensures that the hyperparameters are the same across 2 runs
def check_params(params, params_0):
    for key in [k for k in params.keys() if k != 'random_seed']:
        if params[key] != params_0[key]:
            raise ValueError(f'Parameter {key} is not the same across runs.')


def cos_sim(v1: Union[torch.Tensor, np.ndarray],
            v2: Union[torch.Tensor, np.ndarray]) -> float:
    '''
    v1: vector 1
    v2: vector 2
    returns cosine distance between v1 and v2
    '''
    if isinstance(v1, torch.Tensor):
        v1 = v1.detach().numpy()
    if isinstance(v2, torch.Tensor):
        v2 = v2.detach().numpy()

    v1 = v1.flatten()
    v2 = v2.flatten()
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
