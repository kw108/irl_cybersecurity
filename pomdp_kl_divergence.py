"""Script to compute KL divergence
"""

import numpy as np
import os
from bag2pomdp import POMDP
from typing import Dict, List, Tuple, Optional, Union
from numpy.typing import NDArray
from globals import GlobalParams


def kl_divergence(
        T_true: NDArray[np.float64],
        T_est: NDArray[np.float64],
        epsilon: float = 1e-10
) -> Tuple[NDArray[np.float64], np.float64]:
    """Vectorized computation of KL divergence.

    Args:
        T_true: True transition kernel
        T_est: Estimated transition kernel
        epsilon: Small positive value to avoid division by zero

    Returns:
        kl: KL Divergence KL(T_est || T_true)
        np.mean(kl): Mean KL Divergence along all axes
    """
    P = T_est + epsilon
    Q = T_true + epsilon

    P_norm = P / P.sum(axis=2, keepdims=True)
    Q_norm = Q / Q.sum(axis=2, keepdims=True)

    kl = np.sum(P_norm * np.log(P_norm / Q_norm), axis=2)

    return kl, np.mean(kl)


if __name__ == '__main__':
    # =========================================================================
    # Initialization
    # =========================================================================
    params = GlobalParams()

    # Set a random seed for reproducibility
    np.random.seed(params.random_seed)

    # =========================================================================
    # POMDP Instance Creation
    # =========================================================================

    # Create POMDP and IRLSolver instances
    pomdp = POMDP.create_pomdp_instance(params.NODES, params.EDGES, params.obs_err, params.stats_dir)

    T = pomdp.trans_kernel
    if isinstance(params.num_runs_T, int):
        N_1 = np.load(os.path.join(pomdp.stats_dir, params.N_1_filepath()))
        N_0 = np.load(os.path.join(pomdp.stats_dir, params.N_0_filepath()))
    else:
        T = np.minimum(np.maximum(T, 1e-3), 1 - 1e-3)
        N_1 = T * 1e1
        N_0 = (1 - T) * 1e1

    s = list()
    for i in range(10):
        s.append(kl_divergence(np.random.beta(N_1, N_0), T)[1])
    print(s)
