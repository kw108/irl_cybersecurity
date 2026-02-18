"""Script to perform Thompson Sampling in POMDP

Estimates transition kernels and save it.
"""

import numpy as np
from bag2pomdp import POMDP
from typing import Dict, List, Tuple, Optional, Union
from numpy.typing import NDArray
from globals import GlobalParams


def thompson_sampling(params: GlobalParams) -> None:
    """Perform Thompson sampling given number of samples."""

    # Set a random seed for reproducibility
    np.random.seed(params.random_seed)

    # =========================================================================
    # POMDP Instance Creation
    # =========================================================================

    pomdp = POMDP.create_pomdp_instance(params.NODES, params.EDGES, params.obs_err,
                                        params.stats_dir, params.cache_file)

    # =========================================================================
    # Thompson Sampling
    # =========================================================================

    pomdp.perform_thompson_sampling(params.num_runs_T, params.num_steps, params.save_interval,
                                    params.N_1_filepath(), params.N_0_filepath(), params.random_seed)


if __name__ == '__main__':
    # TODO: set hyperparameters in globals.py
    params = GlobalParams()
    for T in [400, 600, 800, 1000, "inf"]:
        params.num_runs_T = T
        thompson_sampling(params)
