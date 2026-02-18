"""
Script to compute feature counts from trajectories using transition kernels

Creates POMDP instances, loads demonstration data, and loads transition kernel matrices
to compute feature counts for IRL analysis and policy evaluation.
"""

import numpy as np
import pickle
import os
from bag2pomdp import POMDP
from globals import GlobalParams


def compute_feature_counts(params: GlobalParams):
    """Compute feature counts given trajectories and transition kernel."""

    # Set a random seed for reproducibility
    np.random.seed(params.random_seed)

    # =========================================================================
    # POMDP Instance Creation
    # =========================================================================

    # Create POMDP instance and create IRLSolver instance
    pomdp = POMDP.create_pomdp_instance(params.NODES, params.EDGES, params.obs_err,
                                        params.stats_dir, params.cache_file)

    # =========================================================================
    # File Path Configuration
    # =========================================================================

    # Compose demonstration dataset with consistent naming convention
    demo_filepath = params.demo_filepath()
    fc_filepath = params.fc_filepath()

    demo_filepath = os.path.join(pomdp.stats_dir, demo_filepath)
    fc_filepath = os.path.join(pomdp.stats_dir, fc_filepath)

    # =========================================================================
    # Load Demonstration Data
    # =========================================================================

    with open(demo_filepath, 'rb') as f:
        demo = pickle.load(f)

    obs_trajs = demo['obs_trajs']

    # =========================================================================
    # Load or Compute Transition Kernel
    # =========================================================================

    # Compose transition kernel with consistent naming convention
    if isinstance(params.num_runs_T, int):
        N_1 = np.load(os.path.join(pomdp.stats_dir, params.N_1_filepath()))
        N_0 = np.load(os.path.join(pomdp.stats_dir, params.N_0_filepath()))
        T = N_1 / (N_1 + N_0)
    else:
        T = pomdp.trans_kernel

    # =========================================================================
    # Compute Feature Counts
    # =========================================================================

    # Compute and save feature counts with the pomdp method
    feat_counts = pomdp.obs_trajs_to_feat_counts(obs_trajs, params.discount, fc_filepath, T)


if __name__ == '__main__':
    # TODO: set hyperparameters in globals.py
    params = GlobalParams()
    for T in [400, 600, 800, 1000, "inf"]:
        params.num_runs_T = T
        compute_feature_counts(params)
