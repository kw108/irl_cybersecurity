"""
Script for solving for POMDP policy in the forward problem
"""

import numpy as np
import pickle
import os
from bag2pomdp import POMDP
from mce_irl_solver import OptOptions, IRLSolver
from globals import GlobalParams


def solve_forward(params: GlobalParams) -> None:
    """Evaluate MCE-IRL and benchmarks with given parameters."""

    # Set a random seed for reproducibility
    np.random.seed(params.random_seed)

    # =========================================================================
    # POMDP Instance Creation
    # =========================================================================

    # Create POMDP and IRLSolver instances with specified observation error
    pomdp = POMDP.create_pomdp_instance(params.NODES, params.EDGES, params.obs_err,
                                        params.stats_dir, params.cache_file)

    # =========================================================================
    # Transition Kernel Loading and Processing
    # =========================================================================

    if isinstance(params.num_runs_T, int):
        # Estimated transition kernel if num_runs_T is int
        N_1 = np.load(os.path.join(pomdp.stats_dir, params.N_1_filepath()))
        N_0 = np.load(os.path.join(pomdp.stats_dir, params.N_0_filepath()))
        T = N_1 / (N_1 + N_0)
    else:
        # True transition kernel if num_runs_T is "inf"
        T = pomdp.trans_kernel

    # =========================================================================
    # IRL Solver Initialization (Training Instance)
    # =========================================================================

    # Recompute pomdp.pred with T but never overwrite true transition kernel with T.
    # For pomdp, T is used only for belief update; for irl_pb, only pomdp.pred is needed instead of T.
    pomdp.recompute_pred(T)

    # Create instances with true transition kernel and estimated pred for training
    m_options = OptOptions(mu=1e4, mu_feat=10.0, maxiter=20, maxiter_weight=10, rho_weight=0.05,
                           policy_epsilon=0, discount=0.95, verbose=True)

    irl_pb = IRLSolver(pomdp, init_trust_region=1.25, options=m_options)

    # Initialize only once
    m_options.maxiter = 0       # force to have no iterations
    irl_pb.compute_maxent_pomdp_policy_via_scp(params.true_weight, params.true_weight, feat_counts=None,
                                               init_problem=True, init_policy=None,
                                               trust_prev=None, init_visit=None)

    # =========================================================================
    # OPTIMAL BENCHMARK: POMDP Policy Under True Weight
    # =========================================================================

    init_pol = None
    trust_reg_val = None
    extra_args = None

    solver_filepath = os.path.join(pomdp.stats_dir, params.solver_filepath())
    if os.path.exists(solver_filepath):
        with open(solver_filepath, 'rb') as f:
            solver_stats = pickle.load(f)
        init_pol = solver_stats['pol']
        trust_reg_val = solver_stats['trust_reg_val']

    m_options.maxiter = 10
    # Optimize and save forward solver statistics
    pol, _, _ = irl_pb.compute_maxent_pomdp_policy_via_scp(params.true_weight, params.true_weight,
                                                           feat_counts=None, init_problem=False,
                                                           init_policy=init_pol, trust_prev=trust_reg_val,
                                                           init_visit=None,
                                                           irl_filepath=params.solver_filepath(),
                                                           logs_filepath=params.logs_filepath())


if __name__ == '__main__':
    # TODO: set hyperparameters in globals.py
    params = GlobalParams()
    for T in [400, 600, 800, 1000, "inf"]:
        params.num_runs_T = T
        solve_forward(params)
