"""
Script to simulate POMDP trajectories and export them as serialized data

Creates a POMDP instance, computes an optimal MDP policy, simulates trajectories,
and saves the results as a pickle file for subsequent IRL analysis.
"""

import numpy as np
import pickle
import os
from bag2pomdp import POMDP
from mce_irl_solver import OptOptions, IRLSolver
from globals import GlobalParams


def simulate_trajectories(params: GlobalParams) -> None:
    """Simulate trajectories given number of trajectories and transition kernel."""

    # Set a random seed for reproducibility
    np.random.seed(params.random_seed)

    # =========================================================================
    # POMDP Instance Initialization
    # =========================================================================

    pomdp = POMDP.create_pomdp_instance(params.NODES, params.EDGES, params.obs_err,
                                        params.stats_dir, params.cache_file)

    # =========================================================================
    # Output File Configuration
    # =========================================================================

    # Compose demonstration dataset with descriptive naming convention
    demo_filepath = os.path.join(pomdp.stats_dir, params.demo_filepath())

    # =========================================================================
    # IRL Solver Configuration and Policy Computation
    # =========================================================================

    # Configure optimization parameters
    m_options = OptOptions(mu=1e4, mu_feat=1.0, maxiter=10, maxiter_weight=10,
                           rho_weight=0.05, policy_epsilon=0, discount=0.95, verbose=True)

    irl_pb = IRLSolver(pomdp, init_trust_region=1.25, options=m_options)

    # =========================================================================
    # Trajectory Simulation
    # =========================================================================

    with open(os.path.join(pomdp.stats_dir, params.solver_filepath()), 'rb') as f:
        solver_stats = pickle.load(f)

    pol = solver_stats['pol']
    opt_pomdp_pol = {o: {a: act_dict[a] / tot if (tot := sum(act_dict.values())) > 0 else 1 / len(act_dict)
                         for a in act_dict} for o, act_dict in pol.items()}

    # Simulate trajectories with optimal POMDP policy
    obs_trajs = pomdp.simulate_pomdp_policy(opt_pomdp_pol, m_options.discount, params.num_trajs,
                                            params.max_steps_per_traj, seed=params.random_seed,
                                            stop_at_accepting_state=True)

    with open(demo_filepath, 'wb') as f:
        pickle.dump({'obs_trajs': obs_trajs}, f)


if __name__ == '__main__':
    # TODO: set hyperparameters in globals.py
    params = GlobalParams()
    for T in [400, 600, 800, 1000, "inf"]:
        params.num_runs_T = T
        simulate_trajectories(params)
