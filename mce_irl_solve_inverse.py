"""
Script for inverse learning POMDP policy from demonstration
"""

import numpy as np
import pickle
import os
from bag2pomdp import POMDP
from mce_irl_solver import OptOptions, IRLSolver
from globals import GlobalParams


def solve_inverse(params: GlobalParams) -> str:
    """Solve MCE-IRL with given parameters."""
    # =========================================================================
    # Initialization
    # =========================================================================

    # Set a random seed for reproducibility
    np.random.seed(params.random_seed)

    # =========================================================================
    # POMDP Instance Creation
    # =========================================================================

    # Create POMDP and IRLSolver instances
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
    m_options = OptOptions(mu=1e4, mu_feat=10.0, maxiter=20, maxiter_weight=10,
                           rho_weight=0.05, policy_epsilon=0, discount=0.95, verbose=True)

    irl_pb = IRLSolver(pomdp, init_trust_region=1.25, options=m_options)

    # =========================================================================
    # IRL Solver Initialization (Testing Instance)
    # =========================================================================

    # Create new instances with true transition kernel and pred for testing
    _pomdp = POMDP.create_pomdp_instance(params.NODES, params.EDGES, params.obs_err, params.stats_dir)
    _m_options = OptOptions(mu=1e4, mu_feat=10.0, maxiter=20, maxiter_weight=10,
                            rho_weight=0.05, policy_epsilon=0, discount=0.95, verbose=True)
    _irl_pb = IRLSolver(_pomdp, init_trust_region=1.25, options=_m_options)

    # =========================================================================
    # Load Feature Counts and Demonstration Data
    # =========================================================================

    # Compose demonstration dataset file with descriptive naming convention
    fc_filepath = params.fc_filepath()
    fc_filepath = os.path.join(pomdp.stats_dir, fc_filepath)

    # Load feature counts and demonstration data
    with open(fc_filepath, 'rb') as f:
        fc = pickle.load(f)

    feat_counts = fc['feat_counts']
    pol = fc['raw_obs_pol']

    # Use normalized observation-action frequencies as initial policy
    init_pol = {o: {a: act_dict[a] / tot if (tot := sum(act_dict.values())) > 0 else 1 / len(act_dict)
                    for a in act_dict} for o, act_dict in pol.items()}

    # =========================================================================
    # Initialize Solvers with Feature Counts
    # =========================================================================

    # Initialize only once
    m_options.maxiter = 0       # force to have no iterations
    irl_pb.compute_maxent_pomdp_policy_via_scp(params.init_weight, params.true_weight, feat_counts=feat_counts,
                                               init_problem=True, init_policy=None,
                                               trust_prev=None, init_visit=None)

    _m_options.maxiter = 0      # force to have no iterations
    _irl_pb.compute_maxent_pomdp_policy_via_scp(params.init_weight, params.true_weight, feat_counts=feat_counts,
                                                init_problem=True, init_policy=None,
                                                trust_prev=None, init_visit=None)

    res_literal = (f'obs_err: {params.err_pct:03d}, num_runs_T: {params.num_runs_T},'
                   f' num_trajs: {params.num_trajs},')

    """
    # =========================================================================
    # BENCHMARK: Optimal MDP Policy (RL Solution)
    # =========================================================================

    opt_mdp_pol, _, _ = irl_pb.compute_optimal_mdp_policy(params.true_weight)

    opt_mdp_pol_val = _pomdp.simulate_mdp_policy_value(opt_mdp_pol, _m_options.discount,
                                                       num_trajs=params.num_trajs,
                                                       max_steps_per_traj=params.max_steps_per_traj,
                                                       weight=params.true_weight)

    # FIXME: a better solution could be to invoke compute_optimal_mdp_policy with opt_mdp_pol in _irl_pb
    # FIXME: that would invoke constr_bellman_flow and extract_varcoeff_from_bellman both with mdp=True
    # FIXME: but resulted incorrect solution by failing to comply with the Bellman flow constraints.
    # FIXME: should work correctly by using this line of code instead
    # opt_mdp_pol, _, _ = _irl_pb.compute_optimal_mdp_policy(params.true_weight, sigma=opt_mdp_pol)

    res_literal += f' optimal policy value with MDP solver: {opt_mdp_pol_val},'
    """

    # =========================================================================
    # OPTIMAL BENCHMARK: POMDP Policy Under True Weight
    # =========================================================================

    with open(os.path.join(pomdp.stats_dir, params.solver_filepath()), 'rb') as f:
        solver_stats = pickle.load(f)

    opt_pol = solver_stats['pol']

    # Obtain state-action visitation for optimal policy under true T
    _, _, nu_s_a = _irl_pb.obtain_visitation_from_policy(opt_pol)

    # Obtain optimal policy value (multiplied by mu_feat) under true T
    opt_pol_val = sum(
        coeff * val * _m_options.mu for coeff, val in _irl_pb.compute_expected_reward(nu_s_a, params.true_weight))

    res_literal += f' optimal policy value: {opt_pol_val},'

    # =========================================================================
    # PROPOSED BENCHMARK: POMDP Policy With Random Weight Initialization (IRL)
    # =========================================================================

    irl_pol = init_pol
    init_weight = params.init_weight

    irl_filepath = os.path.join(pomdp.stats_dir, params.irl_filepath())
    if os.path.exists(irl_filepath):
        with open(irl_filepath, 'rb') as f:
            irl_stats = pickle.load(f)

        irl_pol = irl_stats['pol']
        init_weight = irl_stats['weight']

    # TODO: set number of iterations with maxiter and maxiter_weight
    m_options.maxiter = 10
    m_options.maxiter_weight = 5

    # Obtain irl policy under estimated T
    _, irl_pol, _ = irl_pb.solve_irl_pomdp_given_traj(init_weight, params.true_weight, feat_counts, irl_pol,
                                                      irl_filepath=params.irl_filepath(),
                                                      logs_filepath=params.logs_filepath(),
                                                      from_saved_stats=os.path.exists(irl_filepath))

    # Obtain state-action visitation for irl policy under true T
    _, _, nu_s_a = _irl_pb.obtain_visitation_from_policy(irl_pol)

    # Obtain irl policy value (multiplied by mu_feat) under true T
    irl_pol_val = sum(
        coeff * val * _m_options.mu for coeff, val in _irl_pb.compute_expected_reward(nu_s_a, params.true_weight))

    res_literal += f' irl policy value: {irl_pol_val},\n'

    # =========================================================================
    # IMITATION BENCHMARK: Imitation POMDP Policy
    # =========================================================================

    imi_pol = init_pol

    # Obtain state-action visitation for imitation policy under true T
    _, _, nu_s_a = _irl_pb.obtain_visitation_from_policy(imi_pol)

    # Obtain imitation policy value (multiplied by mu_feat) under true T
    imi_pol_val = sum(
        coeff * val * _m_options.mu for coeff, val in _irl_pb.compute_expected_reward(nu_s_a, params.true_weight))

    res_literal += f' imitation policy value: {imi_pol_val}.\n'

    return res_literal


if __name__ == '__main__':
    # TODO: set hyperparameters in globals.py
    params = GlobalParams()

    res_literal = solve_inverse(params)
    print(res_literal)

    with open('benchmark_results.txt', 'a') as f:
        f.write(res_literal)
