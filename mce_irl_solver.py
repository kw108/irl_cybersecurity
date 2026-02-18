"""
A Maximum Causal Entropy Inverse Reinforcement Learning Solver

This module implements a Maximum Causal Entropy (MCE) Inverse Reinforcement Learning (IRL) solver
over Partially Observable Markov Decision Process (POMDP). It provides functionality for computing
policy update with Sequential Convex Programming (SCP) and weight update with gradient.
"""

import numpy as np
import gurobipy as gp
import pickle
import time
from bag2pomdp import POMDP
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import os
from globals import GlobalParams
os.environ['GRB_LICENSE_FILE'] = GlobalParams().gurobipy_license


# A structure to set the rules for reducing and augmenting the trust region
set_trust_region = {'red': lambda x: ((x - 1) / 1.5 + 1),
                    'aug': lambda x: min(10, (x - 1) * 1.25 + 1),
                    'lim': 1 + 1e-4}

# Define gradient step size
set_gradient_step_size = lambda iter: 1.0 / np.power(iter + 2, 0.5)

# Define threshold for numerical zero
ZERO_NU_S = 1e-8


class OptOptions:
    """
    Class for setting up options for the optimization problem.
    """
    
    def __init__(
            self,
            mu: float = 1e4,
            mu_feat: float = 10.0,
            rho: float = 0.1,
            rho_weight: float = 0.1,
            maxiter: int = 100,
            max_update: int = 100,
            maxiter_weight: int = 20,
            policy_epsilon: float = 1e-3,
            discount: float = 0.9,
            verbose: bool = True,
            verbose_weight: bool = True,
            verbose_solver: bool = None
    ) -> None:
        """
        Initialization and assignment of parameters. Entropy reward always has factor 1,
        penalty for slack variables has factor mu, trajectory reward has factor mu_feat.

        Args:
            mu: Coefficient for weighing penalty on slack variables
            mu_feat: Coefficient for weighing trajectory rewards
            rho: Coefficient for weighing of feature mismatch
            rho_weight: Learning rate for weight update
            maxiter: Max number of iterations in the SCP method
            max_update: Max number of correct updates in the SCP method
            maxiter_weight: Max number of iteration for finding the weight
            policy_epsilon: The min probability of taking an action at each observation
            discount: Discount factor
            verbose: Enable some logging
            verbose_weight: Enable some logging regarding weight update
            verbose_solver: Enable some logging during invocations to Gurobi solver
        
        Raises:
            RuntimeError: Values are not within valid ranges
        """
        self.mu = mu
        self.rho = rho
        self.rho_weight = rho_weight
        self.maxiter = maxiter
        if max_update is None:
            self.max_update = maxiter
        else:
            self.max_update = max_update
        self.mu_feat = mu_feat
        self.quotmu = self.mu_feat / self.mu
        self.maxiter_weight = maxiter_weight
        self.policy_epsilon = policy_epsilon
        self.discount = discount
        self.verbose = verbose
        self.verbose_weight = verbose_weight
        self.verbose_solver = self.verbose if verbose_solver is None else verbose_solver
        if policy_epsilon < 0:
            raise RuntimeError('Policy epsilon should be >= 0.')
        if discount < 0 or discount >= 1:
            raise RuntimeError('Discount factor should be >= 0 and < 1.')
        if mu <= 0:
            raise RuntimeError('Coefficient mu should be > 0')


class IRLSolver:
    """
    Class that encodes the max entropy inverse reinforcement learning on POMDPs
    as sequential linear optimization problems with trust regions to alleviate
    the error introduced by the linearization. Each LP subproblems are
    solved using Gurobi version 11.0.3.
    
    The class also provides a function to find a policy that maximize the expected
    rewards on the POMDP given a formula.
    """

    def __init__(
            self,
            pomdp: POMDP,
            init_trust_region: float = set_trust_region['aug'](4),
            max_trust_region: float = 4,
            feat_eps: float = 1e-6,
            options: OptOptions = OptOptions()
    ) -> None:
        """
        Function for setting up options for the optimization problem.

        Args:
            pomdp: The POMDP environment
            init_trust_region: Initial trust region radius
            max_trust_region: Max trust region radius
            feat_eps: Threshold to stop iteration over features
            options: Options for the optimization problem
        """
        # Attributes to check performances of building, solving the problems
        self.total_solve_time = 0   # Total time elapsed
        self.init_encoding_time = 0     # Time for encoding the full problem
        self.update_constraint_time = 0     # Total time for updating the constraints and objective function
        self.checking_policy_time = 0   # Total time for finding nu_s and nu_sa given a policy

        # Save the initial trust region
        self._trust_region = init_trust_region
        self._max_trust_region = max_trust_region
        self._options = options
        self._feat_eps = feat_eps
        self._pomdp = pomdp

    def solve_irl_pomdp_given_traj(
            self,
            init_weight: Dict[str, float],
            true_weight: Dict[str, float],
            feat_counts: Dict[str, float],
            init_policy: Dict[int, Dict[int, float]],
            use_pure_gradient: bool = True,
            irl_filepath: Union[Path, str] = None,
            logs_filepath: Union[Path, str] = None,
            from_saved_stats: bool = False
    ) -> Tuple[Dict[str, float], Dict[int, Dict[int, float]], Dict[int, Dict[int, float]]]:
        """
        Solve the IRL problem given the feature count of the sample trajectory.
        init_weight should have an optimal policy init_policy.

        Args:
            init_weight: Initialized weight
            true_weight: True weight for reward function (only used for printing minimax value)
            feat_counts: Feature counts from the expert
            init_policy: Initialized policy
            use_pure_gradient: Whether to use pure gradient on weight update
            irl_filepath: Filepath to irl solver statistics
            logs_filepath: Filepath to logs information
            from_saved_stats: Whether to obtain policy and weight from save statistics
        
        Returns:
            weight: Weight associated with each feature
            pol: Policy
            nu_s_a: State-action visitation
        """
        if from_saved_stats:
            # No initialization of the problem with performing SCP iterations
            # since init_policy is already (nealy) optimal for init_weight
            pol = init_policy
            extra_args = \
                self.verify_solution(self.bellman_opt, self.nu_s_ver, pol, self.bellman_constr)
            trust_reg_val = self._trust_region
            nu_s_a = extra_args[2]
        else:
            # Performing SCP iterations and obtaining pol that is nearly optimal for weight
            # since weight gradient is valid only for nearly optimal policy
            pol, trust_reg_val, extra_args = self.compute_maxent_pomdp_policy_via_scp(init_weight, true_weight,
                                                                                      feat_counts=feat_counts,
                                                                                      init_problem=False,
                                                                                      init_policy=init_policy,
                                                                                      trust_prev=None,
                                                                                      init_visit=None,
                                                                                      irl_filepath=irl_filepath,
                                                                                      logs_filepath=logs_filepath)
            nu_s_a = extra_args[2]

        weight = init_weight

        # Start iterations on policy update
        pol_feat_counts = {f_name: 0.0 for f_name, feat in self._pomdp.reward_feats.items()}

        if logs_filepath is not None:
            self.write_to_txt('Initialized/restored weight dictionary: {}.'.format(weight),
                              logs_filepath=logs_filepath)

        if not use_pure_gradient:
            rmsprop = {f_name: 0.0 for f_name, feat in self._pomdp.reward_feats.items()}
            momentum = {f_name: 0.0 for f_name, feat in self._pomdp.reward_feats.items()}

        for i in range(self._options.maxiter_weight):
            # Store the difference between the expected feature by the policy and the feature counts
            diff_value = 0
            diff_value_dict = dict()

            # Save the new weight
            new_weight = dict()
            step_size = set_gradient_step_size(i + 1)

            if self._options.verbose_weight:
                print('---------------- Printing weight iteration: {} -----------------'.format(i))

            for f_name, val in weight.items():
                feat = self._pomdp.reward_feats[f_name]

                # Get the feature attained by the policy
                feat_pol = sum([feat[(s, a)] * nu_s_a_val
                                for s, nu_s_a_t in nu_s_a.items()
                                for a, nu_s_a_val in nu_s_a_t.items()
                                ])
                pol_feat_counts[f_name] = feat_pol

                # Get the reward by the feature expectation of the trajectories
                feat_demo = feat_counts[f_name]

                # Save the sum of the difference to detect convergence
                diff_value += np.abs(feat_demo - feat_pol)
                diff_value_dict[f_name] = feat_demo - feat_pol

                if not use_pure_gradient:
                    # incorporate rms prop to make sure that the gradients may not vary in magnitude
                    rmsprop[f_name] = 0.9 * rmsprop[f_name] + (1 - 0.9) * np.power(diff_value_dict[f_name], 2)

                    momentum[f_name] = (0.0 * momentum[f_name] + (1 - 0.0) * 
                                        (step_size / np.power(rmsprop[f_name] + 1e-8, 0.5)) * diff_value_dict[f_name])

                    if self._options.verbose_weight:
                        print(feat_pol, feat_demo, f_name, '| rmsprop: ', rmsprop[f_name])

            # Update the weight values
            for f_name, grad_val in diff_value_dict.items():
                if use_pure_gradient:
                    new_weight[f_name] = weight[f_name] + self._options.rho_weight * diff_value_dict[f_name]
                else:
                    new_weight[f_name] = weight[f_name] + self._options.rho_weight * momentum[f_name]

            if np.abs(diff_value) <= self._feat_eps:
                if self._options.verbose_weight:
                    print('Difference with feature counts: {}.'.format(diff_value))
                    print('Weight value: {}.'.format(weight))
                break

            # Update new weight
            weight = new_weight

            if logs_filepath is not None:
                self.write_to_txt('Finished with weight iteration: {}.'.format(i), logs_filepath=logs_filepath)
                self.write_to_txt('Weight dictionary: {}.'.format(weight), logs_filepath=logs_filepath)

            # Compute the new policy based on the obtained weight and previous trust region
            pol, trust_reg_val, extra_args = self.compute_maxent_pomdp_policy_via_scp(weight, true_weight,
                                                                                      feat_counts=feat_counts,
                                                                                      init_problem=False,
                                                                                      init_policy=pol,
                                                                                      trust_prev=trust_reg_val,
                                                                                      init_visit=extra_args,
                                                                                      irl_filepath=irl_filepath,
                                                                                      logs_filepath=logs_filepath)
            nu_s_a = extra_args[2]

            # Do some printing
            if self._options.verbose_weight:
                print('Diff with feature counts: {}.'.format(diff_value))
                print('New weight value: {}.'.format(weight))
                print('Update time: {}s, Checking time: {}s, solve time: {}s.'.format(
                    self.update_constraint_time, self.checking_policy_time, self.total_solve_time))

        return weight, pol, nu_s_a

    def compute_maxent_pomdp_policy_via_scp(
            self,
            weight: Dict[str, float],
            true_weight: Dict[str, float],
            use_weight_reg: bool = False,
            feat_counts: Dict[str, float] = None,
            init_problem: bool = True,
            init_policy: Dict[int, Dict[int, float]] = None,
            trust_prev: float = None,
            init_visit: Tuple[float, Dict[int, float], Dict[int, Dict[int, float]]] = None,
            irl_filepath: Union[Path, str] = None,
            logs_filepath: Union[Path, str] = None
    ) -> Tuple[Dict[int, Dict[int, float]], float, Tuple[float, Dict[int, float], Dict[int, Dict[int, float]]]]:
        """ 
        Given the current weight for each feature functions in the POMDP model,
        and the feature expected reward, compute the optimal policy
        that maximizes the max causal entropy.
        
        Args:
            weight: The coefficient associated to each feature function
            true_weight: The true coefficient associated to each feature function
                (only used for printing minimax value)
            use_weight_reg: Use regularization on weight if True
            feat_counts: The feature counts of dataset
            init_problem: True if the optimization problem hasn't been initialized before
            init_policy: The starting policy of the scp method, uniform policy if None
            trust_prev: Current trust region from the previous IRL iteration
            init_visit: Store the associated nu_s, nu_s_a, ent_reward to init_policy
            irl_filepath: Filepath to irl solver statistics
            logs_filepath: Filepath to logs information
        
        Returns:
            pol_k: Optimal policy
            trust_region: Trust region radius
            (latest_ent, nu_s_k, nu_s_a_k): objective value, state visitation, state-action visitation
        """
        # Create the optimization problem
        if init_problem:
            self.scp_opt = gp.Model('Optimal policy using sequential convex programming with Gurobi solver')
            self.bellman_opt = gp.Model('Exact optimal policy using Bellman flow with Gurobi Solver')
            self.total_solve_time = 0
            self.init_encoding_time = 0
            self.update_constraint_time = 0
            self.checking_policy_time = 0

            (self.nu_s, self.nu_s_a, self.sigma, self.slack_nu_p, self.slack_nu_n,
             self.constr_lin, self.constr_trust_reg, self.nu_s_ver, self.bellman_constr) \
                = self.init_optimization_problem(self.scp_opt, no_linearization=False,
                                                 check_opt=self.bellman_opt)

            if self._options.verbose:
                print('Time used to build the full Model: {}.'.format(self.init_encoding_time))

            # Define the parameters used by Gurobi for the linearized problem
            self.scp_opt.Params.OutputFlag = self._options.verbose_solver
            self.scp_opt.Params.Presolve = 2        # More aggressive presolve step
            self.scp_opt.Params.Method = 2          # The problem is not really a QP
            self.scp_opt.Params.Crossover = 0
            self.scp_opt.Params.CrossoverBasis = 0
            self.scp_opt.Params.BarHomogeneous = 1   # No need for, our problem is always feasible/bound
            self.scp_opt.Params.OptimalityTol = 1e-6
            self.scp_opt.Params.BarConvTol = 1e-6
            self.scp_opt.Params.NoRelHeurTime = 0

            # Define the parameters used by Gurobi for the auxiliary program
            self.bellman_opt.Params.OutputFlag = self._options.verbose_solver
            self.bellman_opt.Params.Presolve = 2
            self.bellman_opt.Params.Method = 2
            self.bellman_opt.Params.Crossover = 0
            self.bellman_opt.Params.CrossoverBasis = 0
            self.bellman_opt.Params.BarHomogeneous = 1
            self.bellman_opt.Params.OptimalityTol = 1e-6
            self.bellman_opt.Params.BarConvTol = 1e-6
            self.bellman_opt.Params.NoRelHeurTime = 0

        trust_region = self._trust_region

        # Initialize trust region from the previous IRL step, if it exists
        if trust_prev is not None:
            trust_region = trust_prev

        # Initialize policy at iteration k
        if init_policy is not None:
            pol_k = init_policy
        else:
            pol_k = {o: {a: 1.0 / len(act_list) for a in act_list} for o, act_list in self._pomdp.obs_actions.items()}

        # If the initial state and action visitation count are not given
        if init_visit is None:
            # Initialize the state and state-action visitation count based on the policy
            ent_reward, nu_s_k, nu_s_a_k = \
                self.verify_solution(self.bellman_opt, self.nu_s_ver, pol_k, self.bellman_constr)
        else:
            assert init_policy is not None
            ent_reward, nu_s_k, nu_s_a_k = init_visit

        # Store the current entropy cost
        latest_ent = ent_reward
        latest_ent_n = latest_ent
        nu_s_a_k_n = nu_s_a_k

        # Add the cost associated to the linear expected discounted reward given nu_s_a (multiplied by mu_feat)
        ent_reward += sum(coeff * val * self._options.mu
                          for coeff, val in self.compute_expected_reward(nu_s_a_k, weight))

        # Add ||(feat_counts - expected feat)||^2 if using regularization on weight
        if use_weight_reg:
            ent_reward -= self.compute_quad_featcount_value(nu_s_a_k, feat_counts) * (self._options.rho / 2)

        # Get the linear expression for expected reward (divided by mu)
        lin_expr_reward = self.compute_expected_reward(self.nu_s_a, weight)

        # Get the cost term ||(feat_counts - expected feat)||^2 if using regularization on weight
        quad_expr_reward = 0
        if use_weight_reg:
            quad_expr_reward = self.compute_quad_featcount(self.nu_s_a, feat_counts) * (self._options.rho / 2)

        # A counter to count the correct updates
        count_correct_update = 0

        # Save if the past step was rejected
        past_rejected = False

        for i in range(self._options.maxiter):
            # Update the set of linearized constraints
            curr_time = time.time()
            self.update_constr_and_trust_region(self.scp_opt, self.constr_lin, self.constr_trust_reg,
                                                nu_s_k, pol_k, self.nu_s, self.sigma, trust_region,
                                                no_linearization=False, only_trust=past_rejected)

            # Set the current objective based on past solution
            if not past_rejected:
                pen_reward_list = self.compute_entropy_reward(nu_s_k, nu_s_a_k, self.nu_s,
                                                              self.nu_s_a, self.slack_nu_p, self.slack_nu_n)
                if use_weight_reg:
                    # TODO: Something more efficient than adding the quadratic term like this
                    self.scp_opt.setObjective(gp.LinExpr([*lin_expr_reward, *pen_reward_list]) - quad_expr_reward,
                                              gp.GRB.MAXIMIZE)
                else:
                    self.scp_opt.setObjective(gp.LinExpr([*lin_expr_reward, *pen_reward_list]), gp.GRB.MAXIMIZE)

            self.update_constraint_time += time.time() - curr_time

            # Solve the optimization problem
            curr_time = time.time()
            self.scp_opt.optimize()
            self.total_solve_time += time.time() - curr_time

            next_pol = {o: {a: self.sigma[o][a].x for a in actList} for o, actList in self._pomdp.obs_actions.items()}

            curr_time = time.time()
            ent_reward_n, nu_s_k_n, nu_s_a_k_n = \
                self.verify_solution(self.bellman_opt, self.nu_s_ver, next_pol, self.bellman_constr)
            self.checking_policy_time += time.time() - curr_time

            latest_ent_n = ent_reward_n

            # Get the reward along trajectories (multiplied by mu_feat)
            traj_reward = sum(
                coeff * val * self._options.mu for coeff, val
                in self.compute_expected_reward(nu_s_a_k_n, weight))

            # Add the total expected reward
            ent_reward_n += traj_reward

            # Add ||(feat_counts - expected reward)||^2 if using regularization on weight
            if use_weight_reg:
                ent_reward_n -= self.compute_quad_featcount_value(nu_s_a_k, feat_counts) * (self._options.rho / 2)

            # Check if the new policy improves over the last obtained policy
            if ent_reward_n > ent_reward:
                pol_k = next_pol
                nu_s_k = nu_s_k_n
                nu_s_a_k = nu_s_a_k_n
                ent_reward = ent_reward_n
                latest_ent = latest_ent_n
                trust_region = np.minimum(set_trust_region['aug'](trust_region), self._max_trust_region)
                past_rejected = False

            else:
                trust_region = set_trust_region['red'](trust_region)
                past_rejected = True
                if self._options.verbose:
                    print('Iter {}: ----> reject the current step.'.format(i))

            if self._options.verbose:
                # print('Iter {}: finding the state and state-action visitation count given a policy.'.format(i))
                # print('Iter {}: optimal policy: {}.'.format(i, pol_k))
                print('Iter {}: entropy reward: {}, trust region: {}.'.format(i, ent_reward, trust_region))
                print('Iter {}: update time: {}s, checking time: {}s, solve time: {}s.'.format(i,
                                                                                              self.update_constraint_time,
                                                                                              self.checking_policy_time,
                                                                                              self.total_solve_time))

            if trust_region < set_trust_region['lim']:
                if self._options.verbose:
                    print('Iter {}: ----> min trust value reached.'.format(i))
                # just to prevent trust region from shrinking forever
                trust_region = self._trust_region
                break

            minimax_value = 0
            if feat_counts is not None:
                minimax_value = ent_reward - sum(
                    weight * feat for weight, feat in zip(weight.values(), feat_counts.values())
                ) * self._options.mu_feat
            
            true_reward = sum(
                coeff * val * self._options.mu for coeff, val
                in self.compute_expected_reward(nu_s_a_k_n, true_weight))

            if self._options.verbose:
                # compare belief feature counts and expected feature counts
                pol_feat_counts = dict()
                for f_name, val in weight.items():
                    feat = self._pomdp.reward_feats[f_name]
                    feat_pol = sum([feat[(s, a)] * nu_s_a_val
                                    for s, nu_s_a_t in nu_s_a_k_n.items()
                                    for a, nu_s_a_val in nu_s_a_t.items()
                                    ])
                    pol_feat_counts[f_name] = feat_pol

                if feat_counts is not None:
                    print('Iter {}: belief_feat_counts: {}.'.format(i,
                                                                    ' '.join(f'{k}: {v}' for k, v in feat_counts.items())))
                print('Iter {}: pol_feat_counts: {}.'.format(i,
                                                             ' '.join(f'{k}: {v}' for k, v in pol_feat_counts.items())))

            # Save the solver statistics if the filepath is provided
            extra_args = (latest_ent, nu_s_k, nu_s_a_k)

            if irl_filepath is not None:
                solver_stats = {'weight': weight, 'true_weight': true_weight,
                                'pol': pol_k, 'trust_reg_val': trust_region, 'extra_args': extra_args}
                self.write_to_pkl(solver_stats, irl_filepath=irl_filepath)

            if logs_filepath is not None:
                self.write_to_txt('Iter {}: Done with SCP having minimax value {},'
                                  ' entropy + reward {}.'.format(i, minimax_value, ent_reward), logs_filepath=logs_filepath)
                self.write_to_txt('Iter {}: Done with SCP having reward {},'
                                  ' true reward {}.'.format(i, traj_reward, true_reward), logs_filepath=logs_filepath)

            # only update policy within the max_updates
            count_correct_update += 1
            if count_correct_update >= self._options.max_update:
                break

        return pol_k, trust_region, (latest_ent, nu_s_k, nu_s_a_k)

    def compute_optimal_mdp_policy(
            self,
            weight: Dict[str, float],
            sigma: Dict[int, Dict[int, float]] = None
    ) -> Tuple[Dict[int, Dict[int, float]], float, Tuple[float, Dict[int, float], Dict[int, Dict[int, float]]]]:
        """ 
        Given the weight for each feature functions in the underlying MDP model,
        compute the optimal policy that maximizes the expected reward (multiplied by factor mu_feat).
        
        Args:
            weight: A dictionary with its keys being feature name and value being associated weight
            sigma: Provided policy

        Returns:
            res_pol: Optimal policy
            trust_region: Trust region radius: 1 when problem solution is optimal and 0 otherwise
            (res_pol_val, res_nu_s, res_nu_s_a): Objective value, state visitation, state-action visitation
        """
        # Create the optimization problem
        m_opt = gp.Model('Optimal policy of the MDP with Gurobi solver')
        self.total_solve_time = 0       # Total time elapsed
        self.init_encoding_time = 0     # Time for encoding the full problem
        self.update_constraint_time = 0
        self.checking_policy_time = 0

        if self._options.verbose:
            print('Initialize linear MDP to be solved.')

        # Util functions for one/two-dimension dictionaries of positive Gurobi variable
        build_var_1d = lambda pb, data, id: {s: pb.addVar(lb=0, name='{}[{}]'.format(id, s)) for s in data}
        build_var_2d = lambda pb, data, id: {s: {a: pb.addVar(lb=0, name='{}[{},{}]'.format(id, s, a)) for a in data_val}
                                             for s, data_val in data.items()}

        # Store the current time for compute time logging
        curr_time = time.time()

        # Store the state visitation count
        nu_s = build_var_1d(m_opt, self._pomdp.states, 'nu')
        # Store the state action visitation count
        nu_s_a = build_var_2d(m_opt, self._pomdp.state_actions, 'nu')
        # Add the constraints between the state visitation count and the state-action visitation count
        self.constr_state_action_to_state_visitation(m_opt, nu_s, nu_s_a, name='vis_count')
        # Add the bellman equation constraints
        self.constr_bellman_flow(m_opt, nu_s, nu_s_a=nu_s_a, sigma=sigma, gamma=self._options.discount, mdp=True, name='bellman')

        # Obtain time for encoding problem
        self.init_encoding_time += time.time() - curr_time

        # Define the parameters used by Gurobi for this problem
        m_opt.Params.OutputFlag = self._options.verbose_solver
        m_opt.Params.Presolve = 2    # More aggressive presolve step

        # Build the objective function
        lin_expr_reward = gp.LinExpr(self.compute_expected_reward(nu_s_a, weight))

        # Set the objective function to maximize
        m_opt.setObjective(lin_expr_reward, gp.GRB.MAXIMIZE)

        # Solve the problem
        curr_time = time.time()
        m_opt.write('model.lp')
        m_opt.optimize()
        self.total_solve_time += time.time() - curr_time

        # Do some printing
        if m_opt.status == gp.GRB.OPTIMAL:
            print('Time used to build the full Model: {}.'.format(self.init_encoding_time))
            print('Total solving time: {}.'.format(self.total_solve_time))
            print('Optimal expected reward: {}.'.format(m_opt.objVal / self._options.quotmu))
        else:
            return dict(), 0, (0, dict(), dict())

        res_pol = {s: {a: (p.x / nu_s[s].x if nu_s[s].x > ZERO_NU_S else 1.0 / len(act_val))
                       for a, p in act_val.items()}
                   for s, act_val in nu_s_a.items()}
        res_nu_s = {s: nu_s[s].x for s, state_val in nu_s.items()}
        res_nu_s_a = {s: {a: p.x for a, p in act_val.items()} for s, act_val in nu_s_a.items()}
        res_pol_val = m_opt.objVal * self._options.mu

        return res_pol, 1, (res_pol_val, res_nu_s, res_nu_s_a)

    def init_optimization_problem(self, m_opt: gp.Model, no_linearization: bool = False, check_opt: gp.Model = None):
        """ 
        Initialize the linearized subproblem to solve at iteration k
        and parametrized the constraints induced by the linearization
        such that they can be modified without being recreated/deleted later.
        
        Args:
            m_opt: A Gurobi model
            no_linearization: Useful when solving the problem using sequential convex optimization
            check_opt: If not None, it is a Gurobi model for finding feasible solution of the bellman equation

        Returns:
            nu_s: State visitation
            nu_s_a: State-action visitation
            sigma: Policy
            slack_nu_p: Positive part of slack variable
            slack_nu_n: Negative part of slack variable
            constr_lin: Linear constraint
            constr_trust_reg: Trust region constraint
            nu_s_ver: Verified/valid state visitation
            bellman_constr: Bellman flow constraint
        """
        if self._options.verbose:
            print('Initialize subproblem to be solved with linearization: {}.'.format(not no_linearization))

        # Util functions for one/two-dimension dictionaries of positive Gurobi variable
        build_var_1d = lambda pb, data, id: {s: pb.addVar(lb=0, name='{}[{}]'.format(id, s)) for s in data}
        build_var_2d = lambda pb, data, id: {s: {a: pb.addVar(lb=0, name='{}[{},{}]'.format(id, s, a)) for a in data_val}
                                             for s, data_val in data.items()}

        # Store the current time for compute time logging
        curr_time = time.time()

        # Store the state visitation count
        nu_s = build_var_1d(m_opt, self._pomdp.states, 'nu')

        # Store the state action visitation count
        nu_s_a = build_var_2d(m_opt, self._pomdp.state_actions, 'nu')

        # Policy variable as a function of obs and state
        sigma = build_var_2d(m_opt, self._pomdp.obs_actions, 'sig')

        # Add the constraints implied by the policy -> sum_a sigma[o,a] == 1
        for o, act_dict in sigma.items():
            m_opt.addLConstr(gp.LinExpr([(1, sigma_o_a) for a, sigma_o_a in act_dict.items()]),
                             gp.GRB.EQUAL, 1,
                             name='sum_pol[{}]'.format(o))
            for a, sigma_o_a in act_dict.items():
                m_opt.addConstr(sigma_o_a >= self._options.policy_epsilon, name='sigma_o_a_epsilon[{},{}]'.format(o, a))

        # Add the constraints between the state visitation count and the state-action visitation count
        self.constr_state_action_to_state_visitation(m_opt, nu_s, nu_s_a, name='vis_count')

        # Add the bellman equation constraints
        self.constr_bellman_flow(m_opt, nu_s, nu_s_a=nu_s_a, sigma=None, gamma=self._options.discount, name='bellman')

        # Add the parameterized trust region constraint on the policy
        constr_trust_reg = self.constr_trust_region(m_opt, sigma)

        # Create variables of the problem to find admissible visitation count given a policy
        assert check_opt is not None
        nu_s_ver = build_var_1d(check_opt, self._pomdp.states, 'nu')

        # Create a dummy policy to initialize parameterized the bellman constraint
        dummy_pol = {o: {a: 1.0 / len(actList) for a in actList} for o, actList in self._pomdp.obs_actions.items()}

        # Add the bellman constraint knowing the policy
        bellman_constr = self.constr_bellman_flow(check_opt, nu_s_ver, nu_s_a=None, sigma=dummy_pol,
                                                  gamma=self._options.discount, name='bellman')

        if no_linearization:
            # Add the bilinear constraint into the problem (won't need to update)
            self.constr_bilinear(m_opt, nu_s, nu_s_a, sigma, name='bil_constr')
            self.init_encoding_time += time.time() - curr_time
            return nu_s, nu_s_a, sigma, None, None, list(), constr_trust_reg, nu_s_ver, bellman_constr

        # If linearization of the nonconvex constraint is enabled, create the slack variables
        # Create the slack variables that will be used for linearizing constraints
        slack_nu_p = build_var_2d(m_opt, self._pomdp.state_actions, 's1p')
        slack_nu_n = build_var_2d(m_opt, self._pomdp.state_actions, 's1n')

        # Add the parametrized linearized constraint
        constr_lin = self.constr_linearized_bilr(m_opt, nu_s, nu_s_a, sigma, slack_nu_p, slack_nu_n, name='lin_bil')

        # Save the encoding compute time the of the problem
        self.init_encoding_time += time.time() - curr_time
        return nu_s, nu_s_a, sigma, slack_nu_p, slack_nu_n, constr_lin, constr_trust_reg, nu_s_ver, bellman_constr

    @staticmethod
    def constr_state_action_to_state_visitation(m_opt, nu_s, nu_s_a, name='vis_count'):
        """
        Encode the constraint between nu_s and nu_s_a. Basically, compute nu_s = sum_a nu_s_a.
        
        Args:
            m_opt: The Gurobi model of the problem
            nu_s: State visitation count
            nu_s_a: State-action visitation count
            name: Name of the constraint
        """
        for s, nu_s_v in nu_s.items():
            m_opt.addLConstr(gp.LinExpr([*((1, nu_s_a_val) for a, nu_s_a_val in nu_s_a[s].items()), (-1, nu_s_v)]),
                             gp.GRB.EQUAL, 0, name='{}[{}]'.format(name, s))

    def constr_bellman_flow(self, m_opt, nu_s, nu_s_a=None, sigma=None, gamma=1.0, mdp=False, name='bellman'):
        """
        Compute for all states the constraints by the Bellman flow equation.
        This function allows one of nu_s_a or sigma to be None. When both
        are provided, only policy is considered and nu_s_a is disregarded.

        Args:
            m_opt: The Gurobi model of the problem
            nu_s: State visitation count
            nu_s_a: State-action visitation count
            sigma: Policy (not a Gurobi) variable
            gamma: The discount factor
            mdp: Whether it is a policy in the MDP
            name: Name of the constraint

        Returns:
            dict_constr: Dictionary of constraints
        """
        dict_constr = dict()
        for s, nu_s_v in nu_s.items():
            if sigma is not None:
                dic_coeff = self.extract_varcoeff_from_bellman(s, sigma, gamma, mdp)
                val_expr = gp.LinExpr([
                    *((sum(coeff_v), nu_s[pred_s]) for pred_s, coeff_v in dic_coeff.items()), (-1, nu_s_v)])
            else:
                val_expr = gp.LinExpr([*((t_prob * gamma, nu_s_a[pred_s][a])
                                         for (pred_s, a, t_prob) in self._pomdp.pred.get(s, [])), (-1, nu_s_v)])

            # Add the linear constraint -> RHS corresponds to the probability the state is an initial state
            dict_constr[s] = m_opt.addLConstr(val_expr, gp.GRB.EQUAL,
                                              -self._pomdp.state_init_probs.get(s, 0),
                                              name='{}[{}]'.format(name, s))
        return dict_constr

    def constr_bilinear(self, m_opt, nu_s, nu_s_a, sigma, name='bilinear'):
        """
        Enforce the bilinear constraint.

        Args:
            m_opt: The Gurobi model of the problem
            nu_s: State visitation count
            nu_s_a: State-action visitation count
            sigma: Policy variable
            name: Name of the constraint
        """
        for s, nu_s_v in nu_s.items():
            obs_distr = self._pomdp.state_obss[s]
            for a, nu_s_a_val in nu_s_a[s].items():
                m_opt.addConstr(nu_s_a_val - nu_s_v * sum(sigma[o][a] * p for o, p in obs_distr.items()) == 0,
                                name='{}[{},{}]'.format(name, s, a))

    def compute_quad_featcount(self, nu_s_a, feat_counts):
        """
        Compute the Gurobi quadratic expression for ||(feat_counts - expected features)||^2

        Args:
            nu_s_a: The state-action visitation count Gurobi variable
            feat_counts: The expected feature counts

        Returns:
            quad_term: Quadratic cost term
        """
        quad_term = 0
        for rname, feat_val in feat_counts.items():
            feat_dict = self._pomdp.reward_feats[rname]
            temp_v = gp.LinExpr(
                [(feat_dict[(s, a)], nu_s_a_val)
                 for s, nu_s_a_t in nu_s_a.items()
                 for a, nu_s_a_val in nu_s_a_t.items()]
            )
            temp_v.addConstant(-feat_val)
            quad_term += temp_v * temp_v

        return quad_term

    def compute_quad_featcount_value(self, nu_s_a, feat_counts):
        """
        Compute the actual value of ||(feat_counts - expected features)||^2,
        given the value of the state-action visitation count and the feat counts.

        Args:
            nu_s_a: The state-action visitation count
            feat_counts: The expected feature counts

        Returns:
            quad_term: Quadratic cost term
        """
        quad_term = 0
        for rname, feat_val in feat_counts.items():
            feat_dict = self._pomdp.reward_feats[rname]
            temp_v = sum(
                [(feat_dict[(s, a)] * nu_s_a_val)
                 for s, nu_s_a_t in nu_s_a.items()
                 for a, nu_s_a_val in nu_s_a_t.items()]
            )
            temp_v += -feat_val
            quad_term += temp_v * temp_v

        return quad_term

    def constr_linearized_bilr(self, m_opt, nu_s, nu_s_a, sigma, slack_nu_p, slack_nu_n, name='lin_bil'):
        """
        Return a parametrized linearized constraint of the
        bilinear constraint involving nu_s, nu_s_a, and sigma
        Each term sigma[o][a] is associated with the coefficient O(o|s)*nu_s^k
        Each term nu_s[s] is associated with the coefficient sum_o O(o|s) sigma[o|s]^k
        The RHS (constant) is associated with the value

        Args:
            m_opt: The Gurobi model of the problem
            nu_s: State visitation count
            nu_s_a: State-action visitation count
            sigma: Policy variable
            slack_nu_p: The (pos) slack variable to render the linear constraint feasible
            slack_nu_n: The (neg) slack variable to render the linear constraint feasible
            name: Name of the constraint

        Returns:
            list_constr: List of constraints
        """
        list_constr = list()
        for s, nu_s_v in nu_s.items():
            obs_distr = self._pomdp.state_obss[s]
            for a, nu_s_a_val in nu_s_a[s].items():
                val_expr = gp.LinExpr([*((1, sigma[o][a]) for o, p in obs_distr.items()),
                                       (1, nu_s_v), (-1, nu_s_a_val),
                                       (1, slack_nu_p[s][a]), (-1, slack_nu_n[s][a])])
                list_constr.append(((s, a), m_opt.addLConstr(val_expr, gp.GRB.EQUAL, 0,
                                                             name='{}[{},{}]'.format(name, s, a))))
        return list_constr

    @staticmethod
    def constr_trust_region(m_opt, sigma):
        """
        Parametrized the linear constraint by the trust region and returned
        the saved constraint to be modified later in the algorithm.

        Args:
            m_opt: Gurobi model
            sigma: observation based policy

        Returns:
            list_constr: List of constraints.
        """
        list_constr = list()
        for o, a_dict in sigma.items():
            for a, pol_val in a_dict.items():
                list_constr.append(((o, a, True), m_opt.addLConstr(pol_val,
                                                                   gp.GRB.GREATER_EQUAL, 0,
                                                                   name='trust_inf[{},{}]'.format(o, a))))
                list_constr.append(((o, a, False), m_opt.addLConstr(pol_val,
                                                                    gp.GRB.LESS_EQUAL, 1.0,
                                                                    name='trust_sup[{},{}]'.format(o, a))))
        return list_constr

    def compute_expected_reward(self, nu_s_a, weight):
        """
        Provide a linear expression for the expected features
        given a weight dictionary and the feature count.

        Args:
            nu_s_a: The state-action visitation count
            weight: The weight dictionary for the feature function

        Returns:
            val_expr: The list of (value, prob)
        """
        val_expr = [(feat[(s, a)] * weight[rName] * self._options.quotmu, nu_s_a_val)
                    for rName, feat in self._pomdp.reward_feats.items()
                    for s, nu_s_a_t in nu_s_a.items()
                    for a, nu_s_a_val in nu_s_a_t.items()]
        return val_expr

    def update_linearized_bil_sonstr(self, m_opt, constr_lin, nu_s_past, sigma_past, nu_s, sigma):
        """
        Update the constraints implied by the linearization of the bilinear constraint
        around the solution of the past iterations

        Args:
            m_opt: Gurobi model
            constr_lin: Bilinear constraints
            nu_s_past: The state-action visitation count in the last iteration
            sigma_past: The observation based policy in the last iteration
            nu_s: The state visitation count
            sigma: The observation based policy
        """
        for ((s, a), constrV) in constr_lin:
            curr_nu = nu_s_past[s]
            obs_distr = self._pomdp.state_obss[s]
            prod_obs_prob_sigma_past = sum(p * sigma_past[o][a] for o, p in obs_distr.items())
            constrV.RHS = curr_nu * prod_obs_prob_sigma_past
            # Update all the coefficients associated to sigma[o][a]
            for o, p in obs_distr.items():
                m_opt.chgCoeff(constrV, sigma[o][a], curr_nu * p)
            # Update the coefficient associated to nu_s[s]
            m_opt.chgCoeff(constrV, nu_s[s], prod_obs_prob_sigma_past)

    def update_constr_and_trust_region(self, m_opt, constr_lin, constr_trust,
                                       nu_s_past, sigma_past, nu_s, sigma, curr_trust,
                                       no_linearization=False, only_trust=False):
        """
        Update the constraint implied by the linearization of the bilinear constraint and
        the trust region constraint using the solutions of the past iteration.

        Args:
            m_opt: A Gurobi model of the problem
            constr_lin: The linearized constraints
            constr_trust: The trust region constraints
            nu_s_past: State visitation count obtained at last iteration
            sigma_past: Policy obtained at the last iteration
            nu_s: State visitation to be obtained at the current iteration
            sigma: Policy to be obtained at the current iteration
            curr_trust: The current trust region
            no_linearization: No use of linearization on bilinear constraints
            only_trust: If True only update the trust region
        """
        # Start by updating the constraint by the trust region
        inv_trust = 1.0 / curr_trust
        for ((o, a, t_rhs), constr_v) in constr_trust:
            # t_rhs is True corresponds to lower bound of the trust region and inversely
            constr_v.RHS = sigma_past[o][a] * (inv_trust if t_rhs else curr_trust)

        # If update only the trust region is allowed then return
        if only_trust:
            return

        if not no_linearization:
            # Now update the constraints from the linearization
            self.update_linearized_bil_sonstr(m_opt, constr_lin, nu_s_past, sigma_past, nu_s, sigma)

    def compute_entropy_reward(self, nu_s_past, nu_s_a_past, nu_s, nu_s_a, slack_nu_p, slack_nu_n, add_entropy=True):
        """
        Return a linearized entropy function around the solution of the past
        iteration given by nu_s_past and nu_s_a_past.

        Args:
            nu_s_past: State visitation at the past iteration
            nu_s_a_past: State-action visitation at the past iteration
            nu_s: State visitation
            nu_s_a: State-action visitation
            slack_nu_p: The (pos) slack variable to render the linear constraint feasible
            slack_nu_n: The (neg) slack variable to render the linear constraint feasible
            add_entropy: If True add entropy terms

        Returns:
            list_reward_term: list of reward terms
        """
        # Save the pair coeff variable for each term in the linear cost function
        list_reward_term = list()

        for s, nu_s_val in nu_s.items():
            if slack_nu_p is not None and slack_nu_n is not None:
                nu_s_past_val, nu_s_a_past_sval, slack_nu_p_sval, slack_nu_n_sval = nu_s_past[s], nu_s_a_past[s], \
                slack_nu_p[s], slack_nu_n[s]
            else:
                nu_s_past_val, nu_s_a_past_sval = nu_s_past[s], nu_s_a_past[s]
            for a, nu_s_a_val in nu_s_a[s].items():
                nu_s_a_past_saval = nu_s_a_past_sval[a]
                # First, add the entropy terms
                if nu_s_past_val > ZERO_NU_S and nu_s_a_past_saval > ZERO_NU_S and add_entropy:
                    nu_ratio = nu_s_a_past_saval / nu_s_past_val
                    list_reward_term.append((nu_ratio / self._options.mu, nu_s_val))
                    list_reward_term.append((-(np.log(nu_ratio) + 1) / self._options.mu, nu_s_a_val))

                if slack_nu_p is not None and slack_nu_n is not None:
                    # Then add the cost associated to the linearization errors
                    list_reward_term.append((-1, slack_nu_p_sval[a]))
                    list_reward_term.append((-1, slack_nu_n_sval[a]))

        return list_reward_term

    def extract_varcoeff_from_bellman(self, state, sigma, gamma=1.0, mdp=False):
        """
        Utility function that given a policy, provides the coefficient of each
        variable nu_s[pred(state)] (in the linear expression) for all predecessor of state.

        Args:
            state: The current state of the pomdp
            sigma: The underlying policy
            gamma: Discount factor
            mdp: Whether it is a policy in the MDP

        Returns:
            dict_pred_coeff: Variable to store the coefficient associated to nu_s[pred(state)]
        """
        dict_pred_coeff = dict()
        # Get from the past optimal value, the coefficient for each variable in the bellman constraint
        for (pred_s, a, t_prob) in self._pomdp.pred.get(state, []):
            if pred_s not in dict_pred_coeff:
                dict_pred_coeff[pred_s] = list()
            if not mdp:
                for o, p in self._pomdp.state_obss[pred_s].items():
                    dict_pred_coeff[pred_s].append(sigma[o][a] * p * t_prob * gamma)
            else:
                dict_pred_coeff[pred_s].append(sigma[pred_s][a] * 1 * t_prob * gamma)
        return dict_pred_coeff

    def verify_solution(self, m_opt, nu_s, policy, constr_bellman):
        """
        Given a policy that is solution of the past iteration,
        This function computes the corresponding state and state-action
        visitation count that satisfy the bellman equation flow.
        Then, it computes the objective attained by this policy.

        Args:
            m_opt: The gurobi model encoding the solution of the bellman flow
            nu_s: State visitation count of m_opt problem
            policy: The optimal policy obtained during this iteration
            constr_bellman: Constraint representing the bellman equation

        Returns:
            ent_reward: Entropy reward
            res_nu_s: State visitation
            res_nu_s_a: State-action visitation
        """
        # Update the optimization problem with the given policy
        for s in nu_s:
            dic_coeff = self.extract_varcoeff_from_bellman(s, policy, gamma=self._options.discount)
            for pred_s, coeff_v in dic_coeff.items():
                sum_coeff = sum(coeff_v)
                m_opt.chgCoeff(constr_bellman[s], nu_s[pred_s], sum_coeff - (1 if pred_s == s else 0))

        # Now solve the problem to get the corresponding state, state-action visitation count
        m_opt.setObjective(0, gp.GRB.MINIMIZE)
        m_opt.optimize()

        # Save the resulting state and state_action value
        res_nu_s = {s: val.x for s, val in nu_s.items()}
        res_nu_s_a = {s: {a: res_nu_s[s] * sum(p * policy[o][a] for o, p in self._pomdp.state_obss[s].items())
                          for a in act_list}
                      for s, act_list in self._pomdp.state_actions.items()}

        ent_reward = sum(0 if (res_nu_s_a_val <= ZERO_NU_S or res_nu_s_val <= ZERO_NU_S)
                         else (-np.log(res_nu_s_a_val / res_nu_s_val) * res_nu_s_a_val)
                         for s, res_nu_s_val in res_nu_s.items()
                         for a, res_nu_s_a_val in res_nu_s_a[s].items())

        return ent_reward, res_nu_s, res_nu_s_a

    def obtain_visitation_from_policy(self, policy):
        """
        Given a policy, this function computes the corresponding state
        and state-action visitation count that satisfy the bellman equation flow.
        Then, it computes the entropy reward attained by this policy.

        Args:
            policy: The optimal policy obtained during this iteration

        Returns:
            ent_reward: Entropy reward
            res_nu_s: State visitation
            res_nu_s_a: State-action visitation
        """
        ent_reward, res_nu_s, res_nu_s_a = (
            self.verify_solution(self.bellman_opt, self.nu_s_ver, policy, self.bellman_constr))

        return ent_reward, res_nu_s, res_nu_s_a

    def write_to_pkl(self, solver_stats: Dict[str, Union[int, float, list, dict]],
                     irl_filepath: Union[Path, str] = None) -> None:
        """
        Save solver statistics to file.

        Args:
             solver_stats: Dictionary that contains solver statistics
             irl_filepath: Filepath to irl solver statistics
        """
        with open(os.path.join(self._pomdp.stats_dir, irl_filepath), 'wb') as f:
            pickle.dump(solver_stats, f)
    
    def write_to_txt(self, info: str, to_screen: bool = True,
                     logs_filepath: Union[Path, str] = None) -> None:
        """
        Write progress information to file.

        Args:
             info: Information to track solution progress
             to_screen: whether also output info to screen
             logs_filepath: File path to logs information
        """
        if to_screen:
            print(info)
        with open(logs_filepath, 'a') as f:
            f.write(info)
            f.write('\n')
