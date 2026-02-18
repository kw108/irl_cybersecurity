"""
POMDP for Bayesian Attack Graphs (BAG)

This module implements a Partially Observable Markov Decision Process (POMDP) for
modeling security in Bayesian Attack Graphs. It provides functionality for computing
transition kernels, observation kernels, reward functions, and belief updates.
"""

import numpy as np
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from numpy.typing import NDArray


class BAG:
    """Bayesian Attack Graph representing system vulnerabilities and dependencies.

    Attributes:
        nodes: Dictionary mapping node IDs to node features
        edges: List of edges with transition probabilities
        in_neighbors: Dictionary mapping destination nodes to (source, probability) pairs
    """

    def __init__(
            self,
            nodes: Dict[str, Tuple[str, float]],
            edges: List[Tuple[str, str, float]]
    ):
        """Initialize the Bayesian Attack Graph.

        Args:
            nodes: Dictionary mapping node IDs to (node_type, probability) tuples
            edges: List of (source, destination, probability) tuples
        """
        self.nodes = nodes
        self.edges = edges
        self.in_neighbors: Dict[str, List[Tuple[str, float]]] = {}
        self._build_graph()

    def _build_graph(self) -> None:
        """Build adjacency list representation of the graph."""
        self.in_neighbors = {node_id: [] for node_id in self.nodes}

        for src, dst, prob in self.edges:
            self.in_neighbors[dst].append((src, prob))

    @property
    def num_nodes(self) -> int:
        """Get number of nodes in the graph."""
        return len(self.nodes)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save graph to file."""
        graph = {'nodes': self.nodes, 'edges': self.edges}
        with open(filepath, 'w') as f:
            json.dump(graph, f, indent=4)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'BAG':
        """Load graph from file."""
        with open(filepath, 'r') as f:
            graph = json.load(f)
        return cls(graph['nodes'], graph['edges'])


class POMDP:
    """Partially Observable Markov Decision Process for Bayesian Attack Graphs.

    This class computes transition and observation kernels for the POMDP
    based on the underlying Bayesian Attack Graph.
    """

    def __init__(self, bag: BAG, obs_err, stats_dir: Union[Path, str]):
        """Initialize POMDP from Bayesian Attack Graph.

        Args:
            bag: BayesianAttackGraph instance that underlies POMDP
            obs_err: Observation error rate
            stats_dir: Directory to save statistics files
        """
        self.bag = bag
        self.num_nodes = bag.num_nodes
        self.est_trans_kernel = None    # May need to estimate transition kernel

        # TODO: change this part for different test cases when necessary
        self.feat_names = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8',
                           'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8']
        self.num_feats = len(self.feat_names)
        self.obs_err = obs_err          # Binary observation error rate
        self.pseudo_obs = 1e-3          # Pseudo observation count for Thompson Sampling
        self.stats_dir = stats_dir
        os.makedirs(self.stats_dir, exist_ok=True)

        # TODO: change the _compute_reward_feats function when necessary

        # Initialize transition kernel and observation kernel with proper dimensions
        self._init_kernels()

        # Initialize reward features
        self._init_reward_feats()

    def _init_kernels(self) -> None:
        """Initialize transition and observation kernels."""
        num_nodes = self.num_nodes
        # State: 0 secure, 1 compromised
        self.num_states = 2 ** num_nodes
        # Action: 0 no action, 1 reimaging (at most choose 2 nodes to reimage)
        # TODO: change action spaces when necessary
        self.num_actions = 1 + num_nodes + num_nodes * (num_nodes - 1) // 2
        # Observation: 0 secure, 1 compromised
        self.num_obss = 2 ** num_nodes

        # Define dictionaries for states, actions, and observations
        self.states = list(range(self.num_states))
        # TODO: change action spaces when necessary
        self.actions = [a for a in range(self.num_states) if sum(self._int_to_base(a, 2, num_nodes)) <= 2]
        self.action_map = {a: i for i, a in enumerate(self.actions)}
        self.reverse_action_map = {i: a for i, a in enumerate(self.actions)}
        self.state_actions = {s: self.actions for s in range(self.num_states)}
        self.obs_actions = {o: self.actions for o in range(self.num_obss)}

        # TODO: change state initialization distribution when necessary
        self.state_init_probs = {s: 0.0 for s in range(self.num_states)}
        self.state_init_probs[0] = 1.0
        # self.state_init_probs = {s: 1.0 / self.num_states for s in range(self.num_states)}

        # Transition kernel: T(s' = 1 | s, a) -> TODO: change when necessary
        self.trans_kernel = np.zeros((self.num_states, self.num_actions, num_nodes))
        self._compute_trans_kernel()

        # Observation kernel: Z(o = 1 | s) -> TODO: change when necessary
        self.obs_kernel = np.zeros((self.num_states, num_nodes))
        self._compute_obs_kernel()

        # Define and compute dictionaries with probabilities
        self.state_obss = {s: {o: self._compute_product_probs(self._int_to_base(o, 2, num_nodes),
                                                              self.obs_kernel[s])
                               for o in range(self.num_obss)}
                           for s in range(self.num_states)}

        # State-predecessor dictionary (needed by MCE-IRL solver)
        self.pred = {s: [] for s in range(self.num_states)}
        self._compute_pred()

    @staticmethod
    def _int_to_base(value: int, base: int, length: int) -> List[int]:
        """Convert integer to fixed-length base-n list such that LSB is indexed by 0.

        Args:
            value: Integer to convert
            base: Base for conversion (2 for binary, 3 for ternary, etc.)
            length: Desired length of output list

        Returns:
            List of digits in specified base such that LSB is indexed by 0
        """
        digits = []
        while value > 0:
            value, remainder = divmod(value, base)
            digits.append(remainder)

        return digits + [0] * (length - len(digits))

    @staticmethod
    def _base_to_int(digits: List[int], base: int) -> int:
        """Convert fixed-length base-n list to integer such that LSB is indexed by 0.

        Args:
            digits: List of integers to convert
            base: Base for conversion (2 for binary, 3 for ternary, etc.)

        Returns:
            value: Integer in specified base such that LSB is indexed by 0
        """
        value = 0
        for i, digit in enumerate(digits):
            value += digit * (base ** i)

        return value

    @staticmethod
    def _compute_product_probs(binaries: List[int], probs: List[float]) -> float:
        """Compute product of probabilities based on binary list and probabilities list.

        Args:
            binaries: Binary list indicating which probabilities to use
            probs: List of probabilities indicating compromise

        Returns:
            Product of probabilities or their complements

        Raises:
            AssertionError: lists are not equal in length or have non-positive length
        """
        assert len(binaries) == len(probs) and len(binaries) > 0, \
            'Lists are not equal in length or have non-positive length.'

        adjusted_probs = list(map(lambda b, p: p if b == 1 else 1 - p, binaries, probs))

        return np.prod(adjusted_probs, dtype=np.float64)

    def _compute_trans_kernel(self) -> None:
        """Compute complete transition kernel."""
        num_states, num_actions, _ = self.trans_kernel.shape

        for state in self.states:
            for i, action in enumerate(self.state_actions[state]):
                self.trans_kernel[state, i] = self._compute_trans_probs(state, action)

        # self.trans_kernel += 1e-8
        # self.trans_kernel = self.trans_kernel / self.trans_kernel.sum(axis=2, keepdims=True)

    def _compute_trans_probs(self, state: int, action: int) -> NDArray[np.float64]:
        """Compute transition probabilities for all nodes.

        Args:
            state: Current POMDP state
            action: Current POMDP action

        Returns:
            probs: Array of probabilities for each node to be compromised
        """
        node_states = self._int_to_base(state, 2, self.num_nodes)
        node_actions = self._int_to_base(action, 2, self.num_nodes)
        probs = np.zeros(self.num_nodes, dtype=np.float64)

        for node_id in range(self.num_nodes):
            node_key = str(node_id)
            node_feat = self.bag.nodes[node_key]

            if node_actions[node_id] == 1:
                probs[node_id] = 0.0    # Skip if node is being reimaged (action = 1)
                continue

            if node_states[node_id] == 1:
                probs[node_id] = 1.0    # Node already compromised
                continue

            # Node currently secure
            if node_feat[0] == 'AND':
                probs[node_id] = self._compute_and_prob(node_key, node_states)
            elif node_feat[0] == 'OR':
                probs[node_id] = self._compute_or_prob(node_key, node_states)
            else:   # LEAF node
                probs[node_id] = node_feat[1]

        return probs

    def _compute_and_prob(self, node_key: str, node_states: List[int]) -> float:
        """Compute probability of compromise for AND node.

        Args:
            node_key: Key of the AND node
            node_states: List of node states

        Returns:
            Probability of compromise
        """
        in_neighbors = self.bag.in_neighbors.get(node_key, [])

        # Check if all in-neighbors are compromised
        if not all(node_states[int(src)] == 1 for src, _ in in_neighbors):
            return 0.0

        # All in-neighbors compromised, compute joint probability
        return np.prod([prob for _, prob in in_neighbors], dtype=np.float64)

    def _compute_or_prob(self, node_key: str, node_states: List[int]) -> float:
        """Compute probability of compromise for OR node.

        Args:
            node_key: key of the OR node
            node_states: List of node states

        Returns:
            Probability of compromise
        """
        in_neighbors = self.bag.in_neighbors.get(node_key, [])

        # Get compromised in-neighbors
        compromised_in_neighbors = [prob for src, prob in in_neighbors
                                    if node_states[int(src)] == 1]

        if not compromised_in_neighbors:
            return 0.0

        return 1.0 - np.prod([1.0 - p for p in compromised_in_neighbors], dtype=np.float64)

    def _compute_obs_kernel(self) -> None:
        """Compute complete observation kernel.

        Raises:
            AssertionError: If self.obs_err is not in [0,1]
        """
        assert 0 <= self.obs_err <= 1, 'Observation error rate is not in [0, 1].'
        num_states, num_obss = self.obs_kernel.shape

        for state in range(num_states):
            node_states = self._int_to_base(state, 2, self.num_nodes)
            self.obs_kernel[state] = np.array([1 - self.obs_err if s == 1 else self.obs_err for s in node_states],
                                              dtype=np.float64)

    def _init_reward_feats(self) -> None:
        """Initialize reward features."""
        self.reward_features = np.zeros((self.num_states, self.num_actions, self.num_feats))
        for state in self.states:
            for i, action in enumerate(self.state_actions[state]):
                self.reward_features[state, i] = self._compute_reward_feats(state, action)

        self.reward_feats = dict()
        for i, key in enumerate(self.feat_names):
            self.reward_feats[key] = {(s, a): self.reward_features[s, self.action_map[a], i] for s in self.states
                                      for a in self.state_actions[s]}

    def _compute_reward_feats(self, state: int, action: int, sec_lvl: float = 10.0) -> NDArray[np.float64]:
        """Compute reward features for state-action pair.

        Args:
            state: Current POMDP state
            action: Current POMDP action
            sec_lvl: Base security level parameter

        Returns:
            Computed reward features
        """
        trans_probs = self.trans_kernel[state, self.action_map[action]]

        # Number of compromised nodes in next state
        # compromised_nodes = np.sum(np.random.binomial(1, trans_probs))
        sec_feat = [-np.sum(trans_probs[1 * i: 1 * i + 1]) for i in range(self.num_nodes)]

        # Number of actions to reimage nodes
        node_actions = self._int_to_base(action, 2, self.num_nodes)
        action_feat = [-np.sum(node_actions[1 * i: 1 * i + 1]) for i in range(self.num_nodes)]

        reward_feats = sec_feat + action_feat
        # reward_feats = sec_feat + action_feat + [sec_lvl]

        return np.array(reward_feats, dtype=np.float64)

    def _compute_pred(self, T: Optional[NDArray[np.float64]] = None) -> None:
        """Compute (predecessor state, action, probability) tuple list for state.

        Args:
            T: Alternative transition kernel
        """
        trans_kernel = T if T is not None else self.trans_kernel

        for s in range(self.num_states):
            node_states = self._int_to_base(s, 2, self.num_nodes)
            self.pred[s] = [(s_pred, a, prob)
                            for s_pred in range(self.num_states)
                            for a in self.state_actions[s_pred]
                            if (prob := self._compute_product_probs(
                                node_states,
                                trans_kernel[s_pred, self.action_map[a]]
                            )) > 0]

    def _belief_update(
            self,
            belief: NDArray[np.float64],
            action: int,
            next_obs: int,
            T: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        """Update belief state using Bayes' rule.

        Args:
            belief: Current belief state distribution
            action: Current action taken
            next_obs: Next observation received
            T: Alternative transition kernel

        Returns:
            next_belief: Next belief state distribution
        """
        trans_kernel = self.trans_kernel if T is None else T
        next_belief = np.zeros_like(belief, dtype=np.float64)
        next_node_obss = self._int_to_base(next_obs, 2, self.num_nodes)

        for next_state in range(len(belief)):
            next_node_states = self._int_to_base(next_state, 2, self.num_nodes)
            obs_prob = self._compute_product_probs(next_node_obss, self.obs_kernel[next_state])

            if not obs_prob:
                continue

            pred = 0.0
            for state in range(len(belief)):
                trans_prob = self._compute_product_probs(next_node_states,
                                                         trans_kernel[state, self.action_map[action]])
                pred += trans_prob * belief[state]

            next_belief[next_state] = obs_prob * pred

        total = np.sum(next_belief)
        if total > 0:
            next_belief /= total

        return next_belief

    def simulate_pomdp_policy(
            self,
            policy: Dict[int, Dict[int, float]],
            gamma: float,
            num_trajs: int,
            max_steps_per_traj: int,
            seed: Optional[int] = None,
            stop_at_accepting_state: bool = True
    ) -> List[List[Tuple[int, int]]]:
        """Simulates observation-based policy execution in a partially observable environment.

        This method performs multiple Monte Carlo simulations under a given stochastic
        policy. The environment dynamics follow a POMDP model with transition
        and observation kernels.

        Args:
            policy: Mapping from observations to probability distributions over
                actions. Format: {obs: {action: probability, ...}, ...}
            gamma: Gamma in [0,1) for discounting future rewards
            num_trajs: Number of independent Monte Carlo trajectories
            max_steps_per_traj: Maximum horizon for each trajectory
            seed: Random seed for reproducibility. If None, uses random seed.
            stop_at_accepting_state: Whether to terminate trajectory upon reaching
                accepting/terminal state

        Returns:
            obs_trajs: List of trajectories where each trajectory
                is a list of (observation, action) pairs

        Raises:
            AssertionError: If discount_factor is not in [0,1)
        """
        assert 0 <= gamma < 1, f'Discount factor must be in [0,1), got {gamma}.'

        rand_seed = np.random.randint(0, 10000) if seed is None else seed
        np.random.seed(rand_seed)

        obs_trajs = []

        for i in range(num_trajs):
            print(f'Currently running simulation {i}/{num_trajs}')

            state = np.random.choice(range(self.num_states))        # Initialize simulation state

            node_obss = np.random.binomial(1, self.obs_kernel[state])     # Sample observation
            obs = self._base_to_int(node_obss.tolist(), 2)

            obs_traj = []

            for j in range(max_steps_per_traj):
                action = np.random.choice(np.array([a for a in policy[obs]]),
                                          p=np.array([prob for a, prob in policy[obs].items()]))
                obs_traj.append((obs, action))

                # Transition to next state, observation, and belief
                next_node_states = np.random.binomial(1, self.trans_kernel[state, self.action_map[action]])
                next_state = self._base_to_int(next_node_states.tolist(), 2)

                # Termination condition
                if stop_at_accepting_state and next_state == self.num_states - 1:
                    break

                # Update for next iteration
                next_node_obss = np.random.binomial(1, self.obs_kernel[next_state])
                next_obs = self._base_to_int(next_node_obss.tolist(), 2)

                state = next_state
                obs = next_obs

            obs_trajs.append(obs_traj)

        return obs_trajs

    def perform_thompson_sampling(
            self,
            num_runs: int,
            num_steps: int,
            save_interval: int,
            N_1_filepath: Union[Path, str],
            N_0_filepath: Union[Path, str],
            seed: Optional[int] = None
    ) -> None:
        """Learn the transition kernel with Thompson Sampling.

        Args:
            num_runs: Number of episodes
            num_steps: Number of steps per episode
            save_interval: Interval to re-estimate and save transition kernel
            N_1_filepath: Filepath to save N_1
            N_0_filepath: Filepath to save N_0
            seed: Random seed
        """
        assert self.pseudo_obs > 0, \
            'Pseudo observation count for Thompson Sampling is not positive.'

        rand_seed = np.random.randint(0, 10000) if seed is None else seed
        np.random.seed(rand_seed)

        self.est_trans_kernel = np.zeros_like(self.trans_kernel, dtype=np.float64)
        N_1 = np.ones_like(self.trans_kernel, dtype=np.float64) * self.pseudo_obs
        N_0 = np.ones_like(self.trans_kernel, dtype=np.float64) * self.pseudo_obs

        for m in range(num_runs):
            print(f'Currently running episode {m}/{num_runs}.')
            b = np.zeros(self.num_states)
            s = np.random.choice(self.states)
            b[s] = 1
            est_s = s

            for t in range(num_steps):
                a = np.random.choice(self.state_actions[s])
                next_ss = np.random.binomial(n=1, p=self.trans_kernel[s, self.action_map[a]])

                next_s = self._base_to_int(next_ss.tolist(), 2)
                next_os = np.random.binomial(n=1, p=self.obs_kernel[next_s])

                next_o = self._base_to_int(next_os.tolist(), 2)

                next_b = self._belief_update(b, a, next_o, self.est_trans_kernel)
                est_next_s = np.argmax(next_b)
                est_next_ss = self._int_to_base(est_next_s, 2, self.num_nodes)

                N_0[est_s, self.action_map[a]] += (1 - np.array(est_next_ss))
                N_1[est_s, self.action_map[a]] += np.array(est_next_ss)

                s = next_s
                est_s = est_next_s

            # Periodically re-estimate transition kernel and save occurrences
            if (m + 1) % save_interval == 0:
                self.est_trans_kernel = N_1 / (N_1 + N_0)

                tmp_filepath = N_1_filepath.replace(f'T{num_runs}', f'T{(m + 1)}')
                np.save(os.path.join(self.stats_dir, tmp_filepath), N_1)

                tmp_filepath = N_0_filepath.replace(f'T{num_runs}', f'T{(m + 1)}')
                np.save(os.path.join(self.stats_dir, tmp_filepath), N_0)

    def obs_trajs_to_feat_counts(
            self,
            obs_trajs: List[List[Tuple[int, int]]],
            gamma: float,
            save_filepath: Union[Path, str],
            T: Optional[NDArray[np.float64]] = None,
    ) -> Dict[str, float]:
        """Compute belief feature counts from observation trajectories.

        Args:
            obs_trajs: List of observation-action pairs
            gamma: Discount factor
            save_filepath: filepath to save results from all trajectories
            T: Alternative transition kernel

        Returns:
            feat_count: Belief feature counts

        Raises:
            AssertionError: If discount_factor is not in [0,1)
        """
        assert 0 <= gamma < 1, f'Discount factor must be in [0,1), got {gamma}.'
        feat_counts = {key: 0.0 for key in self.reward_feats.keys()}

        T = self.trans_kernel if T is None else T
        num_trajs = len(obs_trajs)
        raw_obs_pol = {o: {a: 0.0 for a in self.obs_actions[o]} for o in self.states}

        for traj_id, obs_traj in enumerate(obs_trajs):
            print(f'Currently running trajectory {traj_id}/{len(obs_trajs)}')

            o = obs_traj[0][0]
            a = obs_traj[0][1]
            raw_obs_pol[o][a] += 1.0

            belief = np.ones(self.num_states) * 1 / self.num_states
            traj_feat_counts = {key: 0.0 for key in self.reward_feats.keys()}

            for s in range(self.num_states):
                feat = self.reward_features[s, self.action_map[a]]
                for feat_val, key in zip(feat, feat_counts.keys()):
                    traj_feat_counts[key] += feat_val * belief[s]

            prev_a = a

            for j, (o, a) in enumerate(obs_traj[1:]):
                raw_obs_pol[o][a] += 1.0

                # Update for next iteration
                belief = self._belief_update(belief, prev_a, o, T)

                # Accumulate discounted reward features
                for s in range(self.num_states):
                    feat = self.reward_features[s, self.action_map[a]]
                    for feat_val, key in zip(feat, feat_counts.keys()):
                        traj_feat_counts[key] += gamma ** (j + 1) * feat_val * belief[s]

                prev_a = a

            # Adjust multiplicative factor according to length of trajectory
            traj_factor = 1 / (1 - gamma ** len(obs_traj))
            feat_counts = {k: feat_counts[k] + traj_feat_counts[k] * traj_factor for k in feat_counts.keys()}
            traj_id += 1

            tmp_filepath = save_filepath.replace(f'S{num_trajs}', f'S{traj_id}')
            with open(tmp_filepath, 'wb') as f:
                pickle.dump({'feat_counts': {k: v / traj_id for k, v in feat_counts.items()},
                             'raw_obs_pol': raw_obs_pol}, f)

        # Normalize by number of simulations
        feat_counts = {k: v / num_trajs for k, v in feat_counts.items()}

        return feat_counts

    def simulate_mdp_policy_value(
            self,
            policy: Dict[int, Dict[int, float]],
            gamma: float,
            num_trajs: int,
            max_steps_per_traj: int,
            weight: Dict[str, float],
            seed: Optional[int] = None
    ) -> float:
        """Simulates policy execution in an MDP environment.

        Args:
            policy: Mapping from observations to probability distributions over
                actions.
            gamma: Gamma in [0,1) for discounting future rewards
            num_trajs: Number of independent Monte Carlo trajectories
            max_steps_per_traj: Maximum horizon for each trajectory
            weight: Dictionary of weight
            seed: Random seed for reproducibility. If None, uses random seed.

        Returns:
            mdp_pol_val: MDP policy value

        Raises:
            AssertionError: If discount_factor is not in [0,1)
        """
        assert 0 <= gamma < 1, f'Discount factor must be in [0,1), got {gamma}.'

        rand_seed = np.random.randint(0, 10000) if seed is None else seed
        np.random.seed(rand_seed)

        val = 0
        for i in range(num_trajs):
            print(f'Currently running simulation {i}/{num_trajs}')

            # Initialize simulation state
            state = np.random.choice(list(range(self.num_states)), p=list(self.state_init_probs.values()))

            for j in range(max_steps_per_traj):
                action = np.random.choice(np.array([a for a in policy[state]]),
                                          p=np.array([prob for a, prob in policy[state].items()]))

                feat = self.reward_features[state, self.action_map[action]]
                val += gamma ** j * sum([f * w for f, w in zip(feat, weight.values())])

                # Transition to next state
                next_node_states = np.random.binomial(1, self.trans_kernel[state, self.action_map[action]])
                next_state = self._base_to_int(next_node_states.tolist(), 2)

                state = next_state

        return val / max(num_trajs, 1)

    def recompute_pred(self, T: Optional[NDArray[np.float64]] = None) -> None:
        """Recompute dictionary of (predecessor state, action, probability) tuple list.

        Args:
            T: Alternative transition kernel
        """
        self._compute_pred(T)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save POMDP to file."""
        print(f'Saving POMDP to {filepath}...')
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'POMDP':
        """Load POMDP from file."""
        print(f'Loading POMDP from {filepath}...')
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    @classmethod
    def create_pomdp_instance(
            cls,
            nodes: Dict[str, Tuple[str, float]],
            edges: List[Tuple[str, str, float]],
            obs_err: float,
            stats_dir: Union[Path, str],
            cache_file: Union[Path, str] = None
    ) -> 'POMDP':
        """Create and initialize POMDP instance.

        Args:
            nodes: Dictionary mapping node IDs to (node_type, probability) tuples
            edges: List of (source, target, probability) tuples
            obs_err: Observation Error rate
            stats_dir: Directory to save statistics files
            cache_file: Filepath for POMDP instance

        Returns:
            Initialized POMDP instance

        Raises:
            AssertionError: Neither pomdp file nor nodes and edges exists
        """
        assert nodes is not None and edges is not None, \
            'Not both nodes and edges exist.'

        if cache_file and Path(cache_file).exists():
            print(f'Loading cached POMDP from {cache_file}...')
            return POMDP.load(cache_file)

        print('Creating new POMDP instance...')
        bag = BAG(nodes, edges)
        pomdp = cls(bag, obs_err, stats_dir)

        if cache_file:
            print(f'Caching POMDP to {cache_file}...')
            pomdp.save(cache_file)

        return pomdp
