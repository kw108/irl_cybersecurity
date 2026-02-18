"""
Global configuration parameters for the IRL Cybersecurity project
"""
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import os


class GlobalParams:
    """Centralized configuration parameters."""

    # Gurobipy license
    gurobipy_license = '/home/wang1899/PycharmProjects/irl_cybersecurity/gurobi.lic'

    # BAG structure
    NODES = {
        '0': ('LEAF', 0.69),
        '1': ('LEAF', 0.62),
        '2': ('LEAF', 0.53),
        '3': ('OR', 0.00),
        '4': ('OR', 0.00),
        '5': ('OR', 0.00),
        '6': ('OR', 0.00),
        '7': ('AND', 0.00),
        # '8': ('OR', 0.00),
        # '9': ('AND', 0.00)
    }
    EDGES = [
        ('0', '3', 0.57),
        ('1', '3', 0.57),
        ('2', '3', 0.57),
        ('2', '4', 0.4329),
        ('3', '5', 0.8054),
        ('3', '6', 0.7722),
        ('4', '5', 0.8054),
        ('4', '6', 0.7722),
        ('4', '7', 0.3549),
        ('5', '6', 0.7722),
        # ('5', '8', 0.34),
        # ('5', '9', 0.3811),
        ('6', '7', 0.3549),
        # ('6', '9', 0.3811)
    ]
    num_nodes = len(NODES)

    # POMDP parameters
    obs_err = 0.10      # 0.01, 0.05, 0.10
    discount = 0.95
    stats_dir = 'stats_dir'
    err_pct = int(obs_err * 100)
    cache_file = f'pomdp_E{err_pct:03d}.pkl'

    # Runtime configuration
    random_seed = 42

    # Simulating trajectories
    num_trajs = 10      # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    max_steps_per_traj = 100

    # Thompson sampling and computing feature counts
    # Set num_runs_T="inf" for true transition kernel
    num_runs_T = 400    # 400, 600, 800, 1000, "inf"
    num_steps = 100
    save_interval = 100

    # MCE-IRL in POMDP
    true_weight = {'s1': 1, 's2': 2, 's3': 3, 's4': 4, 's5': 1, 's6': 2, 's7': 3, 's8': 4,
                   'a1': 1, 'a2': 2, 'a3': 3, 'a4': 4, 'a5': 1, 'a6': 2, 'a7': 3, 'a8': 4}
    init_weight = {'s1': 2.5, 's2': 2.5, 's3': 2.5, 's4': 2.5, 's5': 2.5, 's6': 2.5, 's7': 2.5, 's8': 2.5,
                   'a1': 2.5, 'a2': 2.5, 'a3': 2.5, 'a4': 2.5, 'a5': 2.5, 'a6': 2.5, 'a7': 2.5, 'a8': 2.5}
    num_feats = len(true_weight)

    @staticmethod
    def logs_filepath() -> Union[Path, str]:
        """Create filepath for logs information."""
        return 'logs_info.txt'

    def N_1_filepath(self) -> Union[Path, str]:
        return f'N_1_E{self.err_pct:03d}_T{self.num_runs_T}.npy'

    def N_0_filepath(self) -> Union[Path, str]:
        return f'N_0_E{self.err_pct:03d}_T{self.num_runs_T}.npy'

    def solver_filepath(self) -> Union[Path, str]:
        """Create filepath for forward solver."""
        return f'solver_N{self.num_nodes}_F{self.num_feats}_R{self.random_seed}' \
               f'_E{self.err_pct:03d}_T{self.num_runs_T}.pkl'

    def demo_filepath(self) -> Union[Path, str]:
        """Create filepath for forward solver."""
        return f'demo_N{self.num_nodes}_F{self.num_feats}_R{self.random_seed}' \
               f'_E{self.err_pct:03d}_S{self.num_trajs}_T{self.num_runs_T}.pkl'

    def fc_filepath(self) -> Union[Path, str]:
        """Create filepath for forward solver."""
        return f'fc_N{self.num_nodes}_F{self.num_feats}_R{self.random_seed}' \
               f'_E{self.err_pct:03d}_S{self.num_trajs}_T{self.num_runs_T}.pkl'

    def irl_filepath(self) -> Union[Path, str]:
        """Create filepath for inverse solver."""
        return f'irl_N{self.num_nodes}_F{self.num_feats}_R{self.random_seed}' \
               f'_E{self.err_pct:03d}_S{self.num_trajs}_T{self.num_runs_T}.pkl'
