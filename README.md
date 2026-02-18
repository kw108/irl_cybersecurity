# Maximum Causal Entropy Inverse Reinforcement Learning (MCE-IRL) over Partially Observable Markov Decision Processes (POMDPs)

## Overview

A project that leverages Maximum Causal Entropy Inverse Reinforcement Learning (MCE-IRL) 
over Partially Observable Markov Decision Processes (POMDPs) with Sequential Convex Programming (SCP).

## Features

- Adaptation of MCE-IRL from MDP to POMDP
- Scalability with thousands of states and hundreds of actions per state
- Efficient solution with SCP

## Prerequisites

- Python version 3.7 or later
- Gurobipy version 11.0.3 or later (academic or commercial license required)

## Usage

1. Apply for or purchase a Gurobi license, store it locally, and specify filepath in globals.py.
2. Modify parameters in globals.py and bag2pomdp.py according to use cases.
3. Perform Thompson sampling to estimate transition kernel using script pomdp_thompson_sampling.py.
4. Optimize MCE-IRL policy over POMDPs using script mce_irl_forward.py. 
5. Simulate expert trajectories with pomdp_simulating_trajectories.py using script pomdp_simulating_trajectories.py.
6. Compute belief feature counts with trajectories and transition kernel using script pomdp_feature_counts.py.
7. Evaluate proposed method with benchmarks using script mce_irl_inverse.py.

## Known Issues

### Issue One

- **Issue Description**: Program occasionally got killed or halted while running Gurobi solver.
- **Possible Cause**: Randomization in Gurobi solver affects memory usage; sparsity in transition kernel.
- **Temporary Workaround**: Re-run the evaluation; add tiny epsilon to all transition kernel entries and normalize.
- **Status**: Fix pending. The normalized true transition kernel and estimated transition kernel using Thompson
sampling with pseudo counts significantly slow down execution.

## Acknowledgements

The majority of MCE-IRL solver code was forked and modified from the repository of
[Task-Guided IRL in POMDPs that Scales](https://github.com/wuwushrek/MCE_IRL_POMDPS).
However, the reward in the POMDP in their repository is observation-based, and thus is
fundamentally different from conventional POMDPs.
