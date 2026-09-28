# Sequential Decision-Making Tutorial - Online Facility Assignment

Simulation environment and policies for sequential decision-making in an online facility assignment problem: customers arrive one at a time and must be assigned to a facility (warehouse) with limited capacity.

The description of each policy and of the environment is presented in the paper.

```
Sequential_Decision_Making_Tutorial_Online_Facility_Assignment/
├── env.py                    # simulation environment
├── miscellaneous.py          # shared helpers
├── milp_lp_solver.py         # Gurobi MILP/LP solver
├── instances_generator.py    # generates instances
├── visualize.py              # generates illustrative figures → visualization_examples/
├── export_results.py         # evaluates all policies → results/
├── export_*.ipynb            # LaTeX tables and plots for the paper
├── policies/                 # one file per policy
├── instances/                # test + training instances
├── training/                 # train_* scripts and trained artifacts
├── results/                  # raw results, tables, plots
└── visualization_examples/   # illustrative figures from the paper
```

See [Repository layout (details)](#repository-layout-details) for a full description.

## What's included

The repository contains nearly everything needed to reproduce the paper's outputs without re-running the expensive steps:

- **Results** - the raw evaluation results of every policy are already in `results/`.
- **Trained policies** - the trained artifacts of every policy are already in `training/*_training/`, except for the optimal policy (`policies/exact_value_function_policy.py`).

This means that you can immediately:

- build all the tables and plots (`export_tables.ipynb`, `export_plots_results.ipynb`, `export_plots_training.ipynb`);
- test any trained policy, once you have generated the instances (see below).

You can still regenerate any of these outputs from scratch, but you don't have to. Only two things are not included and must be created by you:

**Instances.** To keep the GitHub repository lightweight, `instances/` is not included. Generate it by running, from the repository root:

```bash
python instances_generator.py
```

Every instance is generated from a fixed seed, so this recreates exactly the instances used in the paper, in the three subfolders described in [`instances/`](#instances). You need them to run policy comparison experiments (`export_results.py`) and also for policy training procedures.

**Exact value tables.** The optimal policy is the other exception. Its value tables add up to over 15 GB, which is too large for the GitHub repository, so `training/exact_value_function_training/` is empty. The computation itself is fast because it is parallelized: getting the value tables for all families of instances combined takes around 2.5 hours. To generate the value tables, run:

```bash
python training/train_exact_value_function.py
```

You need these tables to run `exact_value_function_policy.py`, and therefore to run `export_results.py`, which evaluates that policy along with the others.

> **Runtime warning.** Running `export_results.py` as presented takes about a day and a half as it evaluates every policy, with 200 simulation episodes per instance family, across all 32 families (`num_warehouses` in `[2, 3, 4, 5]` × `num_customers` in `[50, 100, 200, 400]` × `capacity_distribution` in `['uniform', 'uneven']`). Full training of every learned policy (see `training/` below) is also slow. For quicker experiments:
> - In `export_results.py`, lower `num_instances` (episodes per family, currently `200`) and/or trim `num_warehouses_options` / `num_customers_options` / `capacity_distribution_options` (currently 4 × 4 × 2 families) near the bottom of the file.
> - If you're (re)training a policy, cut its training budget (generations/epochs/timesteps - see the relevant `train_*` script/notebook) so you get a working artifact fast rather than a paper-quality one.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`milp_lp_solver.py` uses Gurobi to solve MILP and LP optimization models, which requires a Gurobi license - free for academic use at https://www.gurobi.com/academia/academic-program-and-licenses/. The license is only needed for the policies that call the solver (PHS, PEL, DEL, PLA, LPH); every other policy runs without it.

## Quick start

**Sanity-check the environment.** `env.py` has a smoke test at the bottom that runs one random episode and checks the paper's invariants (every customer served, all capacity used):

```bash
python env.py
# OK: served 50 orders, cumulative reward -3515.97
```

**Build the tables and figures.** Results and trained policies are already included (see [What's included](#whats-included)), so the notebooks below run directly on the stored results. Only `export_results.py` needs the instances and the exact value tables from the optimal policy to be generated first.

The results pipeline runs in this order:

1. `export_results.py` - runs every policy on the held-out test instances (`instances/instances_test/`) and writes the raw per-episode results to `results/tables/` and `results/full_tables/`. This is the slow step. These results are already included, so you can skip this step unless you want to re-run the evaluation.
2. `export_tables.ipynb` - reads `results/full_tables/` and produces the LaTeX tables used in the paper (policy comparison, LVFA/PLA parameter tables).
3. `export_plots_results.ipynb` - turns the same result tables into the paper's results figures (cost/runtime scatter, proximity-rank bar charts), saved under `results/plots/`.
4. `export_plots_training.ipynb` - turns the raw `training/*_training/` logs into the paper's training-curve figures (GP/DQN/LVFA/PPO).

Steps 2–4 are independent of each other: each notebook reads only the stored outputs of step 1 (`results/full_tables/`) and/or the training artifacts (`training/*_training/`), so they can be run in any order.

## Repository layout (details)

Core files at the repository root, shared by the policies, training scripts and notebooks:

- `env.py` - the `InventoryEnv` gym environment (facility locations/capacities, customer arrivals, reward).
- `miscellaneous.py` - small shared helpers (`distance_calculator`, `read_instance`).
- `milp_lp_solver.py` - Gurobi LP and MILP solver, used in all policies requiring solving the offline facility assignment problem with some assumed demand (PHS, DEL, PEL, PLA, LPH).
- `instances_generator.py` - generates the customer-arrival instances (`instances_seed_*.json`) used across every experiment.
- `visualize.py` - plotting/animation helpers used to produce the illustrative figures in `visualization_examples/`.
- `export_results.py`, `export_tables.ipynb`, `export_plots_results.ipynb`, `export_plots_training.ipynb` - the results pipeline described in Quick start above.
- `requirements.txt` - dependencies for `pip install -r requirements.txt`.

### `instances/`

Customer-arrival instances, not included in the repository - run `python instances_generator.py` to create them. They are divided into three subfolders:

- `instances_test/` - held-out instances used for evaluation (`export_results.py`).
- `instances_train_imitation_learning/` - instances used to train the imitation learning (IL) policy.
- `instances_train_parameterized_lookahead_approximation/` - instances used to tune the parameterized lookahead approximation (PLA) policy.

### `policies/`

One file per policy, all implementing the common `BasePolicy` interface (`base_policy.py`: `reset(instance)` once per episode, `act(state)` once per customer). The implemented policies are the following: `myopic_policy.py`, `random_policy.py`, `genetic_programming_policy.py`, `linear_value_function_approximation_policy.py`, `deep_q_networks_policy.py`, `proximal_policy_optimization_policy.py`, `imitation_learning_policy.py`, `point_estimate_lookahead_policy.py`, `distributional_estimate_lookahead_policy.py`, `parameterized_lookahead_approximation_policy.py`, `linear_programming_heuristic.py`, `perfect_hindsight_policy.py`, `exact_value_function_policy.py`.

`auxiliaries/` holds code shared by more than one policy file: `direct_lookahead_functions.py` (region-generation helpers used by the lookahead/LP-based policies) and `imitation_learning_model.py` (the structure of the neural network used by the imitation learning policy).

### `results/`

Output of the evaluation pipeline (`export_results.py` → `export_tables.ipynb` / `export_plots_results.ipynb` / `export_plots_training.ipynb`):

- `tables/` - per-configuration summary CSVs (`table_w_<warehouses>_c_<customers>_d_<distribution>.csv`).
- `full_tables/` - per-configuration CSVs (`full_results_w_<warehouses>_c_<customers>_d_<distribution>.csv`) with the raw per-episode rewards/times/proximity ranks that the tables and plots are built from.
- `plots/` - rendered figures (PNG/PDF).

### `training/`

Training scripts and notebooks for each policy that needs offline training, and the folder where each stores its raw training artifacts:

- `train_genetic_programming.py` → `genetic_programming_training/`
- `train_linear_value_function_approximation.py` → `linear_value_function_approximation_training/`
- `train_imitation_learning.ipynb` → `imitation_learning_training/`
- `train_parameterized_lookahead_approximation.ipynb` → `parameterized_lookahead_approximation_training/`
- `train_deep_q_networks.ipynb` → `deep_q_networks_training/`
- `train_proximal_policy_optimization.ipynb` → `proximal_policy_optimization_training/`
- `train_exact_value_function.py` → `exact_value_function_training/` - the value tables are not included in the repository (they add up to over 15 GB); run `train_exact_value_function.py` to generate them.

The trained artifacts of every other policy are included, so their scripts/notebooks are only needed if you want to retrain a policy from scratch.

### `visualization_examples/`

Standalone illustrative figures used in the paper, produced by the functions in `visualize.py` (`warehouse_assignments.png`, `warehouse_simulation.gif`, `facilities.pdf`, `myopic_vs_lookahead.pdf`).
