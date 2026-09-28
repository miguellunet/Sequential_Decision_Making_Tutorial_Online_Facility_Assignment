import pandas as pd
from pathlib import Path

from policies.auxiliaries.direct_lookahead_functions import generate_regions_instance
from milp_lp_solver import solve_milp_lp
from policies.base_policy import BasePolicy

TRAIN_DIR = Path(__file__).resolve().parent.parent / "training"

class ParameterizedLookaheadApproximationPolicy(BasePolicy):

    def __init__(self, env, num_warehouses, num_customers, capacity_distribution, num_regions=100, param=None):

        super().__init__(env)
        self.num_regions = num_regions

        # Load the best parameter for this family of instances from the training data, unless a specific parameter is provided
        if param is not None:
            self.param = param
        else:
            cfa_params_df = pd.read_csv(TRAIN_DIR / "parameterized_lookahead_approximation_training" / "parameterized_lookahead_approximation_best_params.csv")
            row = cfa_params_df[
                (cfa_params_df['num_warehouses'] == num_warehouses)
                & (cfa_params_df['num_customers'] == num_customers)
                & (cfa_params_df['capacity_distribution'] == capacity_distribution)
            ]
            self.param = row['best_param'].values[-1]

        # Warm-up call before timing starts
        warmup_state = {
            'customers_left': self.env.num_customers,
            'new_customer': [0, 0.0, 0.0, 1],
            'warehouses_capacity': self.env.warehouses_initial_capacity.copy(),
            'static_info': {'warehouses_location': self.env.warehouses_location},
        }
        self.act(warmup_state)

    def act(self, state):

        # Generates a lookahead instance with the new customer and the expected remaining customers
        current_instance = generate_regions_instance(state['customers_left'], self.num_regions, self.env.grid_size, state['new_customer'][1:3])

        # Scale the warehouses' capacities in the lookahead instance by the learned parameter
        warehouses_capacity = state['warehouses_capacity'].copy()
        for i in range(len(warehouses_capacity)):
            warehouses_capacity[i] = warehouses_capacity[i] * self.param

        # Call the optimization solver to get the best assignment for the new customer in the lookahead instance
        assignments, _, _ = solve_milp_lp(
            current_instance, state['static_info']['warehouses_location'], warehouses_capacity,
            self.num_regions + 1, divisible=True, divisible_except_first_one=True
        ) #the +1 is because the new customer is added to the instance, so we have num_regions + 1 customers in total
        return assignments[0]