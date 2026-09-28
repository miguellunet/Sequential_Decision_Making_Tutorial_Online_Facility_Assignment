import numpy as np

from miscellaneous import distance_calculator
from policies.auxiliaries.direct_lookahead_functions import generate_regions_instance
from milp_lp_solver import solve_milp_lp
from policies.base_policy import BasePolicy

class LinearProgrammingHeuristicPolicy(BasePolicy):
  
    def __init__(self, env, num_warehouses, num_regions=100):
        super().__init__(env)
        self.num_warehouses = num_warehouses
        self.num_regions = num_regions

        # Warm-up call before timing starts
        warmup_state = {
            'customers_left': self.env.num_customers,
            'new_customer': [0, 0.0, 0.0, 1],
            'warehouses_capacity': self.env.warehouses_initial_capacity.copy(),
            'static_info': {'warehouses_location': self.env.warehouses_location},
        }
        self.act(warmup_state)

    def act(self, state):

        #Get instance with the expected distribution of the remaining customers, and solve it with deterministic optimization to get the dual values for each warehouse
        current_instance = generate_regions_instance(state['customers_left'], self.num_regions, self.env.grid_size, None)
        _, _, dual_values = solve_milp_lp(
            current_instance, state['static_info']['warehouses_location'], state['warehouses_capacity'],
            self.num_regions, divisible=True
        )

        # Calculate the expected serving cost for each warehouse based on the dual values and the distance to the new customer, and select the warehouse with the minimum expected cost
        all_values = []
        for f in range(self.num_warehouses):
            if state['warehouses_capacity'][f] == 0:
                all_values.append(np.inf)
                continue
            all_values.append(distance_calculator(state['new_customer'][1:3], state['static_info']['warehouses_location'][f]) - dual_values[f])

        return int(np.argmin(all_values))
