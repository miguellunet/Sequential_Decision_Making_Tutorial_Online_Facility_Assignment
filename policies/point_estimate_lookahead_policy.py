
import numpy as np

from miscellaneous import distance_calculator
from milp_lp_solver import solve_milp_lp
from policies.base_policy import BasePolicy


class PointEstimateLookaheadPolicy(BasePolicy):

    def __init__(self, env, dla_param=1, num_samples=10, seed=42):

        super().__init__(env)
        self.dla_param = dla_param
        self.num_samples = num_samples

        # Set seed for Monte Carlo sampling to ensure reproducibility
        self.rng = np.random.default_rng(seed)

        # Warm-up call before timing starts
        warmup_state = {'customers_left': self.env.num_customers, 'new_customer': [0, 0.0, 0.0, 1]}
        self.act(warmup_state)

    def act(self, state):

        new_customer = state['new_customer']
        customers_left = state['customers_left']    # the number of customers left to be assigned, including the new customer
        updated_capacity = [int(cap * self.dla_param) for cap in self.env.warehouses_capacity]      #dla_param is set to 1 but can be changed to scale the capacity of the warehouses in the sampled instance (option to make it a hybrid-class policy: DLA-CFA)

        votes = []
        for _ in range(self.num_samples):
            # customer 0 is the real, already-known customer; the rest are a random sample of where the remaining (customers_left - 1) orders will land
            instance = [[0, new_customer[1], new_customer[2], 1]]
            for i in range(customers_left - 1):
                x = self.rng.uniform(-self.env.grid_size / 2, self.env.grid_size / 2)
                y = self.rng.uniform(-self.env.grid_size / 2, self.env.grid_size / 2)
                instance.append([i + 1, x, y, 1])

            # Solve the sampled instance with deterministic optimization to get the best assignment for the new customer
            assignments, _, _ = solve_milp_lp(
                instance, self.env.warehouses_location, updated_capacity, len(instance)
            )

            # Store the assignment for the new customer (customer 0) in the votes list
            votes.append(assignments[0])

        # Determine the warehouse with the most votes and return it as the action. In case of a tie, choose the closest warehouse to the new customer.
        max_count = max(votes.count(f) for f in set(votes))
        tied_facilities = [f for f in set(votes) if votes.count(f) == max_count]
        return min(tied_facilities, key=lambda f: distance_calculator(self.env.warehouses_location[f], new_customer[1:3]))