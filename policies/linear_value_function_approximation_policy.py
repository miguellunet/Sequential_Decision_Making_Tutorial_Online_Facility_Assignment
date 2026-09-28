from pathlib import Path
import pandas as pd
import numpy as np

from miscellaneous import distance_calculator
from policies.base_policy import BasePolicy

TRAIN_DIR = Path(__file__).resolve().parent.parent / "training"

def read_theta(num_warehouses, num_customers, capacity_distribution):

    # Read the trained theta parameters for the linear value function approximation policy from the training data
    
    df = pd.read_csv(f'{TRAIN_DIR}/linear_value_function_approximation_training/theta.csv')
    df = df[(df['num_warehouses'] == num_warehouses) & (df['num_customers'] == num_customers) & (df['capacity_distribution'] == capacity_distribution)]
    theta = np.array(list(map(float, df['theta'].iloc[-1].split(','))))

    return theta

class LinearValueFunctionApproximationPolicy(BasePolicy):

    def __init__(self, env, num_warehouses, num_customers, capacity_distribution, discount_factor=0.99, include_squared_capacity_feature=False):
        super().__init__(env)
        self.theta = read_theta(num_warehouses, num_customers, capacity_distribution)
        self.discount_factor = discount_factor
        self.include_squared_capacity_feature = include_squared_capacity_feature

        # Warm-up call before timing starts
        warmup_size = 1
        for _ in range(warmup_size):
            self.act(self.env.obs)

    def act(self, state):

        # The features are the warehouses' capacities, optionally including their squares as well. The value function is a linear combination of these features, with weights given by self.theta.
        capacities = np.array(state['warehouses_capacity'], dtype=float)

        # Compute the cost of assigning the new customer to each warehouse, which is the distance to the warehouse plus the discounted value of the post-decision state (after decrementing the capacity of the chosen warehouse).
        # The post-decision state's value is computed as a linear combination of the features, with weights given by self.theta.
        costs = []
        distances = []
        for i in range(len(capacities)):
            if capacities[i] == 0:
                costs.append(float('inf'))
                distances.append(float('inf'))
                continue
            distance = distance_calculator(state['static_info']['warehouses_location'][i], state['new_customer'][1:3])
            distances.append(distance)
            next_capacities = capacities.copy()
            next_capacities[i] -= 1
            if self.include_squared_capacity_feature:
                features = np.concatenate([next_capacities, next_capacities ** 2])
            else:
                features = next_capacities
            next_value = np.dot(self.theta, features)
            costs.append(distance + self.discount_factor * next_value)

        # The warehouse with the lowest cost is chosen as the action.
        best_cost = min(costs)
        best_actions = [i for i, c in enumerate(costs) if c == best_cost]
        if len(best_actions) == 1:
            return best_actions[0]
        else:
            return min(best_actions, key=lambda i: distances[i])