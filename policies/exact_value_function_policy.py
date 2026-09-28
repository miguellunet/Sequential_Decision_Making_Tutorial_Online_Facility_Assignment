import numpy as np

from miscellaneous import distance_calculator
from policies.base_policy import BasePolicy
from training.train_exact_value_function import aggregate_state, read_dp_values

class ExactValueFunctionPolicy(BasePolicy):
    
    def __init__(self, env, num_warehouses, num_customers, capacity_distribution, gamma=1, num_squares=10_000, aggregation=1):
        super().__init__(env)
        self.values = read_dp_values(num_warehouses, num_customers, capacity_distribution, gamma, num_squares, aggregation)
        self.gamma = gamma
        self.aggregation = aggregation

        # Warm-up call before timing starts
        self.act(self.env.obs)

    def act(self, state):
        
        capacities = np.array(state['warehouses_capacity'], dtype=int)
        rewards = []
        distances = []
        for i in range(len(capacities)):
            if capacities[i] == 0:
                rewards.append(float('-inf'))
                distances.append(float('inf'))
                continue
            distance = distance_calculator(state['static_info']['warehouses_location'][i], state['new_customer'][1:3])
            distances.append(distance)
            next_capacities = capacities.copy()
            #get the post-decision state by decrementing the capacity of the chosen warehouse
            next_capacities[i] -= 1
            #The value is negative because we are minimizing distance (equivalent to maximizing negative distance)
            next_value = self.values.get(aggregate_state(tuple(next_capacities), self.aggregation), 0.0)
            rewards.append(-distance + self.gamma * next_value)

        best_reward = max(rewards)
        best_actions = [i for i, r in enumerate(rewards) if r == best_reward]
        if len(best_actions) == 1:
            return best_actions[0]
        else:
            return min(best_actions, key=lambda i: distances[i])