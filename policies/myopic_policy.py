import numpy as np

from miscellaneous import distance_calculator
from policies.base_policy import BasePolicy

class MyopicPolicy(BasePolicy):
    
    def __init__(self, env):
        super().__init__(env)

        # Warm-up call before timing starts
        warmup_size = 1
        for _ in range(warmup_size):
            self.act(self.env.obs)

    def act(self, state):

        # Select the warehouse with available capacity that is closest to the new customer
        available = [i for i in range(state['static_info']['num_warehouses']) if state['warehouses_capacity'][i] > 0]
        distance_to_warehouses = {i: distance_calculator(state['new_customer'][1:3], state['static_info']['warehouses_location'][i]) for i in available}
        min_distance = min(distance_to_warehouses.values())
        candidates = [i for i, distance in distance_to_warehouses.items() if distance == min_distance]

        # If there are multiple candidates for the same minimum distance (very unlikely as we take continuous values), choose the one with the most capacity left
        if len(candidates) > 1:
            capacities = [state['warehouses_capacity'][i] for i in candidates]
            action = candidates[np.argmax(capacities)]
        else:
            action = candidates[0]

        return action