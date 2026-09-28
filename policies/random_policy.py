import random

from policies.base_policy import BasePolicy

# This policy is not used in the final implementation, but is kept here for reference and testing purposes
# It randomly selects a warehouse to serve a customer from, without considering any optimization criteria

class RandomPolicy(BasePolicy):

    def __init__(self, env):
        super().__init__(env)

        # Warm-up call before timing starts
        self.act(self.env.obs)

    def act(self, state):
        available = [i for i in range(self.env.num_warehouses) if state['warehouses_capacity'][i] > 0]
        return random.choice(available)