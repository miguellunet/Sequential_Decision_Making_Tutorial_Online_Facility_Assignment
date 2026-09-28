from milp_lp_solver import solve_milp_lp
from policies.base_policy import BasePolicy

class PerfectHindsightPolicy(BasePolicy):
  
    def __init__(self, env):
        super().__init__(env)

    def reset(self, instance):
        super().reset(instance)

        # In the first moment of the episode, solve the entire instance with deterministic optimization to get the optimal assignment for all customers
        self.all_actions, _, _ = solve_milp_lp(
            instance, self.env.warehouses_location, self.env.warehouses_initial_capacity, self.env.num_customers
        )
        self.idx = 0

    def act(self, state):
        # Return the optimal assignment for the current customer, as determined by the perfect hindsight solution determined at the start of the episode
        action = self.all_actions[self.idx]
        self.idx += 1
        return action