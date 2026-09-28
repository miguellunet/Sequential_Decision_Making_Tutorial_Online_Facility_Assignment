import numpy as np
from gymnasium import Env, spaces
from miscellaneous import distance_calculator

class InventoryEnv(Env):

    def __init__ (self, num_warehouses, num_customers, capacity_distribution, grid_size = 200):
        
        self.num_warehouses = num_warehouses
        self.num_customers = num_customers
        self.grid_size = grid_size
        self.capacity_distribution = capacity_distribution
        self.booked_customers = 0
        self.instance = None

        # Get the initial capacity and location of each warehouse

        if num_warehouses == 2:
            self.warehouses_location = [[-50, -50], [50, 50]]
            if capacity_distribution == 'uniform':
                self.warehouses_initial_capacity = [int(0.5*num_customers), int(0.5*num_customers)]
                leftovers = num_customers - sum(self.warehouses_initial_capacity)
                for i in range(leftovers):
                    self.warehouses_initial_capacity[i] += 1
            elif capacity_distribution == 'uneven':
                self.warehouses_initial_capacity = [int(0.7*num_customers), int(0.3*num_customers)]
                leftovers = num_customers - sum(self.warehouses_initial_capacity)
                for i in range(leftovers):
                    self.warehouses_initial_capacity[i] += 1
        elif num_warehouses == 3:
            self.warehouses_location = [[-50, -50], [0, 50], [50, -50]]
            if capacity_distribution == 'uniform':
                self.warehouses_initial_capacity = [int(num_customers/3), int(num_customers/3), int(num_customers/3)]
                leftovers = num_customers - sum(self.warehouses_initial_capacity)
                for i in range(leftovers):
                    self.warehouses_initial_capacity[i] += 1
            elif capacity_distribution == 'uneven':
                self.warehouses_initial_capacity = [int(0.5*num_customers), int(0.3*num_customers), int(0.2*num_customers)]
                leftovers = num_customers - sum(self.warehouses_initial_capacity)
                for i in range(leftovers):
                    self.warehouses_initial_capacity[i] += 1
        elif num_warehouses == 4:
            self.warehouses_location = [[-50, -50], [-50, 50], [50, 50], [50, -50]]
            if capacity_distribution == 'uniform':
                self.warehouses_initial_capacity = [int(0.25*num_customers) for i in range(num_warehouses)]
                leftovers = num_customers - sum(self.warehouses_initial_capacity)
                for i in range(leftovers):
                    self.warehouses_initial_capacity[i] += 1
            elif capacity_distribution == 'uneven':
                self.warehouses_initial_capacity = [int(0.4*num_customers), int(0.3*num_customers), int(0.2*num_customers), int(0.1*num_customers)]
                leftovers = num_customers - sum(self.warehouses_initial_capacity)
                for i in range(leftovers):
                    self.warehouses_initial_capacity[i] += 1
        elif num_warehouses == 5:
            self.warehouses_location = [[-50, -50], [-50, 50], [50, 50], [50, -50], [0, 0]]
            if capacity_distribution == 'uniform':
                self.warehouses_initial_capacity = [int(0.20*num_customers) for i in range(num_warehouses)]
                leftovers = num_customers - sum(self.warehouses_initial_capacity)
                for i in range(leftovers):
                    self.warehouses_initial_capacity[i] += 1
            elif capacity_distribution == 'uneven':
                self.warehouses_initial_capacity = [int(0.3*num_customers), int(0.25*num_customers), int(0.20*num_customers), int(0.15*num_customers), int(0.10*num_customers)]
                leftovers = num_customers - sum(self.warehouses_initial_capacity)
                for i in range(leftovers):
                    self.warehouses_initial_capacity[i] += 1
            
        # Current capacity equals initial capacity at the beginning of the episode
        self.warehouses_capacity = self.warehouses_initial_capacity.copy()
        self.warehouses_with_capacity = [i for i in range(self.num_warehouses)]

        # Set seed to 0 for reproducibility, and set the numpy random seed to the same value
        self.seed = 0
        np.random.seed(self.seed)

        # If there is no instance provided, create a new customer. If there is an instance provided, get the next customer from the instance.
        if self.instance is None:
            self.new_customer = self.create_customer()
        else:
            self.new_customer = self.instance[self.booked_customers]

        # Get the initial observation, and set the observation space and action space
        self.obs = self.reset()
        obs_dim = len(self.reset()[0])
        low_dim = np.array([0 for i in range(obs_dim)])
        high_dim = np.array([1 for i in range(obs_dim)])
        self.observation_space = spaces.Box(low = low_dim, high = high_dim, dtype = np.float32)
        self.action_space = spaces.Discrete(self.num_warehouses)

    def set_seed(self, seed):
        self.seed = seed
        np.random.seed(self.seed)
        
    def refresh_seed(self):
        np.random.seed(self.seed)

    def set_instance(self, instance):
        self.instance = instance

    # If no instance is provided, create a new customer. If an instance is provided, get the next customer from the instance.
    def create_customer(self):
        id = self.booked_customers + 1
        x = np.random.uniform(-self.grid_size/2, self.grid_size/2)
        y = np.random.uniform(-self.grid_size/2, self.grid_size/2)
        demand = 1
        return [id, x, y, demand]

    def step(self, action):

        #Calculate the reward
        reward = 0
        done = False
        truncated = False
        info = {}

        if action not in self.warehouses_with_capacity:
            # a policy should never select a facility with 0 capacity
            raise ValueError(f"Action {action} selects a warehouse with 0 capacity remaining (warehouses_with_capacity={self.warehouses_with_capacity})")

        warehouse = action
        # Update the capacity of the selected warehouse, and if it reaches 0, remove it from the list of warehouses with capacity
        self.warehouses_capacity[action] -= 1
        if self.warehouses_capacity[action] <= 0:
            self.warehouses_with_capacity.remove(warehouse)

        # Update history data for analysis and visualization
        self.all_customers.append(self.new_customer)
        self.all_warehouses.append(warehouse)
        self.all_capacities.append(self.warehouses_capacity.copy())
        self.booked_customers += 1

        # Reward is the negative distance between the selected warehouse and the new customer, as we are minimizing the distance
        reward -= distance_calculator(self.warehouses_location[warehouse], self.new_customer[1:3])

        # Episode terminates when all customers have been served, or when all warehouses have 0 capacity
        if len(self.warehouses_with_capacity) == 0:
            done = True
        else:
            if self.instance is None:
                self.new_customer = self.create_customer()
            else:
                self.new_customer = self.instance[self.booked_customers]

        obs = {'static_info': {'warehouses_initial_capacity': self.warehouses_initial_capacity, 'warehouses_location': self.warehouses_location, 'num_warehouses': self.num_warehouses, 'num_customers': self.num_customers},
            'warehouses_capacity': self.warehouses_capacity, 'new_customer': self.new_customer, 'warehouses_distance': [distance_calculator(self.warehouses_location[i], self.new_customer[1:3]) for i in range(self.num_warehouses)], 'booked_customers': self.booked_customers, 'customers_left': self.num_customers - self.booked_customers}

        self.obs = obs
        
        state = self.get_state(self.obs)

        return state, reward, done, truncated, info

    # Given the action, calculate the rank among warehouses with positive capacity only. The closest warehouse has rank 1, the second closest has rank 2, and so on.
    # Useful to later analyze how close the selected warehouse was to the new customer, and to compare the performance of different policies.
    def get_facility_proximity(self, action):
        available_indices = [i for i in range(self.num_warehouses) if self.warehouses_capacity[i] > 0]
        distances = [distance_calculator(self.warehouses_location[i], self.new_customer[1:3]) for i in available_indices]
        sorted_available = np.array(available_indices)[np.argsort(distances)]
        proximity_rank = np.where(sorted_available == action)[0][0] + 1  # +1 to make it 1-indexed instead of 0-indexed
        return proximity_rank
    
    def get_state(self, state):

        # default flat-vector observation (used as-is by DQN and PPO via stable-baselines3).

        data_rows = []

        for i in range(self.num_warehouses):
            data_rows.append(state['warehouses_distance'][i]/212.13)    # normalize distance by the maximum possible distance in the grid
            data_rows.append(state['warehouses_capacity'][i]/state['static_info']['warehouses_initial_capacity'][i]) # normalize capacity by the initial capacity of the warehouse
        
        return np.array(np.nan_to_num(data_rows, nan=0), dtype=np.float32)
            
    def reset(self, seed=0, options=None):

        # Reset the environment to the initial state, and return the initial observation and info dictionary.

        # The initial state is determined by the initial capacity and location of each warehouse.
        self.warehouses_capacity = self.warehouses_initial_capacity.copy()
        self.warehouses_with_capacity = [i for i in range(self.num_warehouses)]

        # If no instance is provided, create a new customer. If an instance is provided, get the next customer from the instance.
        if self.instance is None:
            self.new_customer = self.create_customer()
        else:
            self.new_customer = self.instance[self.booked_customers]

        # Empty the history data for analysis and visualization
        self.all_customers = []
        self.all_warehouses = []
        self.all_capacities = []
        self.booked_customers = 0

        obs = {'static_info': {'warehouses_initial_capacity': self.warehouses_initial_capacity, 'warehouses_location': self.warehouses_location, 'num_warehouses': self.num_warehouses, 'num_customers': self.num_customers},
            'warehouses_capacity': self.warehouses_capacity, 'new_customer': self.new_customer, 'warehouses_distance': [distance_calculator(self.warehouses_location[i], self.new_customer[1:3]) for i in range(self.num_warehouses)], 'booked_customers': self.booked_customers, 'customers_left': self.num_customers - self.booked_customers,
            'warehouses_initial_capacity': self.warehouses_initial_capacity, 'warehouses_location': self.warehouses_location}

        self.obs = obs
        
        state = self.get_state(self.obs)

        return state, {}
    

if __name__ == "__main__":

    # smoke test: run one random episode end-to-end and sanity-check the invariants

    env = InventoryEnv(2, 50, 'uniform', grid_size=100)

    # Use the full state observation (not the flattened one)
    # The flattened observation is used by the RL policies, but for this test we want to see the full state
    def get_state(state):
        return state
    env.get_state = get_state

    state, _ = env.reset()

    done = False
    cum_reward = 0

    while not done:
        available = [i for i in range(state['static_info']['num_warehouses']) if state['warehouses_capacity'][i] > 0]
        action = np.random.choice(available)
        state, reward, done, truncated, info = env.step(action)
        cum_reward += reward

    assert env.booked_customers == env.num_customers, "not all orders were served"
    assert sum(env.warehouses_capacity) == 0, "capacity left unused at episode end"
    print(f"OK: served {env.booked_customers} orders, cumulative reward {cum_reward:.2f}")