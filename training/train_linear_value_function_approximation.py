import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from miscellaneous import distance_calculator

TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(TRAIN_DIR))

class LVFAAgent:

    def __init__(self, env, discount_factor=0.9, alpha=0.1, num_iterations=5, num_simulations=10, stop_criterion=1e-1, include_squared_capacity_feature=False):
        self.env = env

        def get_state(state):
            return state
        self.env.get_state = get_state

        self.discount_factor = discount_factor
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.num_simulations = num_simulations
        self.stop_criterion = stop_criterion
        self.include_squared_capacity_feature = include_squared_capacity_feature

        # one coefficient per facility, no intercept, plus one more per facility for the squared-capacity term if enabled
        self.theta = np.zeros(env.num_warehouses * (2 if include_squared_capacity_feature else 1))

    def features_of(self, capacities):
        # Get state features based on the current capacities of the warehouses. If include_squared_capacity_feature is True, also include the squared capacities as features.
        if not self.include_squared_capacity_feature:
            return capacities
        else:
            return np.concatenate([capacities, capacities ** 2])

    def value_of(self, theta, features):
        # Calculate the value function approximation V(s) = theta^T * features
        return float(np.dot(theta, features))

    def simulate_policy(self, policy):

        # Run a episode using the given policy and return the collected data and final cost. The data consists of tuples of (features_at_t, reward).
        state, _ = self.env.reset()
        all_data = []  # Stores (features_at_t, reward)
        final_reward = 0
        final_cost = 0  # reward = -dist, so cost = dist

        done = False
        while not done:
            action = policy(state, self.theta)
            # feature vector for this decision epoch is built from the pre-decision state (before taking the action), which is the current capacities of the warehouses
            capacities = np.array(state['warehouses_capacity'], dtype=float)
            features = self.features_of(capacities)
            next_state, reward, done, truncated, info = self.env.step(action)
            final_reward += reward
            cost = -reward  # reward = -dist, so cost = dist
            final_cost += cost

            all_data.append((features, cost))
            state = next_state

        # Update all_data with the (discounted) reward-to-go
        cost_to_go = 0
        for i in range(len(all_data) - 1, -1, -1):
            cost_to_go = all_data[i][1] + self.discount_factor * cost_to_go
            all_data[i] = (all_data[i][0], cost_to_go)

        return all_data, final_cost

    def fit_v_function(self, all_data):
        # Fit the linear regression model to the collected data and return the new theta coefficients. The features are the state features, and the target is the discounted reward-to-go (cost-to-go).
        X, y = [], []
        for features, cost_to_go in all_data:
            X.append(features)
            y.append(cost_to_go)

        # Intercept is set to False because we want the value function to be zero when all features are zero (i.e., when all capacities are zero).
        model = LinearRegression(fit_intercept=False).fit(X, y)
        return model.coef_

    def update_policy(self):

        # Run the LVFA algorithm for a specified number of iterations, updating the policy based on the fitted value function.
        # The process continues until the stopping criterion is met or the maximum number of iterations is reached.
        
        stop_flag = False

        evolution_data = []
        all_thetas = []

        for q in range(self.num_iterations):
            if stop_flag:
                break
            print(f"Iteration {q+1}")
            all_data = []
            all_final_costs = []

            # Store the current theta for later analysis of the evolution of the value function coefficients
            all_thetas.append(self.theta)

            # Run multiple simulations to collect data
            for _ in range(self.num_simulations):
                data, final_cost = self.simulate_policy(self.current_policy)
                all_final_costs.append(final_cost)
                all_data.extend(data)

            # Store the mean cost obtained while using the current policy for later analysis of the evolution of the mean cost over iterations
            mean_cost = np.mean(all_final_costs)
            evolution_data.append(mean_cost)

            # Get the new theta coefficients by fitting the value function to the collected data
            theta_new = self.fit_v_function(all_data)

            # Stopping criterion comparing each coefficient of theta and theta_new
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_change = np.abs((self.theta - theta_new) / self.theta)
            rel_change = rel_change[np.isfinite(rel_change)]
            if rel_change.size > 0 and np.all(rel_change < self.stop_criterion):
                stop_flag = True

            # Update theta using a weighted average of the old and new coefficients, controlled by the learning rate alpha.
            if q < self.num_iterations - 1:
                self.theta = (1 - self.alpha) * self.theta + self.alpha * theta_new

        if q == self.num_iterations - 1 or stop_flag:

            # Store the evolution of the mean cost and theta coefficients over iterations to a CSV file for later analysis.
            # The CSV file will contain the iteration number, mean cost, and the theta coefficients for each iteration.
     
            if stop_flag:
                data = {
                    'iteration': list(range(1, q + 1)),
                    'mean_cost': evolution_data,
                    'theta': [','.join(map(str, np.round(theta, 4))) for theta in all_thetas]
                }
            else:
                data = {
                    'iteration': list(range(1, q + 2)),
                    'mean_cost': evolution_data,
                    'theta': [','.join(map(str, np.round(theta, 4))) for theta in all_thetas]
                }

            file_name = f'{TRAIN_DIR}/linear_value_function_approximation_training/lvfa_improved_evolution_w_{self.env.num_warehouses}_c_{self.env.num_customers}_d_{self.env.capacity_distribution}.csv'
            df = pd.DataFrame(data)
            df.to_csv(file_name, mode='w', header=True, index=False)

    def current_policy(self, state, theta):

        # The sampling policy selects the action that minimizes the expected cost-to-go based on the current state and the fitted value function. 

        capacities = np.array(state['warehouses_capacity'], dtype=float)

        costs = []
        distances = []
        for i in range(len(capacities)):
            if capacities[i] == 0:
                costs.append(float('inf'))
                distances.append(float('inf'))
                continue
            distance = distance_calculator(state['static_info']['warehouses_location'][i], state['new_customer'][1:3])
            distances.append(distance)
            # Get the post-decision state by decrementing the capacity of the chosen warehouse
            next_capacities = capacities.copy()
            next_capacities[i] -= 1
            next_features = self.features_of(next_capacities)
            costs.append(distance + self.discount_factor * self.value_of(theta, next_features))

        best_cost = min(costs)
        best_actions = [i for i, c in enumerate(costs) if c == best_cost]
        if len(best_actions) == 1:
            return best_actions[0]
        else:
            return min(best_actions, key=lambda i: distances[i])

LVFA_SEED = 42

def store_theta(num_warehouses, num_customers, capacity_distribution, include_squared_capacity_feature=False):
    from env import InventoryEnv
    env = InventoryEnv(num_warehouses=num_warehouses, num_customers=num_customers, capacity_distribution=capacity_distribution)

    # Set the random seed for reproducibility
    np.random.seed(LVFA_SEED)

    agent = LVFAAgent(env, discount_factor=0.99, alpha=1/100, num_iterations=500, num_simulations=100, stop_criterion=0, include_squared_capacity_feature=include_squared_capacity_feature)
    agent.update_policy()

    data = {
        'num_warehouses': [num_warehouses],
        'num_customers': [num_customers],
        'capacity_distribution': [capacity_distribution],
        'theta': [','.join(map(str, np.round(agent.theta, 4)))]
    }

    df = pd.DataFrame(data)
    theta_path = f'{TRAIN_DIR}/linear_value_function_approximation_training/theta.csv'
    df.to_csv(theta_path, mode='a', header=not os.path.exists(theta_path), index=False)


if __name__ == "__main__":

    num_warehouses_options = [2, 3, 4, 5]
    num_customers_options = [50, 100, 200, 400]
    capacity_distribution_options = ['uniform', 'uneven']

    training_times = []  # one row per (num_warehouses, num_customers, capacity_distribution) family

    for num_customers in num_customers_options:
        for num_warehouses in num_warehouses_options:
            for capacity_distribution in capacity_distribution_options:
                print(f'Running for {num_warehouses} warehouses, {num_customers} customers, {capacity_distribution} capacity distribution')

                start_time = time.perf_counter()
                store_theta(num_warehouses, num_customers, capacity_distribution, False)
                training_time_minutes = (time.perf_counter() - start_time) / 60

                training_times.append({
                    'num_warehouses': num_warehouses,
                    'num_customers': num_customers,
                    'capacity_distribution': capacity_distribution,
                    'training_time': round(training_time_minutes, 2),
                })

    info_dir = os.path.join(TRAIN_DIR, 'linear_value_function_approximation_training')
    os.makedirs(info_dir, exist_ok=True)

    training_times_df = pd.DataFrame(training_times)
    training_times_df.to_csv(os.path.join(info_dir, 'linear_value_function_approximation_training_times.csv'), index=False, float_format='%.5f')