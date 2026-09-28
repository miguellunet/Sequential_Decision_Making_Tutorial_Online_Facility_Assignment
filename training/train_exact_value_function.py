import os
import sys
import csv
import itertools
import time
import multiprocessing as mp
from multiprocessing import shared_memory
from collections import defaultdict

import numpy as np
import pandas as pd

TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(TRAIN_DIR))

from env import InventoryEnv
from miscellaneous import distance_calculator


# In the backward dynamic process algorithm we develop, we can parallelize the computation of the value function for different states.
# However, if the number of states is too small, the overhead of managing multiple processes can outweigh the benefits of parallelization.
# Therefore, we set a threshold for the minimum number of states required to justify parallel processing. If the number of states is below this threshold, we will compute the value function sequentially instead of in parallel.
MIN_STATES_FOR_PARALLEL = 200


def _dp_worker_init(shm_name, shape, distances, gamma, all_possible_actions):

    # Initializer for the worker process in the multiprocessing pool.
    # runs once per worker process
    # attaches to the parent's shared-memory value table

    # These global variables are used to share data between the main process and the worker processes.
    global _shm, _V, _distances, _gamma, _all_possible_actions
    _shm = shared_memory.SharedMemory(name=shm_name)
    _V = np.ndarray(shape, dtype=np.float64, buffer=_shm.buf)
    _distances = distances
    _gamma = gamma
    _all_possible_actions = all_possible_actions


# Gets a chunk of states and computes the value function for each state in the chunk.
def _dp_process_chunk(chunk):
    # writes each state's value directly into the shared V array
    V = _V
    distances = _distances
    gamma = _gamma
    for state in chunk:
        possible_actions = [a for a in _all_possible_actions if state[a] > 0]
        next_values = np.empty(len(possible_actions))
        for i, action in enumerate(possible_actions):
            next_state = list(state)
            next_state[action] -= 1
            next_values[i] = V[tuple(next_state)]
        scores = -distances[:, possible_actions] + gamma * next_values
        # doing the mean over the rows of the scores matrix is equivalent to taking the expected value of the maximum score over all possible customer locations, assuming a uniform distribution over the customer locations.
        V[state] = scores.max(axis=1).mean()


# Divides the list of states into n_chunks approximately equal-sized chunks
# These chunks are then processed in parallel by the worker processes in the multiprocessing pool.
def _chunkify(states, n_chunks):
    n_chunks = max(1, n_chunks)
    chunk_size = max(1, (len(states) + n_chunks - 1) // n_chunks)
    return [states[i:i + chunk_size] for i in range(0, len(states), chunk_size)]


def aggregate_state(capacity, aggregation):
    # Aggregates the state by rounding up each warehouse's capacity to the nearest multiple of the aggregation factor.
    # This reduces the number of unique states, which can help speed up computations in dynamic programming.
    # In our case, we do not use aggregation, as we want to compute the exact value function for every possible state.
    if aggregation <= 1:
        return tuple(capacity)
    return tuple(int(np.ceil(c / aggregation)) * aggregation for c in capacity)


def backward_dynamic_programming(env, gamma=0.99, num_squares=16, aggregation=1, n_workers=None):

    # The parallelization of the backward dynamic programming algorithm is only beneficial when the number of states is large enough to justify the overhead of managing multiple processes.
    # If the number of states is small, the overhead can outweigh the benefits of parallelization.
    if aggregation <= 1:
        return _dp_parallel(env, gamma=gamma, num_squares=num_squares, n_workers=n_workers)
    return _dp_sequential(env, gamma=gamma, num_squares=num_squares, aggregation=aggregation)


def _dp_parallel(env, gamma=0.99, num_squares=16, n_workers=None):
    # Get the number of worker processes to use for parallel processing. 
    n_workers = n_workers or os.cpu_count() or 1
    print(f"Running dynamic backward induction with {n_workers} worker processes")

    # Get the list of all possible actions (i.e., the indices of the warehouses).
    all_possible_actions = np.arange(0, env.num_warehouses, 1)

    grid_side = round(num_squares ** 0.5)
    if grid_side * grid_side != num_squares:
        raise ValueError(f"num_squares={num_squares} must be a perfect square")

    cell_width = env.grid_size / grid_side
    half_grid = env.grid_size / 2
    centers = [-half_grid + cell_width * (i + 0.5) for i in range(grid_side)]
    possible_locations = [(x, y) for x in centers for y in centers]

    # all distances from each possible customer location to each warehouse location, precomputed for efficiency
    distances = np.array([
        [distance_calculator(loc, facility_loc) for facility_loc in env.warehouses_location]
        for loc in possible_locations
    ])

    caps = env.warehouses_capacity
    shape = tuple(c + 1 for c in caps)

    states_by_sum = defaultdict(list)
    for state in itertools.product(*(range(0, c + 1) for c in caps)):
        states_by_sum[sum(state)].append(state)
    total_capacity = sum(caps)

    shm = shared_memory.SharedMemory(create=True, size=int(np.prod(shape)) * 8)
    try:
        V = np.ndarray(shape, dtype=np.float64, buffer=shm.buf)
        V[:] = 0.0

        ctx = mp.get_context('fork')
        with ctx.Pool(n_workers, initializer=_dp_worker_init,
                      initargs=(shm.name, shape, distances, gamma, all_possible_actions)) as pool:
            for customers_left in range(1, total_capacity + 1):
                print(f"Dynamic programming iteration for {customers_left} customers left ({total_capacity - customers_left} done)")

                # Get the list of all post-decision states with the current number of customers left.
                states_list = states_by_sum[customers_left]
                if not states_list:
                    continue

                # If the number of states is below the threshold for parallel processing, compute the value function sequentially.
                # Otherwise, divide the states into chunks and process them in parallel using the worker processes in the multiprocessing pool.
                if len(states_list) < MIN_STATES_FOR_PARALLEL:
                    for state in states_list:
                        possible_actions = [a for a in all_possible_actions if state[a] > 0]
                        next_values = np.empty(len(possible_actions))
                        for i, action in enumerate(possible_actions):
                            next_state = list(state)
                            next_state[action] -= 1
                            next_values[i] = V[tuple(next_state)]
                        scores = -distances[:, possible_actions] + gamma * next_values
                        # doing the mean over the rows of the scores matrix is equivalent to taking the expected value of the maximum score over all possible customer locations, assuming a uniform distribution over the customer locations.
                        V[state] = scores.max(axis=1).mean()
                else:
                    # Divide the states into chunks and process them in parallel using the worker processes in the multiprocessing pool.
                    pool.map(_dp_process_chunk, _chunkify(states_list, n_workers))

        values = {state: float(V[state]) for state in itertools.product(*(range(0, c + 1) for c in caps))}
    finally:
        shm.close()
        shm.unlink()

    return values


def _dp_sequential(env, gamma=0.99, num_squares=16, aggregation=1):

    # Get the list of all possible actions (i.e., the indices of the warehouses).
    all_possible_actions = np.arange(0, env.num_warehouses, 1)
 
    grid_side = round(num_squares ** 0.5)
    if grid_side * grid_side != num_squares:
        raise ValueError(f"num_squares={num_squares} must be a perfect square") # num_squares must be a perfect square

    cell_width = env.grid_size / grid_side
    half_grid = env.grid_size / 2
    centers = [-half_grid + cell_width * (i + 0.5) for i in range(grid_side)]

    possible_locations = [(x, y) for x in centers for y in centers]

    # all distances from each possible customer location to each warehouse location, precomputed for efficiency
    distances = np.array([
        [distance_calculator(loc, facility_loc) for facility_loc in env.warehouses_location]
        for loc in possible_locations
    ])

    states_by_sum = defaultdict(list)
    for state in itertools.product(*(range(0, capacity + 1) for capacity in env.warehouses_capacity)):
        states_by_sum[sum(state)].append(state)

    total_capacity = sum(env.warehouses_capacity)

    # Initialize the value function with the base case: when there are no customers left, the value is 0 for all states.
    values = {aggregate_state((0,) * env.num_warehouses, aggregation): 0.0}

    # Iterate over the number of customers left, from 1 to the total capacity of all warehouses.
    for customers_left in range(1, total_capacity + 1):
        print(f"Dynamic programming iteration for {customers_left} customers left ({total_capacity - customers_left} done)")

        generation_values = defaultdict(list)

        #Compute the value of each post-decision state.
        for state in states_by_sum[customers_left]:

            possible_actions = [action for action in all_possible_actions if state[action] > 0]

            next_values = np.empty(len(possible_actions))
            for i, action in enumerate(possible_actions):
                next_state = list(state)
                next_state[action] -= 1
                next_key = aggregate_state(tuple(next_state), aggregation)
                next_values[i] = values.get(next_key, 0.0)

            scores = -distances[:, possible_actions] + gamma * next_values
            # the mean over the rows of the scores matrix is equivalent to taking the expected value of the maximum score over all possible customer locations, assuming a uniform distribution over the customer locations.
            state_value = scores.max(axis=1).mean()

            generation_values[aggregate_state(state, aggregation)].append(state_value)

        for key, vals in generation_values.items():
            values[key] = float(np.mean(vals))

    return values

# Store the computed value function in a CSV file for later use by the ExactValueFunctionPolicy
def store_dp_values(values, num_warehouses, num_customers, capacity_distribution, gamma, num_squares=16, aggregation=1):
    info_dir = os.path.join(TRAIN_DIR, 'exact_value_function_training')
    os.makedirs(info_dir, exist_ok=True)
    path = os.path.join(info_dir, f'values_w_{num_warehouses}_c_{num_customers}_d_{capacity_distribution}_gamma_{gamma}_squares_{num_squares}_agg_{aggregation}.csv')
    with open(path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['capacity', 'value'])
        for capacity, value in values.items():
            writer.writerow([','.join(map(str, capacity)), value])

# Read the computed value function from a CSV file for later use by the ExactValueFunctionPolicy
def read_dp_values(num_warehouses, num_customers, capacity_distribution, gamma, num_squares=16, aggregation=1):
    path = os.path.join(TRAIN_DIR, 'exact_value_function_training', f'values_w_{num_warehouses}_c_{num_customers}_d_{capacity_distribution}_gamma_{gamma}_squares_{num_squares}_agg_{aggregation}.csv')
    values = {}
    with open(path, 'r') as fh:
        reader = csv.reader(fh)
        next(reader)
        for row in reader:
            capacity = tuple(int(x) for x in row[0].split(','))
            values[capacity] = float(row[1])
    return values


if __name__ == "__main__":

    # The commented-out block below was used to test the effect of the number of squares on the value function for a single configuration of parameters.
    # The respective analysis is conducted in Figure B.12 of the paper
    '''
    num_warehouses_options = [2, 3, 4, 5]
    num_customers_options = [50]
    capacity_distribution_options = ['uniform', 'uneven']

    gamma = 1   #exact value function is computed for gamma=1, as the objective is to maximize the total (undiscounted) reward over the entire horizon
    aggregation = 1 # no aggregation is used for the exact value function, as we want to compute the exact value function for every possible state

    NUM_REGIONS_TEST = [25, 100, 400, 1600, 2500, 5625, 10000, 22500, 40000]

    #Store the value of the first state of the value function for each configuration of the parameters
    df = pd.DataFrame(columns=['num_warehouses', 'num_customers', 'capacity_distribution', 'num_squares', 'initial_value'])
    for num_customers in num_customers_options:
        for num_warehouses in num_warehouses_options:
            for capacity_distribution in capacity_distribution_options:
                for num_squares in NUM_REGIONS_TEST:
                    print(f'Running for {num_warehouses} warehouses, {num_customers} customers, {capacity_distribution} capacity distribution, {num_squares} squares')
                    env = InventoryEnv(num_warehouses=num_warehouses, num_customers=num_customers, capacity_distribution=capacity_distribution)


                    values = _dp_sequential(env, gamma=gamma, num_squares=num_squares, aggregation=aggregation)
                    store_dp_values(values, num_warehouses, num_customers, capacity_distribution, gamma, num_squares, aggregation)
                    initial_value = values.get(aggregate_state(tuple(env.warehouses_initial_capacity), aggregation), 0.0)
                    df = pd.concat([df, pd.DataFrame.from_records([{
                        'num_warehouses': num_warehouses,
                        'num_customers': num_customers,
                        'capacity_distribution': capacity_distribution,
                        'num_squares': num_squares,
                        'initial_value': initial_value
                    }])], ignore_index=True)
    #Store the value of the first state of the value function for each configuration of the parameters
    df.to_csv(os.path.join(TRAIN_DIR, 'exact_value_function_training', 'square_test.csv'), index=False, float_format='%.5f')
    '''
                    
    
    num_customers_options = [50, 100, 200, 400]
    num_warehouses_options = [2, 3, 4, 5]
    capacity_distribution_options = ['uniform', 'uneven']
    
    training_times = []  # one row per (num_warehouses, num_customers, capacity_distribution) family

    gamma = 1   #exact value function is computed for gamma=1, as the objective is to maximize the total (undiscounted) reward over the entire horizon
    num_squares = 10_000
    aggregation = 1 # no aggregation is used for the exact value function, as we want to compute the exact value function for every possible state

    for num_customers in num_customers_options:
        for num_warehouses in num_warehouses_options:
            for capacity_distribution in capacity_distribution_options:
                if num_warehouses == 5 and num_customers == 400:
                    continue  # skip this combination because the number of states is too large for the memory available on the machine
                print(f'Running for {num_warehouses} warehouses, {num_customers} customers, {capacity_distribution} capacity distribution')
            
                env = InventoryEnv(num_warehouses=num_warehouses, num_customers=num_customers, capacity_distribution=capacity_distribution)
                start_time = time.perf_counter()
                values = backward_dynamic_programming(env, gamma=gamma, num_squares=num_squares, aggregation=aggregation)
                training_time_minutes = (time.perf_counter() - start_time) / 60

                # Store the computed value function in a CSV file for later use by the ExactValueFunctionPolicy
                # Also store the training time and number of states in a CSV file for later analysis of the training time as a function of the number of states
                store_dp_values(values, num_warehouses, num_customers, capacity_distribution, gamma, num_squares, aggregation)
                num_states = 1
                for capacity in env.warehouses_initial_capacity:
                    num_states = num_states * (capacity + 1)
                training_times.append({
                    'num_warehouses': num_warehouses,
                    'num_customers': num_customers,
                    'capacity_distribution': capacity_distribution,
                    'training_time': round(training_time_minutes, 2),
                    'states': num_states
                })

    info_dir = os.path.join(TRAIN_DIR, 'exact_value_function_training')
    os.makedirs(info_dir, exist_ok=True)

    training_times_df = pd.DataFrame(training_times)
    training_times_df.to_csv(os.path.join(info_dir, 'exact_value_function_training_times.csv'), index=False, float_format='%.5f')