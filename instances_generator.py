import numpy as np
import json
import os

def create_instances(seed, subfolder=''):

    # Generates one long pool of 1000 orders per seed
    # (higher than the bigger instance we will test, which is 400 orders)

    num_customers = 1000

    np.random.seed(seed)

    instances = []
    for i in range(num_customers):
        customer = []
        customer.append(i+1)                            # customer ID
        customer.append(np.random.uniform(-100, 100))   # x-coordinate
        customer.append(np.random.uniform(-100, 100))   # y-coordinate
        customer.append(1)                              # unitary demand
        instances.append(customer)

    # Store in json format
    instances_json = {}
    instances_json['instances'] = instances

    #Store in json file
    os.makedirs(f'instances/{subfolder}', exist_ok=True)
    with open(f'instances/{subfolder}instances_seed_{seed}.json', 'w') as f:
        json.dump(instances_json, f)

if __name__ == "__main__":

    for seed in range(10001, 12001):
        create_instances(seed, subfolder='instances_test/')

    for seed in range(20001, 25001):
        create_instances(seed, subfolder='instances_train_imitation_learning/')

    for seed in range(30001, 30501):
            create_instances(seed, subfolder='instances_train_parameterized_lookahead_approximation/')