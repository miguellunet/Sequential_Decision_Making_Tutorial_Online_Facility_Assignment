# DEAP documentation
#https://deap.readthedocs.io/en/master/api/tools.html

import os
import sys

TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(TRAIN_DIR))

from deap import algorithms, creator, base, tools, gp
import operator

from env import InventoryEnv
import numpy as np
import random
import pandas as pd
import csv
import time

# Run a single episode with a given GP expression
def run_episode(env, policy_function, gp_expression):
    state, info = env.reset()
    done = False  
    final_reward = 0 
    while not done:
        action = policy_function(env, state, gp_expression)
        state, reward, done, truncated, _ = env.step(action)
        final_reward += reward
        
    return final_reward

#Evaluate the genetic programming policy
def evaluate_gp_policy(env, policy_function, gp_expression, n_eval_episodes=3):
    list_final_reward = []
    env.refresh_seed()
    for i in range(n_eval_episodes):
        final_reward = run_episode(env, policy_function, gp_expression)
        list_final_reward.append(final_reward)
    return np.mean(list_final_reward)

# Policy assigned to the GP expression

def my_strategy(env, state, expression=''):
    
    compiled_expression = compile(expression, '<string>', 'eval')

    # Only score warehouses that still have capacity
    scores = {}
    for i in range(env.num_warehouses):
        if state['warehouses_capacity'][i] == 0:
            continue

        DISTANCE = state['warehouses_distance'][i]
        CAPACITY = state['warehouses_capacity'][i]
        INITIAL_CAPACITY = state['static_info']['warehouses_initial_capacity'][i]
       
        scores[i] = eval(compiled_expression)

    # Select the warehouse with the highest score, breaking ties by choosing the closest warehouse
    best_score = max(scores.values())
    tied = [i for i, score in scores.items() if score == best_score]
    action = min(tied, key=lambda i: state['warehouses_distance'][i])

    return action


# Operators for the GP expressions
def add(a, b):  return a + b
def sub(a, b):  return a - b
def mult(a, b): return a * b
def div(a, b):  return b and a / b or 0  #if b is zero, return 0 to avoid division by zero


def train_gp(num_warehouses, num_customers, capacity_distribution):

    population_size           = 100
    num_generations           = 100
    crossover_probability     = 0.80
    mutation_probability      = 0.10
    elitism_rate              = 0.10
    min_depth                 = 0
    max_depth                 = 4
    n_ep                      = 20
    gp_seed                   = 42

    # Seed for reproducibility
    random.seed(gp_seed)
    np.random.seed(gp_seed)

    # Create the environment in which the GP will be trained
    env = InventoryEnv(num_warehouses, num_customers, capacity_distribution)

    def get_state(state): # Redefine the get_state function to get a dictionary.
        return state
    env.get_state = get_state

    def evaluate_expression(individual):
        # Convert the individual to a string representation
        individual_str = str(individual)

        # Check if the individual's fitness is already cached
        # If it is, return the cached fitness value to avoid redundant evaluations
        if individual_str in fitness_cache:
            return fitness_cache[individual_str]

        # If not cached, evaluate the individual
        mean_reward = evaluate_gp_policy(env, my_strategy, str(individual), n_eval_episodes=n_ep)
        fitness_cache[individual_str] = (mean_reward,)  # Cache the fitness value as a tuple

        return (mean_reward,)

    # Operators for the GP expressions
    pset = gp.PrimitiveSet(name = "MAIN", arity = 3)  
    pset.addPrimitive(add, 2)
    pset.addPrimitive(sub, 2)
    pset.addPrimitive(mult, 2)
    pset.addPrimitive(div, 2)
   
    # Terminals are the features of the state: distance, capacity, initial capacity
    pset.renameArguments(ARG0='DISTANCE')
    pset.renameArguments(ARG1='CAPACITY')
    pset.renameArguments(ARG2='INITIAL_CAPACITY')
    
    # Type of fitness (objective: maximize the reward)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    
    # Individuals will be generated with a PrimitiveTree
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    # Cache to store fitness values of evaluated individuals to avoid redundant evaluations
    fitness_cache = {}

    # Configure a toolbox (which is responsible for generating individuals, evaluating them, and applying genetic operators)
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=min_depth, max_=max_depth)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", evaluate_expression)
    toolbox.register("select", tools.selTournament, tournsize=3)  
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))
    toolbox.register("expr_mut", gp.genHalfAndHalf, min_= min_depth , max_= max_depth)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))

    halloffame = tools.HallOfFame(maxsize=int(elitism_rate*population_size))
    pop = toolbox.population(n=population_size)

    # Keep track of the best individuals across generations for elitism
    best_inds = tools.selBest(pop, k=int(elitism_rate*population_size))
    
    filename_all_inds = f'{TRAIN_DIR}/genetic_programming_training/gp_w_{num_warehouses}_c_{num_customers}_d_{capacity_distribution}_all_individuals.csv'

    file_all_inds_exists = os.path.isfile(filename_all_inds)

    
    # Save all individuals and their fitness values to a CSV file
    with open(filename_all_inds, mode='a', newline='') as file:
                                    writer_all_inds = csv.writer(file)
                                    if not file_all_inds_exists:
                                        # Write the column names if the file is new
                                        writer_all_inds.writerow([
                                            'max_depth',
                                            'generation',
                                            'id',
                                            'individual',
                                            'fitness'])
    
    filename_best_inds = f'{TRAIN_DIR}/genetic_programming_training/gp_w_{num_warehouses}_c_{num_customers}_d_{capacity_distribution}_best_individuals.csv'

    file_best_inds_exists = os.path.isfile(filename_best_inds)

    # Save the best individual from each generation to a separate CSV file
    with open(filename_best_inds, mode='a', newline='') as file:
        writer_best_inds = csv.writer(file)
        if not file_best_inds_exists:
            # Write the column names if the file is new
            writer_best_inds.writerow([
                'max_depth',
                'generation',
                'best_individual',
                'max_fitness',
                'avg_fitness'])  
                                    

    # Run the genetic programming algorithm for the specified number of generations
    for gen in range(num_generations):

        # Get the new population and the logbook after applying the genetic programming algorithm for one generation
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=crossover_probability, mutpb=mutation_probability, ngen=1, halloffame=halloffame, verbose=True)
        
        #Apply elitism to the population
        pop = random.sample(pop, len(pop))
        pop = best_inds + pop[int(elitism_rate*population_size):]
        best_inds = tools.selBest(pop, k=int(elitism_rate*population_size))


        # Store relevant information about the current generation
        gen_fitness = []
        for i, ind in enumerate(pop):
            data = [max_depth, gen + 1, i + 1, str(ind), ind.fitness.values[0]]
            
            with open(filename_all_inds, mode='a', newline='') as file:
                                    writer_all_inds = csv.writer(file)
                                    writer_all_inds.writerow(data)
            
            gen_fitness.append(ind.fitness.values[0])

        best_ind = tools.selBest(pop, k=1)[0]

        data = [max_depth, gen + 1, str(best_ind), best_ind.fitness.values[0], np.mean(gen_fitness)]

        # Store the best individual from the current generation
        with open(filename_best_inds, mode='a', newline='') as file:
            writer_best_inds = csv.writer(file)
            writer_best_inds.writerow(data)

        # If this is the last generation, record the best individual in a separate file for all time best individuals
        if gen == num_generations - 1:
            filename_record_best_individual = f'{TRAIN_DIR}/genetic_programming_training/ALL_TIME_best_individual.csv'
            file_record_best_individual_exists = os.path.isfile(filename_record_best_individual)
            with open(filename_record_best_individual, mode='a', newline='') as file:
                writer_best_individual = csv.writer(file)
                if not file_record_best_individual_exists:
                    # Write the column names if the file is new
                    writer_best_individual.writerow([
                        'num_warehouses',
                        'num_customers',
                        'capacity_distribution',
                        'best_ind',
                        'fitness'
                    ])
                writer_best_individual.writerow([num_warehouses, num_customers, capacity_distribution, str(best_ind), best_ind.fitness.values[0]])
        
    # Record statistics for the final population after all generations have been completed
    # May be useful for analysis or debugging purposes
    stats = tools.Statistics(key=operator.attrgetter("fitness.values"))
    stats.register("max", np.max)
    stats.register("mean", np.mean)
    stats.register("min", np.min)
    record = stats.compile(pop)
    logbook = tools.Logbook()
    logbook.record(gen=num_generations, nevals=population_size, fitness=record)

        

if __name__ == "__main__":

    # Define the options for the number of warehouses, number of customers, and capacity distribution
    options_num_warehouses = [2, 3, 4, 5]
    options_num_customers = [50, 100, 200, 400]
    options_capacity_distribution = ['uniform', 'uneven']

    options_num_warehouses = [2]
    options_num_customers = [50]
    options_capacity_distribution = ['uniform']

    training_times = []  # one row per (num_warehouses, num_customers, capacity_distribution) family

    # Store the training time for each (num_warehouses, num_customers, capacity_distribution) family
    for num_warehouses in options_num_warehouses:
        for num_customers in options_num_customers:
            for capacity_distribution in options_capacity_distribution:
                print(f"Training GP for num_warehouses={num_warehouses}, num_customers={num_customers}, capacity_distribution={capacity_distribution}")
                start_time = time.perf_counter()
                train_gp(num_warehouses, num_customers, capacity_distribution)
                training_time_minutes = (time.perf_counter() - start_time) / 60

                training_times.append({
                    'num_warehouses': num_warehouses,
                    'num_customers': num_customers,
                    'capacity_distribution': capacity_distribution,
                    'training_time': round(training_time_minutes, 2),
                })

    info_dir = os.path.join(TRAIN_DIR, 'genetic_programming_training')
    os.makedirs(info_dir, exist_ok=True)

    training_times_df = pd.DataFrame(training_times)
    training_times_df.to_csv(os.path.join(info_dir, 'genetic_programming_training_times.csv'), index=False, float_format='%.5f')