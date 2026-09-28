import pandas as pd
from pathlib import Path

from policies.base_policy import BasePolicy

TRAIN_DIR = Path(__file__).resolve().parent.parent / "training"

# Define the operators
def _add(a, b): return a + b
def _sub(a, b): return a - b
def _mult(a, b): return a * b
def _div(a, b): return b and a / b or 0

class GeneticProgrammingPolicy(BasePolicy):
   
    def __init__(self, env, num_warehouses, num_customers, capacity_distribution):
        super().__init__(env)
        df = pd.read_csv(TRAIN_DIR / "genetic_programming_training" / "ALL_TIME_best_individual.csv")
        df = df[
            (df['num_warehouses'] == num_warehouses)
            & (df['num_customers'] == num_customers)
            & (df['capacity_distribution'] == capacity_distribution)
        ]
        # ALL_TIME_best_individual.csv stores the best scoring function for each family of instances
        # iloc[-1] takes the most recent run for this family, not the first one ever recorded.
        expression = df.iloc[-1]['best_ind']
        self.compiled_expression = compile(expression, '<string>', 'eval')

        # Warm-up call before timing starts
        warmup_size = 1
        for _ in range(warmup_size):
            self.act(self.env.obs)

    def act(self, state):

        add, sub, mult, div = _add, _sub, _mult, _div

        # Attribute scores to the warehouses with available capacity
        scores = {}
        for i in range(len(state['warehouses_capacity'])):
            if state['warehouses_capacity'][i] == 0:
                continue

            DISTANCE = state['warehouses_distance'][i]
            CAPACITY = state['warehouses_capacity'][i]
            INITIAL_CAPACITY = state['static_info']['warehouses_initial_capacity'][i]
            scores[i] = eval(self.compiled_expression)

        # Prescribe the warehouse with the highest score
        best_score = max(scores.values())
        tied = [i for i, score in scores.items() if score == best_score]
        action = min(tied, key=lambda i: state['warehouses_distance'][i])

        return action
