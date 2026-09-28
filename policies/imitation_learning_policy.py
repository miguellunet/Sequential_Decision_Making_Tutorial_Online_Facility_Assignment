import numpy as np
import torch
from pathlib import Path

from policies.base_policy import BasePolicy
from policies.auxiliaries.imitation_learning_model import ImitationLearningNet

TRAIN_DIR = Path(__file__).resolve().parent.parent / "training"

class ImitationLearningPolicy(BasePolicy):

    def __init__(self, env, num_warehouses, num_customers, capacity_distribution):

        # Load the trained neural network model for imitation learning
        super().__init__(env)
        self.num_warehouses = num_warehouses
        self.model_nn = ImitationLearningNet(num_warehouses)
        state_dict = torch.load(
            TRAIN_DIR / "imitation_learning_training" / "imitation_learning_models" / f"model_nn_w_{num_warehouses}_c_50_d_{capacity_distribution}.pt",
            map_location="cpu",
        )
        self.model_nn.load_state_dict(state_dict)
        self.model_nn.eval()

        # Warm-up call before timing starts
        warmup_size = 1
        dummy_features = [0.0] * (2 * num_warehouses)
        with torch.no_grad():
            for _ in range(warmup_size):
                self.model_nn(torch.tensor(dummy_features, dtype=torch.float32).unsqueeze(0))

    def act(self, state):

        # Generate the features for the neural network based on the current state
        
        warehouses_distance = state["warehouses_distance"]
        warehouses_capacity = state["warehouses_capacity"]

        features = []
        for i in range(self.num_warehouses):
            features.append(warehouses_distance[i] / 212.13)
            features.append(warehouses_capacity[i] / state['static_info']['warehouses_initial_capacity'][i])

        features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        # Get the logits from the neural network and mask out warehouses with no capacity
        with torch.no_grad():
            logits = self.model_nn(features)[0].numpy()

        for i in range(len(logits)):
            if warehouses_capacity[i] == 0:
                logits[i] = -np.inf

        # Return the index of the available warehouse with the highest logit value
        return int(np.argmax(logits))