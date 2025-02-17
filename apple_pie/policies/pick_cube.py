import torch
import numpy as np

class PickCubeRandomPolicy:
    def __init__(self):
        self.action_space = 7

    def act(self, x: dict):
        """
        Implements a random policy.
        """
        # This is a stub that returns random actions. Adjust according to your action_space.
        # Example: assume action space is 4-dimensional.
        action = np.random.uniform(-1, 1, size=(self.action_space,))
        return torch.tensor(action, dtype=torch.float32)

class PickCubeRandomPolicySO100:
    def __init__(self):
        self.action_space = 6

    def act(self, x: dict):
        """
        Implements a random policy.
        """
        # This is a stub that returns random actions. Adjust according to your action_space.
        # Example: assume action space is 4-dimensional.
        action = np.random.uniform(-1, 1, size=(self.action_space,))
        return torch.tensor(action, dtype=torch.float32)