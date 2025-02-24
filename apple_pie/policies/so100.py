import torch
import torch.nn as nn
import numpy as np
from skrl.models.torch import Model, MultivariateGaussianMixin, DeterministicMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space

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


class SO100PPOModel(Model):
    """
    A custom PPO model for the SO100 environment.

    This model uses a subset of the full observation space:
      - "agent_qpos": e.g. the robot's joint positions.
      - "sensor_data_hand_camera_rgb": e.g. the hand camera's RGB image.

    In the forward pass:
      1. Unflatten the complete observation tensor using the full observation space.
      2. Convert the resulting list into a dictionary using unflatttened_space_to_dict.
      3. Extract only the keys in self.observed_keys.
      4. Combine the batch dimension with the num_env dimension.
      5. For the camera observation, permute from channel-last to channel-first and merge
         the image frames and RGB channels (i.e. 4 * 3 = 12 channels).
      6. Process both sub-observations through dedicated networks, combine their features,
         and produce policy and value predictions.
    """
    def __init__(self,
                 observation_space,
                 action_space,
                 device):
        # Initialize the base Model class.
        super().__init__(observation_space, action_space, device)
                         
        self.device = device
        # Define the keys to be used from the observation space.
        self.observed_keys = ["agent_qpos", "sensor_data_hand_camera_rgb"]

        # ----- Network for "agent_qpos" -----
        # Assume the observation space for "agent_qpos" is a gym.spaces.Box and provides a shape.
        agent_qpos_shape = self.observation_space["agent_qpos"].shape

        self.agent_qpos_dim = int(torch.prod(torch.tensor(agent_qpos_shape)))
        self.agent_qpos_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.agent_qpos_dim, 64),
            nn.ReLU()
        )

        # ----- Network for "sensor_data_hand_camera_rgb" -----
        # The observation space is defined as: (4, 128, 128, 3)
        # After unflattening, an RGB observation has shape:
        #   [batch, 4, 128, 128, 3]
        # We assume images are channel-last by default, so we set:
        self.camera_channel_last = True
        # and then merge the 4 image frames with the 3 channels (i.e. 4*3 = 12 channels).
        cam_shape = self.observation_space["sensor_data_hand_camera_rgb"].shape  # (4, 128, 128, 3)
        self.camera_conv_input_channels = cam_shape[0] * cam_shape[-1]  # 4 * 3 = 12
        # Create a small CNN to encode the camera observations.
        self.camera_net = nn.Sequential(
            nn.Conv2d(self.camera_conv_input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # ----- Combined Network -----
        # Concatenate features from both sub-networks and process with a fully-connected layer.
        self.combined_net = nn.Sequential(
            nn.Linear(64 + 32, 128),
            nn.ReLU()
        )
        self.num_actions = self.action_space.shape[-1]
        self.policy_head = nn.Linear(128, self.num_actions)
        self.value_head = nn.Linear(128, 1)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))
        self.to(device)

    def _compute(self, inputs, role):
        """
        Forward pass of the model.

        Parameters:
          obs (torch.Tensor): A flattened tensor of observations.

        Returns:
          A tuple (policy, value): The outputs from the policy and value heads.
        """
        # -----------------------------------------------
        # 1. Unflatten the complete observation tensor.
        #    The unflatten op returns a list, so we convert it to a dict.
        # -----------------------------------------------
        inputs = inputs['states']
        obs_dict = unflatten_tensorized_space(self.observation_space, inputs) 
        # -----------------------------------------------
        # 2. Extract only the selected sub-observations.
        # -----------------------------------------------
        agent_qpos = obs_dict[self.observed_keys[0]]  # Expected shape: [B, num_env, ...]
        camera_obs = obs_dict[self.observed_keys[1]]    # Expected shape: [B, num_env, 4, 128, 128, 3]

        # -----------------------------------------------
        # 3. Combine batch and num_env dimensions for agent_qpos.
        # -----------------------------------------------
        # B_qpos, N_qpos = agent_qpos.shape[:2]
        # agent_qpos = agent_qpos.view(B_qpos * N_qpos, *agent_qpos.shape[2:])
        # Flatten the remaining dimensions.
        agent_qpos = agent_qpos.view(agent_qpos.size(0), -1)
        agent_features = self.agent_qpos_net(agent_qpos)

        # -----------------------------------------------
        # 4. Combine batch and num_env dimensions for camera_obs.
        # -----------------------------------------------
        # Expected camera_obs shape: [B, num_env, 4, 128, 128, 3]
        # B_cam, N_cam = camera_obs.shape[:2]
        # camera_obs = camera_obs.view(B_cam * N_cam, *camera_obs.shape[2:])
        # Now camera_obs has shape: [B*N, 4, 128, 128, 3]

        # Since images are in channel-last format by default, we permute:
        # from [B*N, 4, 128, 128, 3] to [B*N, 4, 3, 128, 128]
        camera_obs = camera_obs.permute(0, 1, 4, 2, 3)
        # Next, merge the 4 image frames with the 3 color channels into one channel dimension:
        # resulting in shape: [B*N, 4*3=12, 128, 128]
        B_N, L, C, H, W = camera_obs.shape  # (L should be 4 and C should be 3)
        camera_obs = camera_obs.reshape(B_N, L * C, H, W)
        camera_features = self.camera_net(camera_obs)

        # -----------------------------------------------
        # 5. Combine features and produce the policy and value outputs.
        # -----------------------------------------------
        combined_features = torch.cat((agent_features, camera_features), dim=1)
        combined_features = self.combined_net(combined_features)
        policy_output = self.policy_head(combined_features)
        value_output = self.value_head(combined_features)

        if role == "policy":
            return policy_output, self.log_std_parameter, {}
        elif role == "value":
            return value_output, {}
        
class SO100PPOActor(MultivariateGaussianMixin, SO100PPOModel):
    def __init__(self, observation_space, action_space, device="cuda:0",
                 clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2):
        SO100PPOModel.__init__(self, observation_space, action_space, device)
        MultivariateGaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)
        self.to(device)

    def compute(self, inputs, role):
        return self._compute(inputs, role)
    
class SO100PPOCritic(DeterministicMixin, SO100PPOModel):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        SO100PPOModel.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        self.to(device)

    def compute(self, inputs, role):
        return self._compute(inputs, role)