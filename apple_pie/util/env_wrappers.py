import numpy as np
import torch
import time
import gymnasium as gym
from collections import deque
from apple_pie.embodiments import *
from apple_pie.tasks import *
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common
from gymnasium.spaces import Box, Dict
from skrl.utils.spaces.torch import (
    flatten_tensorized_space,
    tensorize_space,
    unflatten_tensorized_space,
    untensorize_space,
)
from typing import Any, Tuple

class UnnestObservationWrapper(gym.ObservationWrapper):
    """
    Completely unnests all dictionaries while preserving original observation shapes
    """
    def __init__(self, env) -> None:
        super().__init__(env)
        spaces = {}
        init_obs = self.base_env._init_raw_obs
        
        def unnest_spaces(nested_dict, parent_key=''):
            flat_spaces = {}
            for key, value in nested_dict.items():
                new_key = f"{parent_key}_{key}" if parent_key else key
                if isinstance(value, gym.spaces.Dict):
                    # Recursively unnest Dict spaces
                    flat_spaces.update(unnest_spaces(value.spaces, new_key))
                else:
                    flat_spaces[new_key] = value
            return flat_spaces

        for key, value in self.base_env.observation_space.items():
            if isinstance(value, gym.spaces.Dict):
                spaces.update(unnest_spaces(value.spaces, key))
            else:
                spaces[key] = value
        
        self.observation_space = gym.spaces.Dict(spaces)

    @property
    def base_env(self) -> BaseEnv:
        return self.env.unwrapped

    def observation(self, observation):
        def unnest_dict(nested_dict, parent_key=''):
            flat_dict = {}
            for key, value in nested_dict.items():
                new_key = f"{parent_key}_{key}" if parent_key else key
                if isinstance(value, dict):
                    flat_dict.update(unnest_dict(value, new_key))
                else:
                    flat_dict[new_key] = value
            return flat_dict

        processed_obs = {}
        for key, value in observation.items():
            if isinstance(value, dict):
                processed_obs.update(unnest_dict(value, key))
            else:
                processed_obs[key] = value
        return processed_obs

class GeneralStateStackingWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, frame_stack: int = 4):
        super().__init__(env)
        self.frame_stack = frame_stack
        
        # Initialize frame stacks (deques) and update observation space
        self.frame_stacks = self._initialize_frame_stacks(self.env.observation_space)
        self.observation_space = self._get_stacked_observation_space(self.env.observation_space)

    # Include the corrected method
    def _get_stacked_observation_space(self, space):
        if isinstance(space, Box):
            num_envs = space.shape[0]
            single_shape = space.shape[1:]
            stacked_shape = (num_envs, self.frame_stack) + single_shape
            
            if np.isscalar(space.low):
                low = space.low
                high = space.high
            else:
                low = np.repeat(space.low[:, np.newaxis, ...], self.frame_stack, axis=1)
                high = np.repeat(space.high[:, np.newaxis, ...], self.frame_stack, axis=1)
                low = low.reshape(stacked_shape)
                high = high.reshape(stacked_shape)
            
            return Box(low=low, high=high, shape=stacked_shape, dtype=space.dtype)
        elif isinstance(space, Dict):
            return Dict({key: self._get_stacked_observation_space(subspace) for key, subspace in space.spaces.items()})
        else:
            raise ValueError(f"Unsupported space type: {type(space)}")

    # Placeholder for other methods (ensure they align with the new observation space)
    def _initialize_frame_stacks(self, space):
        from collections import deque
        if isinstance(space, Box):
            return deque(maxlen=self.frame_stack)
        elif isinstance(space, Dict):
            return {key: self._initialize_frame_stacks(subspace) for key, subspace in space.spaces.items()}
        else:
            raise ValueError(f"Unsupported space type: {type(space)}")

    def reset(self):
        obs, info = self.env.reset()
        self._initialize_deques(obs, self.frame_stacks)
        return self._get_observation(self.frame_stacks), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._update_deques(obs, self.frame_stacks)
        return self._get_observation(self.frame_stacks), reward, terminated, truncated, info

    def _initialize_deques(self, obs, frame_stacks):
        if isinstance(frame_stacks, deque):
            frame_stacks.clear()
            for _ in range(self.frame_stack):
                # Use clone() for PyTorch tensors
                frame_stacks.append(obs.clone() if isinstance(obs, torch.Tensor) else obs.copy())
        elif isinstance(frame_stacks, dict):
            for key in frame_stacks:
                self._initialize_deques(obs[key], frame_stacks[key])

    def _update_deques(self, obs, frame_stacks):
        if isinstance(frame_stacks, deque):
            frame_stacks.append(obs.clone() if isinstance(obs, torch.Tensor) else obs.copy())
        elif isinstance(frame_stacks, dict):
            for key in frame_stacks:
                self._update_deques(obs[key], frame_stacks[key])

    def _get_observation(self, frame_stacks):
        if isinstance(frame_stacks, deque):
            return torch.stack(list(frame_stacks), dim=1)  # Use dim=1 to stack after num_envs
        elif isinstance(frame_stacks, dict):
            return {key: self._get_observation(substacks) for key, substacks in frame_stacks.items()}
        else:
            raise ValueError("Invalid frame_stacks structure")

from skrl.envs.wrappers.torch.base import Wrapper
class ManiSkillToSKRLWrapper(Wrapper):
    def __init__(self, env) -> None:
        super().__init__(env)
        self._reset_once = True
        self._observations = None
        self._info = {}
        self._vectorized = True

        self._observation_space = self._create_single_spaces()
        self._action_space = self._create_single_action_space()

    @property
    def observation_space(self):
        return self._observation_space
    
    @property
    def action_space(self):
        return self._action_space

    def _create_single_spaces(self):
        single_spaces = {}
        for key, space in self._env.observation_space.items():
            # The shape of the vectorized space is (num_env, ...)
            # For the single space, we need the shape without num_env, i.e., space.shape[1:]
            single_shape = space.shape[1:]
            
            # Get the low and high bounds
            low = space.low
            high = space.high
            
            # Check if low and high include num_env dimension
            # If low.shape == space.shape (e.g., (num_env, ...)), assume bounds are the same
            # across environments and take the first slice
            if low.shape == space.shape:
                low = low[0]
            if high.shape == space.shape:
                high = high[0]
            
            # Create new Box space for single environment
            single_spaces[key] = gym.spaces.Box(
                low=low,
                high=high,
                shape=single_shape,
                dtype=space.dtype
            )
        return gym.spaces.Dict(single_spaces)
    
    def _create_single_action_space(self):
        action_space = self._env.action_space
        single_shape = action_space.shape[1:]
        return gym.spaces.Box(
            low=action_space.low[0],
            high=action_space.high[0],
            shape=single_shape,
            dtype=action_space.dtype
        )

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of torch.Tensor and any other info
        """
        actions = untensorize_space(
            self.action_space,
            unflatten_tensorized_space(self.action_space, actions),
            squeeze_batch_dimension=not self._vectorized,
        )
        observation, reward, terminated, truncated, info = self._env.step(actions)

        # convert response to torch
        observation = flatten_tensorized_space(tensorize_space(self.observation_space, observation, self.device))
        reward = torch.tensor(reward, device=self.device, dtype=torch.float32).view(self.num_envs, -1)
        terminated = torch.tensor(terminated, device=self.device, dtype=torch.bool).view(self.num_envs, -1)
        truncated = torch.tensor(truncated, device=self.device, dtype=torch.bool).view(self.num_envs, -1)

        # save observation and info for vectorized envs
        if self._vectorized:
            self._observation = observation
            self._info = info

        return observation, reward, terminated, truncated, info

    def reset(self) -> Tuple[torch.Tensor, Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: torch.Tensor and any other info
        """
        # handle vectorized environments (vector environments are autoreset)
        if self._vectorized:
            if self._reset_once:
                observation, self._info = self._env.reset()
                self._observation = flatten_tensorized_space(
                    tensorize_space(self.observation_space, observation, self.device)
                )
                self._reset_once = False
            return self._observation, self._info

        observation, info = self._env.reset()
        observation = flatten_tensorized_space(tensorize_space(self.observation_space, observation, self.device))
        return observation, info

    def render(self, *args, **kwargs) -> Any:
        """Render the environment"""
        if self._vectorized:
            return self._env.call("render", *args, **kwargs)
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        """Close the environment"""
        self._env.close()