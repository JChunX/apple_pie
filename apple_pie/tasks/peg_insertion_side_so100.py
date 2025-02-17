"""
This module defines a custom PegInsertionSide task that supports the so100 robot,
which only has 6 degrees of freedom. It subclasses the original PegInsertionSideEnv
and overrides both the _load_agent and _initialize_episode methods.

By overriding _load_agent we add a "tcp" attribute to the agent (for example, by taking
the last link of the robot articulation). The _initialize_episode method then generates
a qpos vector with 6 elements, rather than the original 9, to match the so100 configuration.

Usage (from command line):
    streamlit run play_env.py -- --env PegInsertionSideSO100-v1 --robot_uids so100_wristcam --policy apple_pie.policies.pick_cube:PickCubeRandomPolicySO100 --render_mode sensors --obs_mode state+rgb
"""

import numpy as np
import sapien
import torch

from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose

from apple_pie.embodiments.so100 import SO100WristCam
from typing import Union, Dict

@register_env("PegInsertionSideSO100-v1", max_episode_steps=100)
class PegInsertionSideSO100Env(PegInsertionSideEnv):
    """
    PegInsertionSide task subclass for the so100 robot.

    Changes:
    - Supports only 6 DOF by generating a 6-element qpos vector.
    - Overrides _load_agent to set the "tcp" attribute (using the last robot link)
      so that observation functions relying on self.agent.tcp do not fail.
    """
    # Limit accepted robot IDs to the so100 robot.
    SUPPORTED_ROBOTS = ["so100_wristcam"]
    agent: Union[SO100WristCam]
    _clearance = 0.003

    def __init__(
        self,
        *args,
        robot_uids="so100_wristcam",
        num_envs=1,
        reconfiguration_freq=None,
        **kwargs,
    ):
        super().__init__(*args, robot_uids=robot_uids, num_envs=num_envs, reconfiguration_freq=reconfiguration_freq, **kwargs)

    def _load_agent(self, options: dict):
        BaseEnv._load_agent(self, options, sapien.Pose([-0.2, 0, 0]))

    @property
    def _default_sensor_configs(self):
        return self.agent._sensor_configs

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.5, -0.5, 0.8], [0.05, -0.1, 0.4])
        return [CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100, shader_pack="rt-fast")] + self.agent._sensor_configs

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            # ---------------------
            # Initialize the peg
            # ---------------------
            xy = randomization.uniform(
                low=torch.tensor([-0.1, -0.3]),
                high=torch.tensor([0.1, 0]),
                size=(b, 2)
            )
            pos = torch.zeros((b, 3))
            pos[:, :2] = xy
            # Use the pre-computed peg_half_sizes to set the z component.
            pos[:, 2] = self.peg_half_sizes[env_idx, 2]
            quat = randomization.random_quaternions(
                b,
                self.device,
                lock_x=True,
                lock_y=True,
                bounds=(np.pi / 2 - np.pi / 3, np.pi / 2 + np.pi / 3),
            )
            self.peg.set_pose(Pose.create_from_pq(pos, quat))

            # ---------------------
            # Initialize the box
            # ---------------------
            xy = randomization.uniform(
                low=torch.tensor([-0.05, 0.2]),
                high=torch.tensor([0.05, 0.4]),
                size=(b, 2)
            )
            pos = torch.zeros((b, 3))
            pos[:, :2] = xy
            pos[:, 2] = self.peg_half_sizes[env_idx, 0]
            quat = randomization.random_quaternions(
                b,
                self.device,
                lock_x=True,
                lock_y=True,
                bounds=(np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8),
            )
            self.box.set_pose(Pose.create_from_pq(pos, quat))

            # ---------------------
            # Initialize the so100 robot with 6 DOF
            # ---------------------
            # Define a base qpos that is appropriate for a 6-DOF so100 robot.
            base_qpos = self.agent.keyframes["rest"].qpos
            # Add a small amount of noise (standard deviation 0.02) to the base configuration.
            qpos = self._episode_rng.normal(0, 0.02, (b, len(base_qpos))) + base_qpos
            self.agent.robot.set_qpos(qpos)