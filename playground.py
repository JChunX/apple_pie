import gymnasium as gym
import torch
from apple_pie.embodiments import *
from apple_pie.tasks import *
from skrl.envs.wrappers.torch import wrap_env
from skrl.utils.spaces.torch import unflatten_tensorized_space
from apple_pie.util.env_wrappers import GeneralStateStackingWrapper, UnnestObservationWrapper, ManiSkillToSKRLWrapper
from apple_pie.policies.so100 import SO100PPOModel
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
 v
from skrl.memories.torch import RandomMemory

def test_env(env):
    print("--------------------------------")
    obs, info = env.reset()
    action = torch.rand(2, 6)
    print("spaces", env.observation_space)
    print("action", action)
    print("action shape", action.shape)

    obs, reward, terminated, truncated, info = env.step(action)
    print(obs)

# Create the environment
env = gym.make("PegInsertionSideSO100-v1", num_envs=2, obs_mode="state_dict+rgb", control_mode="pd_joint_delta_pos", render_mode="sensors", robot_uids="so100_wristcam", max_episode_steps=300)

# Test the base environment
test_env(env)

# Wrap with UnnestObservationWrapper and test
env = UnnestObservationWrapper(env)
test_env(env)

# Wrap with GeneralStateStackingWrapper and test
env = GeneralStateStackingWrapper(env, 4)
test_env(env)

# Wrap with ManiSkillToSKRLWrapper and test
env = ManiSkillToSKRLWrapper(env)
test_env(env)


# Wrap with wrap_env and test
# env = wrap_env(env, "gymnasium")
# test_env(env)
obs, info = env.reset()

model = SO100PPOModel(env.observation_space, env.action_space, device="cuda")

# next_states, rewards, terminated, truncated, infos = env.step(policy.detach())

# print("env is _vectorized:", env._vectorized)
# print("action space", env.action_space)
# print("action policy.shape", policy.shape)
# print("observation space", env.observation_space)

# print("next_states.shape", next_states.shape)
# print("rewards.shape", rewards.shape)
# print("terminated.shape", terminated.shape)
# print("truncated.shape", truncated.shape)
# print("infos", infos)
cfg_agent = PPO_DEFAULT_CONFIG.copy()
cfg_agent["wandb"] = True
cfg_agent["wandb_kwargs"] = {
    "entity": "jchunx",
    "project": "so100_peg_insert_ppo_train"
}

agent = PPO(memory=RandomMemory(memory_size=256),
            models={
                "policy": model,
                "value": model,
            },
            observation_space=env.observation_space,
            action_space=env.action_space,
            device="cuda",
            cfg=cfg_agent)

# import gymnasium as gym

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # import the skrl components to build the RL system
# from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
# from skrl.envs.wrappers.torch import wrap_env
# from skrl.memories.torch import RandomMemory
# from skrl.models.torch import DeterministicMixin, Model
# from skrl.resources.noises.torch import OrnsteinUhlenbeckNoise
# from skrl.trainers.torch import SequentialTrainer
# from skrl.utils import set_seed


# # seed for reproducibility
# set_seed()  # e.g. `set_seed(42)` for fixed seed


# # define models (deterministic models) using mixin
# class Actor(DeterministicMixin, Model):
#     def __init__(self, observation_space, action_space, device, clip_actions=False):
#         Model.__init__(self, observation_space, action_space, device)
#         DeterministicMixin.__init__(self, clip_actions)

#         self.linear_layer_1 = nn.Linear(self.num_observations, 400)
#         self.linear_layer_2 = nn.Linear(400, 300)
#         self.action_layer = nn.Linear(300, self.num_actions)

#     def compute(self, inputs, role):
#         x = F.relu(self.linear_layer_1(inputs["states"]))
#         x = F.relu(self.linear_layer_2(x))
#         # Pendulum-v1 action_space is -2 to 2
#         return 2 * torch.tanh(self.action_layer(x)), {}

# class Critic(DeterministicMixin, Model):
#     def __init__(self, observation_space, action_space, device, clip_actions=False):
#         Model.__init__(self, observation_space, action_space, device)
#         DeterministicMixin.__init__(self, clip_actions)

#         self.linear_layer_1 = nn.Linear(self.num_observations + self.num_actions, 400)
#         self.linear_layer_2 = nn.Linear(400, 300)
#         self.linear_layer_3 = nn.Linear(300, 1)

#     def compute(self, inputs, role):
#         x = F.relu(self.linear_layer_1(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)))
#         x = F.relu(self.linear_layer_2(x))
#         return self.linear_layer_3(x), {}


# # load and wrap the gymnasium environment.
# # note: the environment version may change depending on the gymnasium version
# try:
#     env = gym.make_vec("Pendulum-v1", num_envs=10, vectorization_mode="sync")
# except (gym.error.DeprecatedEnv, gym.error.VersionNotFound) as e:
#     env_id = [spec for spec in gym.envs.registry if spec.startswith("Pendulum-v")][0]
#     print("Pendulum-v1 not found. Trying {}".format(env_id))
#     env = gym.make_vec(env_id, num_envs=10, vectorization_mode="sync")
# env = wrap_env(env)
# env.reset()

# print("env is _vectorized:", env._vectorized)
# next_states, rewards, terminated, truncated, infos = env.step(torch.randn(10,1))
# print("action space", env.action_space)
# print("observation space", env.observation_space)
# print("next_states.shape", next_states.shape)
# print("rewards.shape", rewards.shape)
# print("terminated.shape", terminated.shape)
# print("truncated.shape", truncated.shape)
# print("infos", infos)
