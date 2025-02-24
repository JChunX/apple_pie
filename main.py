import gymnasium as gym
import mani_skill.envs
import streamlit as st
import numpy as np
import time
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import wandb

import torch
from apple_pie.embodiments import *
from apple_pie.tasks import *
from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG

# HYDRA_FULL_ERROR=1 streamlit run main.py -- --config-name=so100_peg_insert_ppo_train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_env(cfg: DictConfig):
    """
    Creates and returns the ManiSkill environment.
    
    Parameters:
        cfg: Hydra configuration object containing environment settings
        
    Returns:
        A Gymnasium environment instance.
    """
    env = gym.make(
        cfg.name,
        num_envs=cfg.num_envs,
        obs_mode=cfg.obs_mode,
        control_mode=cfg.control_mode,
        render_mode=cfg.render_mode,
        robot_uids=cfg.robot_uids,
        max_episode_steps=cfg.max_episode_steps
    )
    if cfg.wrappers is not None:    
        for wrapper in cfg.wrappers:
            env = instantiate(wrapper, env)
    return env

def process_image(frame):
    if hasattr(frame, 'cpu'):  # if frame is a torch tensor
        frame = frame.cpu().numpy()
    if frame.dtype != np.uint8:
        frame = (frame * 255).astype(np.uint8)
    return frame

def rollout(policy, env):
    """
    Runs the simulation rollout using the provided policy and environment.
    
    Parameters:
         policy: An instance of a policy with an 'act' method.
         env: The environment to run the rollout on.
    """
    # Create placeholders for image and metrics
    image_placeholder = st.empty()
    col1, col2, col3 = st.columns(3)
    reward_text = col1.empty()
    terminated_text = col2.empty()
    step_text = col3.empty()

    # Reset the environment
    obs, _ = env.reset(seed=0)
    done = False
    step = 0

    while not done:
        obs_input = obs if isinstance(obs, dict) else {"obs": obs}
        action = policy.act(obs_input)
        obs, reward, terminated, truncated, info = env.step(action)
        print(obs)
        done = terminated or truncated

        frame = process_image(env.render())
        reward_item = reward[0].cpu().numpy() if hasattr(reward[0], 'cpu') else reward[0]

        # Update the UI components
        image_placeholder.image(frame, caption=f"Step {step}", use_column_width=True)

        reward_text.metric("Reward", f"{reward_item:.3f}")
        terminated_text.metric("Terminated", terminated)
        step_text.metric("Step", step)

        step += 1
        time.sleep(0.1)

    env.close()

def run_training(cfg: DictConfig):
    """
    Trains the agent using the PPO algorithm.
    This function:
      • Initializes wandb tracking.
      • Creates the environment and wraps it with a custom VisionStateStackingWrapper.
      • Instantiates the VisionStateActorCritic network and wraps it into PPO.
      • Runs a dummy rollout/update loop over multiple epochs.
    """
    clean_cfg = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(config=clean_cfg, entity=cfg.wandb.entity, project=cfg.wandb.project)

    env = create_env(cfg.environment)

    models_factory = {
        k: instantiate(v) for k, v in cfg.models.items()
    }
    models = {
        k: models_factory[k](env.observation_space, env.action_space, cfg.device) for k in models_factory.keys()
    }
    memory = instantiate(cfg.agent.memory, 
                         num_envs=cfg.environment.num_envs, 
                         device=cfg.device)


    agent_cfg = cfg.agent.cfg.copy()
    agent_cfg["experiment"]["wandb"] = True
    agent_cfg["experiment"]["wandb_kwargs"] = {
        "entity": cfg.wandb.entity,
        "project": cfg.wandb.project,
    }
    
    agent = instantiate(cfg.agent, 
                        memory=memory,
                        models=models, 
                        observation_space=env.observation_space, 
                        action_space=env.action_space, 
                        device=cfg.device,
                        cfg=agent_cfg)

    trainer = instantiate(cfg.trainer, env=env, agents=agent)
    trainer.train()


def run_evaluation(cfg: DictConfig):
    st.title("ManiSkill Environment Visualization")
    print(cfg)

    # Dynamically import the policy class
    agent = instantiate(cfg.agent.module)

    if st.button("Run Simulation"):
        # Create the environment using the helper function
        env = create_env(cfg.environment)
        # Run rollout using the chosen policy and environment
        rollout(agent, env)
        st.success("Simulation completed!")

    st.write("### Effective Configuration")
    st.code(OmegaConf.to_yaml(cfg), language="yaml")


@hydra.main(version_base=None, config_path="cfg")
def main(cfg: DictConfig):
    """
    Main function that runs the environment visualization with Hydra config
    """
    OmegaConf.to_yaml(cfg)
    if cfg.mode == "training":
        run_training(cfg)
    else:
        run_evaluation(cfg)

if __name__ == "__main__":
    main()