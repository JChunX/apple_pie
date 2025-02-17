import gymnasium as gym
import mani_skill.envs
import streamlit as st
import numpy as np
import time
import importlib
import argparse
import sys
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from apple_pie.embodiments import *
from apple_pie.tasks import *

# streamlit run play_env.py -- --env PegInsertionSideSO100-v1 --robot_uids so100_wristcam --policy apple_pie.policies.pick_cube:PickCubeRandomPolicySO100

class EnvironmentVideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.frame = None

    def transform(self, frame):
        if self.frame is not None:
            return self.frame
        return frame

def create_env(env_name: str, **kwargs):
    """
    Creates and returns the ManiSkill environment.
    
    Parameters:
        env_name (str): Name of the environment.
        **kwargs: Additional keyword arguments for gym.make.
        
    Returns:
        A Gymnasium environment instance.
    """
    return gym.make(env_name, **kwargs)

def process_image(frame):
    """
    Process an image frame to ensure it's in the correct format.
    
    Parameters:
        frame: The image frame to process, can be torch tensor or numpy array
        
    Returns:
        Processed numpy array in uint8 format
    """
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
        # Ensure the observation is a dict (since Policy.act expects a dict)
        obs_input = obs if isinstance(obs, dict) else {"obs": obs}
        action = policy.act(obs_input)
        obs, reward, terminated, truncated, info = env.step(action)
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

# --- Main execution ---

# Parse command-line arguments for environment and policy settings.
# Note: When running with Streamlit, you may need to pass these arguments after a '--'
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="PickCube-v1", help="Name of the environment")
    parser.add_argument("--max_episode_steps", type=int, default=300, help="Maximum number of steps per episode")
    parser.add_argument("--robot_uids", type=str, required=False, help="robot_uid for spawning into environment")
    parser.add_argument("--control_mode", type=str, default="pd_joint_delta_pos", required=False, help="control mode for spawning into environment")
    parser.add_argument(
        "--policy",
        type=str,
        default="apple_pie.policies.pick_cube:PickCubeRandomPolicy",
        help="Policy to use in the format <module>:<ClassName>)"
    )
    parser.add_argument("--render_mode", type=str, default="rgb_array", help="Render mode for the environment")
    parser.add_argument("--obs_mode", type=str, default="state", help="Observation mode for the environment")
    args, _ = parser.parse_known_args()

    env_name = args.env
    robot_uids = args.robot_uids
    control_mode = args.control_mode
    policy_spec = args.policy  # expected format "module:ClassName"
    module_name, class_name = policy_spec.split(":")
    max_episode_steps = args.max_episode_steps
    # Dynamically import the policy class
    policy_module = importlib.import_module(module_name)
    policy_class = getattr(policy_module, class_name)
    policy = policy_class()

    render_mode = args.render_mode
    obs_mode = args.obs_mode
    st.title("ManiSkill Environment Visualization")

    if st.button("Run Simulation"):
        # Create the environment using the helper function
        env = create_env(
            env_name,
            num_envs=1,
            obs_mode=obs_mode,
            control_mode=control_mode,
            render_mode=render_mode,
            robot_uids=robot_uids,
            max_episode_steps=max_episode_steps
        )
        # Run rollout using the chosen policy and environment
        rollout(policy, env)
        st.success("Simulation completed!")