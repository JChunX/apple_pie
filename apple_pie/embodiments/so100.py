import sapien
import numpy as np
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils.structs.actor import Actor
from mani_skill.sensors.camera import CameraConfig
import torch
from mani_skill.utils import common, sapien_utils
from scipy.spatial.transform import Rotation as R

@register_agent()
class SO100Agent(BaseAgent):
    uid = "so100"
    urdf_path = "/teamspace/studios/this_studio/code/SO-ARM100/URDF/SO_5DOF_ARM100_8j_URDF.SLDASM/urdf/SO_5DOF_ARM100_8j_URDF.SLDASM.urdf"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

@register_agent()
class SO100WristCam(SO100Agent):
    uid = "so100_wristcam"
    ee_link_name = "Wrist_Pitch_Roll"
    camera_width = 128
    camera_height = 128

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [
                    np.deg2rad(90),
                    -np.deg2rad(90),
                    np.deg2rad(90),
                    -np.deg2rad(90),
                    np.deg2rad(90),
                    0.0,
                ]
            ),
            pose=sapien.Pose(),
        )
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _sensor_configs(self):
        p = np.array([0, 0, 0.05])
        q = R.from_euler('xyz', [90, 0, -90], degrees=True).as_quat(scalar_first=True)
        # q = np.array([1, 0, 0, 0])
        return [
            CameraConfig(
                uid="hand_camera",
                pose=sapien.Pose(p=p, q=q),
                width=self.camera_width,
                height=self.camera_height,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.robot.links_map["Fixed_Jaw"],
                shader_pack="minimal", #rt-fast
            )
        ]

    def _after_init(self):
        self.finger1_link = sapien_utils.get_obj_by_name(self.robot.get_links(), "Fixed_Jaw")
        self.finger2_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "Moving Jaw"
        )
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """Check if the robot is grasping an object

        Args:
            object (Actor): The object to check if the robot is grasping
            min_force (float, optional): Minimum force before the robot is considered to be grasping the object in Newtons. Defaults to 0.5.
            max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 85.
        """
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        # direction to open the gripper
        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :-2]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold