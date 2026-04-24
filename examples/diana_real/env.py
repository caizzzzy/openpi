import dataclasses
import logging
from typing import Any

import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

logger = logging.getLogger(__name__)


JOINT_NAMES = tuple(f"joint_{i}.pos" for i in range(7))
GRIPPER_NAME = "gripper.pos"


@dataclasses.dataclass(frozen=True)
class DianaRobotArgs:
    port: str = "192.168.10.76"
    control_mode: str = "joint"
    image_topic: str = "/camera/color/image_raw"
    image_topic_fisheye: str = "/camera_fisheye/color/image_raw"
    image_topic_global: str = "/global_camera/color/image_raw"
    pika_pose_topic: str = "/pika_pose"
    gripper_state_topic: str = "/gripper/joint_state"
    gripper_ctrl_topic: str = "/joint_states"


class DianaRealEnvironment(_environment.Environment):
    """OpenPI runtime adapter around the LeRobot DianaFollower hardware bridge."""

    def __init__(
        self,
        robot_args: DianaRobotArgs,
        *,
        prompt: str | None = None,
        dry_run: bool = False,
        render_height: int = 224,
        render_width: int = 224,
    ) -> None:
        if robot_args.control_mode != "joint":
            raise ValueError(
                "The current OpenPI Diana policy expects joint control with state/action "
                "[joint_0..joint_6, gripper]. Use --control-mode=joint."
            )

        self._prompt = prompt
        self._dry_run = dry_run
        self._render_height = render_height
        self._render_width = render_width

        try:
            from lerobot.robots.diana_follower import DianaFollower
            from lerobot.robots.diana_follower import DianaFollowerConfig
        except ImportError as exc:
            raise ImportError(
                "Could not import lerobot DianaFollower. Add /mnt/nas/projects/robot/lerobot/src to PYTHONPATH "
                "and source the ROS2 environment before starting this client."
            ) from exc

        cameras = {
            "cam_high": {"width": 640, "height": 480, "fps": 30},
            "cam_global": {"width": 640, "height": 480, "fps": 30},
        }
        config = DianaFollowerConfig(
            port=robot_args.port,
            control_mode=robot_args.control_mode,
            image_topic=robot_args.image_topic,
            image_topic_fisheye=robot_args.image_topic_fisheye,
            image_topic_global=robot_args.image_topic_global,
            pika_pose_topic=robot_args.pika_pose_topic,
            gripper_state_topic=robot_args.gripper_state_topic,
            gripper_ctrl_topic=robot_args.gripper_ctrl_topic,
            cameras=cameras,
        )
        self._robot = DianaFollower(config)
        self._robot.connect()
        self._last_action: dict[str, float] | None = None

    @override
    def reset(self) -> None:
        self._last_action = None

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict[str, Any]:
        raw_obs = self._robot.get_observation()
        state = np.asarray([raw_obs[name] for name in (*JOINT_NAMES, GRIPPER_NAME)], dtype=np.float32)

        obs = {
            "observation.state": state,
            "observation.images.cam_high": self._prepare_image(raw_obs["cam_high"]),
            "observation.images.cam_global": self._prepare_image(raw_obs["cam_global"]),
        }
        if self._prompt:
            obs["prompt"] = self._prompt
        return obs

    @override
    def apply_action(self, action: dict) -> None:
        action_vec = np.asarray(action["actions"], dtype=np.float32)
        if action_vec.shape != (8,):
            raise ValueError(f"Expected Diana action shape (8,), got {action_vec.shape}")

        robot_action = {
            name: float(value) for name, value in zip((*JOINT_NAMES, GRIPPER_NAME), action_vec, strict=True)
        }
        self._last_action = robot_action
        if self._dry_run:
            logger.info("Dry run action: %s", robot_action)
            return

        self._robot.send_action(robot_action)

    def close(self) -> None:
        self._robot.disconnect()

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        image = image_tools.convert_to_uint8(np.asarray(image))
        return image_tools.resize_with_pad(image, self._render_height, self._render_width)
