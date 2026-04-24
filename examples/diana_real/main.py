import dataclasses
import logging

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

from examples.diana_real import env as _env


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8080
    api_key: str | None = None

    robot_port: str = "192.168.10.76"
    control_mode: str = "joint"
    prompt: str | None = None

    action_horizon: int = 50
    max_hz: float = 30
    num_episodes: int = 1
    max_episode_steps: int = 0
    dry_run: bool = False

    render_height: int = 224
    render_width: int = 224

    image_topic: str = "/camera/color/image_raw"
    image_topic_fisheye: str = "/camera_fisheye/color/image_raw"
    image_topic_global: str = "/global_camera/color/image_raw"
    pika_pose_topic: str = "/pika_pose"
    gripper_state_topic: str = "/gripper/joint_state"
    gripper_ctrl_topic: str = "/joint_states"


def main(args: Args) -> None:
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
    )
    logging.info("Server metadata: %s", ws_client_policy.get_server_metadata())

    environment = _env.DianaRealEnvironment(
        _env.DianaRobotArgs(
            port=args.robot_port,
            control_mode=args.control_mode,
            image_topic=args.image_topic,
            image_topic_fisheye=args.image_topic_fisheye,
            image_topic_global=args.image_topic_global,
            pika_pose_topic=args.pika_pose_topic,
            gripper_state_topic=args.gripper_state_topic,
            gripper_ctrl_topic=args.gripper_ctrl_topic,
        ),
        prompt=args.prompt,
        dry_run=args.dry_run,
        render_height=args.render_height,
        render_width=args.render_width,
    )

    runtime = _runtime.Runtime(
        environment=environment,
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=ws_client_policy,
                action_horizon=args.action_horizon,
            )
        ),
        subscribers=[],
        max_hz=args.max_hz,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    try:
        runtime.run()
    finally:
        environment.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
