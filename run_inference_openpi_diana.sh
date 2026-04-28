#!/bin/bash

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEROBOT_DIR="/mnt/nas/projects/robot/lerobot"
ROS_DISTRO_SETUP="/opt/ros/humble/setup.bash"
PIKA_ROS_SETUP="/mnt/nas/projects/robot/pika_ros/install/setup.bash"
PYTHON_BIN=$(command -v python)
CLIENT_PYTHON_BIN="${SCRIPT_DIR}/.venv-client310/bin/python"

# ==========================================
# 用户配置区域
# ==========================================

SERVER_HOST="0.0.0.0"
CLIENT_HOST="127.0.0.1"
PORT="8080"

ROBOT_PORT="192.168.10.76"
CONTROL_MODE="joint"

CONFIG_NAME="pi0_diana_pick_place_lora"
CHECKPOINT_DIR="${SCRIPT_DIR}/checkpoints/pi0_diana_pick_place_lora/0405_pi0_lora/30000"
TASK="Pick up the red bell pepper and place it into the box"

ACTION_HORIZON=50
MAX_HZ=30
MAX_EPISODE_STEPS=0
DRY_RUN=False

IMAGE_TOPIC="/camera/color/image_raw"
IMAGE_TOPIC_FISHEYE="/camera_fisheye/color/image_raw"
IMAGE_TOPIC_GLOBAL="/global_camera/color/image_raw"
PIKA_POSE_TOPIC="/pika_pose"
GRIPPER_STATE_TOPIC="/gripper/joint_state"
GRIPPER_CTRL_TOPIC="/joint_states"

# ==========================================
# 脚本逻辑
# ==========================================

MODE=$1

export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/packages/openpi-client/src:${LEROBOT_DIR}/src:${PYTHONPATH}"

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export no_proxy="localhost,127.0.0.1,${CLIENT_HOST},${SERVER_HOST},192.168.0.0/16"
export NO_PROXY="$no_proxy"
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

if [ "$MODE" == "server" ]; then
    echo "=========================================="
    echo "正在启动 OpenPI 策略服务器..."
    echo "监听地址: ${SERVER_HOST}:${PORT}"
    echo "配置: ${CONFIG_NAME}"
    echo "Checkpoint: ${CHECKPOINT_DIR}"
    echo "=========================================="

    cd "${SCRIPT_DIR}"
    "$PYTHON_BIN" scripts/serve_policy.py \
        --port="${PORT}" \
        --default-prompt="${TASK}" \
        policy:checkpoint \
        --policy.config="${CONFIG_NAME}" \
        --policy.dir="${CHECKPOINT_DIR}"

elif [ "$MODE" == "client" ]; then
    if [ -f "$ROS_DISTRO_SETUP" ]; then
        source "$ROS_DISTRO_SETUP"
    else
        echo "未找到 ROS 环境: ${ROS_DISTRO_SETUP}"
        exit 1
    fi

    if [ -f "$PIKA_ROS_SETUP" ]; then
        source "$PIKA_ROS_SETUP"
    fi

    export LD_LIBRARY_PATH="/opt/ros/humble/lib:${LD_LIBRARY_PATH}"
    export PYTHONPATH="/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages:${PYTHONPATH}"

    if [ ! -x "${CLIENT_PYTHON_BIN}" ]; then
        echo "未找到 client 专用 Python: ${CLIENT_PYTHON_BIN}"
        echo "请先运行: ./setup_diana_client_env.sh"
        exit 1
    fi

    echo "=========================================="
    echo "正在启动 Diana 真机客户端..."
    echo "连接至 OpenPI server: ${CLIENT_HOST}:${PORT}"
    echo "机器人: ${ROBOT_PORT} (${CONTROL_MODE})"
    echo "Dry run: ${DRY_RUN}"
    echo "Client Python: ${CLIENT_PYTHON_BIN}"
    echo "=========================================="

    DRY_RUN_ARG="--dry-run"
    if [ "$DRY_RUN" != "True" ] && [ "$DRY_RUN" != "true" ] && [ "$DRY_RUN" != "1" ]; then
        DRY_RUN_ARG="--no-dry-run"
    fi

    "$CLIENT_PYTHON_BIN" -m examples.diana_real.main \
        --host="${CLIENT_HOST}" \
        --port="${PORT}" \
        --robot-port="${ROBOT_PORT}" \
        --control-mode="${CONTROL_MODE}" \
        --prompt="${TASK}" \
        --action-horizon="${ACTION_HORIZON}" \
        --max-hz="${MAX_HZ}" \
        --max-episode-steps="${MAX_EPISODE_STEPS}" \
        "${DRY_RUN_ARG}" \
        --image-topic="${IMAGE_TOPIC}" \
        --image-topic-fisheye="${IMAGE_TOPIC_FISHEYE}" \
        --image-topic-global="${IMAGE_TOPIC_GLOBAL}" \
        --pika-pose-topic="${PIKA_POSE_TOPIC}" \
        --gripper-state-topic="${GRIPPER_STATE_TOPIC}" \
        --gripper-ctrl-topic="${GRIPPER_CTRL_TOPIC}"

else
    echo "用法错误。"
    echo "启动服务器: ./run_inference_openpi_diana.sh server"
    echo "启动客户端: ./run_inference_openpi_diana.sh client"
fi
