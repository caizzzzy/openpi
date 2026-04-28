#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CLIENT_VENV_DIR="${SCRIPT_DIR}/.venv-client310"
CLIENT_PYTHON_BIN="${CLIENT_VENV_DIR}/bin/python"
LEROBOT_DIR="/mnt/nas/projects/robot/lerobot"

BASE_PYTHON=${BASE_PYTHON:-python3.10}

if ! command -v "${BASE_PYTHON}" >/dev/null 2>&1; then
    echo "找不到 ${BASE_PYTHON}，请先安装 Python 3.10。"
    exit 1
fi

echo "=========================================="
echo "创建 Diana client Python 3.10 环境"
echo "Base Python: $(${BASE_PYTHON} --version)"
echo "Venv: ${CLIENT_VENV_DIR}"
echo "=========================================="

"${BASE_PYTHON}" -m venv "${CLIENT_VENV_DIR}"

"${CLIENT_PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel

"${CLIENT_PYTHON_BIN}" -m pip install \
    "numpy==1.26.4" \
    "scipy>=1.11,<1.15" \
    "torch==2.7.1" \
    "tyro>=0.9.5" \
    "datasets>=4.0.0,<4.2.0" \
    "pandas>=2.2.2,<2.4.0" \
    "opencv-python-headless>=4.9.0,<4.13.0" \
    "draccus==0.10.0" \
    "pyserial>=3.5,<4.0" \
    "deepdiff>=7.0.1,<9.0.0"

"${CLIENT_PYTHON_BIN}" -m pip install -e "${SCRIPT_DIR}/packages/openpi-client"
"${CLIENT_PYTHON_BIN}" -m pip install -e "${LEROBOT_DIR}" --no-deps

echo
echo "客户端环境已准备好。"
echo "解释器路径: ${CLIENT_PYTHON_BIN}"
echo
echo "建议用下面命令快速验证："
echo "  source /opt/ros/humble/setup.bash"
echo "  source /mnt/nas/projects/robot/pika_ros/install/setup.bash"
echo "  ${CLIENT_PYTHON_BIN} - <<'PY'"
echo "import rclpy, cv_bridge, lerobot, openpi_client, tyro, numpy, websockets, PIL"
echo "print('client env ok')"
echo "PY"
