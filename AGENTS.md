# AGENTS.md

This file is a quick working map for coding agents in this repository. It favors practical orientation over exhaustive documentation.

## Project Shape

`openpi` is a Python robotics policy repository from Physical Intelligence. It contains JAX and PyTorch implementations of PI VLA models, policy wrappers, training/data pipelines, a WebSocket inference server, a lightweight client package, and runnable robot/simulator examples.

Core paths:

- `src/openpi/`: main package.
- `packages/openpi-client/`: small dependency client/runtime package for robot-side inference.
- `scripts/`: training, normalization-stat computation, and policy serving entrypoints.
- `examples/`: robot and benchmark integrations for ALOHA sim/real, DROID, LIBERO, simple client, UR5.
- `docs/`: setup notes for Docker, remote inference, and normalization stats.
- `third_party/`: submodules/vendor code; avoid broad edits here.

## Environment And Tooling

- Python project managed by `uv`; root package requires Python `>=3.11`.
- `packages/openpi-client` supports Python `>=3.7` and is included as a `uv` workspace member.
- Use `GIT_LFS_SKIP_SMUDGE=1 uv sync` for setup, matching the README.
- Lint/format with Ruff. Config lives in `pyproject.toml`:
  - line length: `120`
  - target version: `py311` for root, `py37` for `openpi-client`
  - import sorting is strict and mostly single-line imports.
- Pre-commit runs `uv-lock`, `ruff --fix`, and `ruff-format`.
- Pytest config:
  - test paths: `src`, `scripts`, `packages`
  - manual tests use marker `manual`
  - CI command: `uv run pytest --strict-markers -m "not manual"`

Useful commands:

```bash
uv run pytest --strict-markers -m "not manual"
uv run pytest src/openpi/transforms_test.py
uv run ruff check .
uv run ruff format .
uv run scripts/compute_norm_stats.py --config-name pi05_libero
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite
uv run scripts/serve_policy.py --env ALOHA_SIM
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid --policy.dir=gs://openpi-assets/checkpoints/pi05_droid
```

## Main Architecture

### Configs Drive Training And Inference

The central registry is `src/openpi/training/config.py`.

- `TrainConfig` defines model, data, optimizer, checkpoint, wandb, FSDP, and policy metadata.
- `_CONFIGS` registers named configs such as `pi0_aloha`, `pi05_droid`, `pi05_libero`, `pi0_aloha_sim`, and debug configs.
- Use `get_config(name)` in code and the Tyro CLI in scripts.
- `DataConfigFactory` subclasses build `DataConfig` objects for LeRobot, RLDS/DROID, fake data, ALOHA, LIBERO, and DROID variants.
- `AssetsConfig` controls normalization statistics and other assets. For fine-tuning, pay close attention to whether a config recomputes stats or reuses checkpoint stats.

When adding a new robot/dataset, usually start here:

1. Add or adapt policy transforms in `src/openpi/policies/*_policy.py`.
2. Add a `DataConfigFactory` or reuse `SimpleDataConfig`.
3. Register a `TrainConfig` in `_CONFIGS`.
4. Compute norm stats unless intentionally reusing stats from a base/fine-tuned checkpoint.

### Transform Pipeline

Transforms live in `src/openpi/transforms.py` and are the main bridge between robot/dataset schemas and model schemas.

Key concepts:

- `Group(inputs=..., outputs=...)` bundles forward and reverse transforms.
- `Group.push()` appends input transforms and prepends output transforms.
- `RepackTransform` maps nested dataset keys into the expected structure.
- `Normalize` / `Unnormalize` use stats from `openpi.shared.normalize`.
- Model transforms handle prompt injection, image resize, tokenization, FAST action extraction, and state/action padding.

Expected model-side data is generally organized around image dictionaries, proprioceptive `state`, tokenized prompt fields, and `actions`.

### Models

JAX model code is in `src/openpi/models/`.

- `model.py`: base model interfaces, observations/actions, restore helpers.
- `pi0.py`, `pi0_config.py`: PI0/PI05 flow-matching models and config.
- `pi0_fast.py`, `gemma_fast.py`: PI0-FAST autoregressive model.
- `gemma.py`, `siglip.py`, `vit.py`, `tokenizer.py`: model components.
- `lora.py`: LoRA support.

PyTorch model code is in `src/openpi/models_pytorch/`.

- `pi0_pytorch.py`, `gemma_pytorch.py`, `preprocessing_pytorch.py`.
- `transformers_replace/` contains local patches for `transformers==4.53.2`; README warns these can affect the uv cache when copied into site-packages.
- PyTorch support currently does not cover PI0-FAST, mixed precision training, FSDP, LoRA training, or EMA training.

### Training

JAX training entrypoint: `scripts/train.py`.

- Builds a data loader via `openpi.training.data_loader.create_data_loader`.
- Initializes JAX/Flax NNX model state.
- Loads partial or full weights via `openpi.training.weight_loaders`.
- Saves checkpoints and norm stats through `openpi.training.checkpoints`.
- Uses W&B unless `wandb_enabled=False`.

PyTorch training entrypoint: `scripts/train_pytorch.py`.

- Uses the same config/data pipeline where possible.
- Supports single GPU, DDP single-node, and DDP multi-node via `torchrun`.
- Saves `model.safetensors`, `optimizer.pt`, `metadata.pt`, and assets.

Data loading is in `src/openpi/training/data_loader.py`.

- LeRobot datasets are wrapped with torch-style loaders.
- DROID full-dataset training uses RLDS through `droid_rlds_dataset.py`.
- `FakeDataset` powers debug configs and lightweight tests.

### Policy And Serving

Policy creation is in `src/openpi/policies/policy_config.py`.

- `create_trained_policy()` downloads/checks checkpoint paths with `download.maybe_download`.
- It detects PyTorch checkpoints by presence of `model.safetensors`.
- It loads norm stats from checkpoint assets for inference, not from the training config asset directory.
- It composes input transforms as repack -> default prompt -> robot data transforms -> normalize -> model transforms.
- It composes output transforms as model outputs -> unnormalize -> robot output transforms -> repack outputs.

Runtime policy wrapper: `src/openpi/policies/policy.py`.

- `Policy.infer()` accepts an unbatched observation dict and returns `actions` plus timing metadata.
- JAX uses jitted `sample_actions`; PyTorch converts leaves to tensors on the chosen device.
- `PolicyRecorder` can dump input/output records under `policy_records/`.

Serving:

- `scripts/serve_policy.py` creates default or checkpoint-backed policies and starts a server.
- `src/openpi/serving/websocket_policy_server.py` sends metadata on connect, accepts msgpack-numpy observations, returns actions, and exposes `/healthz`.
- Default env modes: `ALOHA`, `ALOHA_SIM`, `DROID`, `LIBERO`.

Client:

- `packages/openpi-client/src/openpi_client/websocket_client_policy.py` is the robot-side WebSocket client.
- `msgpack_numpy.py` handles numpy serialization.
- `image_tools.py` has resize/uint8 helpers shared by examples.
- `action_chunk_broker.py` manages open-loop execution from action chunks.
- `runtime/` provides `Runtime`, `Environment`, `Agent`, and `Subscriber` interfaces used by examples.

## Examples

- `examples/aloha_sim/`: simulator runtime. Start server with `uv run scripts/serve_policy.py --env ALOHA_SIM`; run sim with its own Python 3.10 env per README, or Docker compose.
- `examples/aloha_real/`: real ALOHA robot integration, data conversion, robot utilities, video display.
- `examples/droid/`: DROID inference, data conversion, DROID training notes, non-idle range computation.
- `examples/libero/`: LIBERO benchmark evaluation and data conversion.
- `examples/simple_client/`: minimal random-observation client to test inference without a robot.
- `examples/convert_jax_model_to_pytorch.py`: checkpoint conversion utility.

Most examples include their own `requirements.in`, `requirements.txt`, `Dockerfile`, and `compose.yml`. Keep robot-specific dependency changes local to the example when possible.

## Data And Assets

- Checkpoints default to `gs://openpi-assets/...` and are cached under `~/.cache/openpi`.
- `OPENPI_DATA_HOME` can override the cache/download location.
- Training outputs default to `./checkpoints/<config>/<exp_name>/`.
- Config assets default to `./assets/<config>/`.
- Norm stats are required for real datasets before training unless a config intentionally points to existing assets.
- `scripts/compute_norm_stats.py` writes stats to `config.assets_dirs / data_config.repo_id`.
- See `docs/norm_stats.md` before changing action/state conventions or reusing base-model statistics.

## Style And Editing Notes

- Prefer dataclasses and typed, explicit config objects; this repo leans on Tyro for CLIs.
- Keep transform order deliberate. A small order change can affect both training and inference.
- Prefer adding tests near existing `*_test.py` files in the touched package.
- Avoid modifying generated lock data unless dependency changes require it.
- Avoid broad changes in `third_party/` and in `src/openpi/models_pytorch/transformers_replace/*` unless the task is specifically about those patches.
- For robot integrations, keep observation/action schema changes in policy transform files rather than scattering key rewrites through runtime code.
- When changing inference behavior, check both direct `Policy` use and WebSocket server/client use.

## Quick Debug Paths

- Config not found or wrong model/data pairing: inspect `_CONFIGS` in `src/openpi/training/config.py`.
- Missing norm stats: run `scripts/compute_norm_stats.py` or verify `AssetsConfig.assets_dir` and `asset_id`.
- WebSocket client hangs: server is not listening or host/port are wrong; client retries forever on connection refusal.
- Server sends a string response: it is an exception traceback from policy inference.
- PyTorch checkpoint not detected: verify `model.safetensors` exists directly in the checkpoint directory passed to `create_trained_policy()`.
- Image shape/type mismatch: check `openpi_client.image_tools`, `ResizeImages`, and the robot-specific input transform.
- Action dimension mismatch: check `model.action_dim`, `PadStatesAndActions`, robot output transforms, and reused norm stats dimensions.

## High-Value First Reads

For most tasks, read these files first:

1. `README.md`
2. `pyproject.toml`
3. `src/openpi/training/config.py`
4. `src/openpi/transforms.py`
5. `src/openpi/policies/policy_config.py`
6. The relevant `src/openpi/policies/*_policy.py`
7. The relevant example README and `main.py`

For the currently opened ALOHA simulator flow, the most relevant files are:

- `examples/aloha_sim/README.md`
- `examples/aloha_sim/main.py`
- `examples/aloha_sim/env.py`
- `scripts/serve_policy.py`
- `src/openpi/training/config.py` config `pi0_aloha_sim`
- `src/openpi/policies/aloha_policy.py`
