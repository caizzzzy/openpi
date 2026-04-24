import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


ACTION_DIM = 8


def make_diana_example() -> dict:
    """Creates a random input example for the Diana policy."""
    return {
        "observation/state": np.random.rand(ACTION_DIM),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/global_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick and place",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DianaInputs(transforms.DataTransformFn):
    """Inputs for Diana fine-tuning and inference.

    Expected inputs:
    - observation/image: primary camera image.
    - observation/global_image: secondary global camera image.
    - observation/state: [7 joints, gripper].
    - actions: optional [action_horizon, 7 joints + gripper] during training.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        global_image = _parse_image(data["observation/global_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": global_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class DianaOutputs(transforms.DataTransformFn):
    """Outputs for Diana inference."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :ACTION_DIM])}
