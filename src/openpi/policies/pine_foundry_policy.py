"""Policy transforms for the Pine "foundry" Industrial_Arm 3-camera setup.

Dataset schema (LeRobot v2.1):
  - action            float32 [7]  = (x, y, z, rx, ry, rz, gripper)   absolute EEF pose + gripper
  - observation.state float32 [13] = pose(7) + force/torque(6: fx,fy,fz,tx,ty,tz)
  - observation.images.hand / view1 / view2  (480x640x3 video)

pi0.5 has three image slots; all three real cameras are used (masks all True):
    base_0_rgb        <- view1 (primary external)
    left_wrist_0_rgb  <- hand  (wrist)
    right_wrist_0_rgb <- view2 (secondary external)

State and the action chunk are padded to the model action dim; outputs slice back
to the first 7 dims. (Delta-vs-absolute action handling is decided by
``extra_delta_transform`` in the data config, matching pi05_fr3.)
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_pine_foundry_example() -> dict:
    """Creates a random input example (post-repack) for the pine_foundry policy."""
    return {
        "observation/state": np.random.rand(13),
        "observation/image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/extra_view_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": "pick up the disk and insert it",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    image = np.squeeze(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:  # CHW -> HWC
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class PineFoundryInputs(transforms.DataTransformFn):
    """Pine foundry dataset sample -> OpenPI model inputs."""

    # The action dimension of the model (state/action get padded to this).
    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0
    state_dim: int = 13

    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type == _model.ModelType.PI0  # pi0-FAST does not mask

        state = transforms.pad_to_dim(data["observation/state"], self.action_dim)

        base_image = _parse_image(data["observation/image"])           # view1
        wrist_image = _parse_image(data["observation/wrist_image"])     # hand
        extra_image = _parse_image(data["observation/extra_view_image"])  # view2

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": extra_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # All three cameras are real; only pi0 (flow) would mask a *padding*
                # slot, but here the slot is filled, so keep it True for both.
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = transforms.pad_to_dim(data["actions"], self.action_dim)

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class PineFoundryOutputs(transforms.DataTransformFn):
    """Model action space -> dataset action space (first 7 dims)."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}
