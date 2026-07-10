# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Mapping

import numpy as np
import slangpy as spy


NUM_LATENTS = 8

# This is also the canonical linear PointerAddress order in model.slang.
LAYER_SPECS = (
    ("encoder_layer0", 29, 64),
    ("encoder_layer1", 64, 64),
    ("encoder_layer2", 64, 64),
    ("encoder_layer3", 64, 8),
    ("decoder_layer0", 14, 32),
    ("decoder_layer1", 32, 32),
    ("decoder_layer2", 32, 3),
)
PARAMETER_COUNT = sum(inputs * outputs + outputs for _, inputs, outputs in LAYER_SPECS)
ENCODER_PARAMETER_COUNT = sum(inputs * outputs + outputs for _, inputs, outputs in LAYER_SPECS[:4])
DECODER_PARAMETER_OFFSET = ENCODER_PARAMETER_COUNT


def resolve_checkpoint_npz(checkpoint: str | PathLike[str]) -> Path:
    path = Path(checkpoint).expanduser()
    if path.is_dir():
        path = path / "network.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Neural checkpoint not found: {path}")
    return path.resolve()


def initialize_parameters(seed: int = 42) -> np.ndarray:
    """Xavier-uniform weights and zero biases in the shared flat layout."""
    rng = np.random.default_rng(seed)
    packed: list[np.ndarray] = []
    for _, inputs, outputs in LAYER_SPECS:
        scale = np.sqrt(6.0 / float(inputs + outputs))
        weights = rng.uniform(-scale, scale, (outputs, inputs)).astype(np.float32)
        biases = np.zeros((outputs,), dtype=np.float32)
        packed.extend((weights.reshape(-1), biases))
    result = np.concatenate(packed)
    assert result.size == PARAMETER_COUNT
    return result


def _find_old_prefixes(data: Mapping[str, np.ndarray]) -> tuple[str, ...]:
    candidates = (
        tuple(name for name, _, _ in LAYER_SPECS),
        (
            "encoder_e0",
            "encoder_e1",
            "encoder_e2",
            "encoder_e3",
            "decoder_d0",
            "decoder_d1",
            "decoder_d2",
        ),
    )
    for prefixes in candidates:
        if all(f"{prefix}_weights" in data and f"{prefix}_biases" in data for prefix in prefixes):
            return prefixes
    raise KeyError(
        "Checkpoint has neither a flat 'parameters' array nor a recognized "
        "encoder/decoder layer set."
    )


def flat_parameters_from_mapping(data: Mapping[str, np.ndarray]) -> np.ndarray:
    """Read the new flat format or convert the previous per-layer format."""
    if "parameters" in data:
        parameters = np.asarray(data["parameters"], dtype=np.float32).reshape(-1)
        if parameters.size != PARAMETER_COUNT:
            raise ValueError(f"Expected {PARAMETER_COUNT} parameters, got {parameters.size}.")
        return np.ascontiguousarray(parameters)

    prefixes = _find_old_prefixes(data)
    packed: list[np.ndarray] = []
    for prefix, (_, inputs, outputs) in zip(prefixes, LAYER_SPECS):
        weights = np.asarray(data[f"{prefix}_weights"], dtype=np.float32)
        biases = np.asarray(data[f"{prefix}_biases"], dtype=np.float32)
        if weights.shape != (outputs, inputs):
            raise ValueError(
                f"{prefix}_weights has shape {weights.shape}; expected {(outputs, inputs)}."
            )
        if biases.shape != (outputs,):
            raise ValueError(f"{prefix}_biases has shape {biases.shape}; expected {(outputs,)}.")
        packed.extend((weights.reshape(-1), biases))
    return np.ascontiguousarray(np.concatenate(packed), dtype=np.float32)


def load_parameters(checkpoint: str | PathLike[str]) -> np.ndarray:
    path = resolve_checkpoint_npz(checkpoint)
    with np.load(path) as data:
        return flat_parameters_from_mapping(data)


def tensor_from_numpy(device: spy.Device, values: np.ndarray) -> spy.Tensor:
    values = np.ascontiguousarray(values, dtype=np.float32)
    usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
    tensor = spy.Tensor.empty(device, values.shape, "float", usage=usage)
    tensor.storage.copy_from_numpy(values)
    return tensor


@dataclass
class BakedCheckpoint:
    path: Path
    parameters: spy.Tensor
    latent_texture: spy.Tensor

    @property
    def parameter_address(self) -> int:
        return self.parameters.storage.device_address


def load_baked_checkpoint(
    device: spy.Device,
    checkpoint: str | PathLike[str],
) -> BakedCheckpoint:
    path = resolve_checkpoint_npz(checkpoint)
    with np.load(path) as data:
        parameters_np = flat_parameters_from_mapping(data)
        if "latent_texture" not in data:
            raise KeyError(
                "Checkpoint does not contain 'latent_texture'. Run train.py to bake one."
            )
        latents_np = np.asarray(data["latent_texture"], dtype=np.float32)
        if latents_np.ndim != 3 or latents_np.shape[2] != NUM_LATENTS:
            raise ValueError(f"Expected latent texture HxWx{NUM_LATENTS}, got {latents_np.shape}.")
        latents_np = np.ascontiguousarray(latents_np)

    return BakedCheckpoint(
        path=path,
        parameters=tensor_from_numpy(device, parameters_np),
        latent_texture=tensor_from_numpy(device, latents_np),
    )
