# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import slangpy as spy


VectorBackend = Literal["inline", "wave"]
DeviceBackend = Literal["vulkan", "metal"]
NeuralExecutionMode = Literal["training", "inference"]
HERE = Path(__file__).resolve().parent


def default_device_backend() -> DeviceBackend:
    return "metal" if sys.platform == "darwin" else "vulkan"


def create_device(
    vector_backend: VectorBackend = "inline",
    *,
    device_backend: DeviceBackend | None = None,
    execution_mode: NeuralExecutionMode = "inference",
) -> spy.Device:
    """Create a GPU device specialized for neural training or inference."""
    if vector_backend not in ("inline", "wave"):
        raise ValueError(f"Unknown vector backend: {vector_backend}")
    if device_backend is None:
        device_backend = default_device_backend()
    if device_backend not in ("vulkan", "metal"):
        raise ValueError(f"Unknown device backend: {device_backend}")
    if execution_mode not in ("training", "inference"):
        raise ValueError(f"Unknown neural execution mode: {execution_mode}")

    device = spy.Device(
        type={
            "vulkan": spy.DeviceType.vulkan,
            "metal": spy.DeviceType.metal,
        }[device_backend],
        compiler_options={
            "include_paths": [spy.SHADER_PATH, HERE],
            "defines": {
                "NEURAL_VECTOR_WAVE": "1" if vector_backend == "wave" else "0",
                "NEURAL_TARGET_CUDA": "0",
                "NEURAL_TARGET_METAL": "1" if device_backend == "metal" else "0",
                "NEURAL_EXECUTION_INFERENCE": ("1" if execution_mode == "inference" else "0"),
            },
            "enable_experimental_features": True,
        },
        enable_hot_reload=False,
    )

    # Slang RHI currently exposes this feature bit for Vulkan cooperative-matrix
    # extensions, while Metal lowers the same Slang capability to simdgroup_matrix.
    if (
        vector_backend == "wave"
        and device_backend == "vulkan"
        and spy.Feature.cooperative_matrix not in device.features
    ):
        device.close()
        raise RuntimeError(
            f"The wave backend requires cooperative-matrix support on {device_backend}. "
            "Use --vector-backend inline on this device."
        )
    return device


def require_wave_safe_work_size(vector_backend: VectorBackend, item_count: int, label: str) -> None:
    """WaveTangledVector requires every lane in each 32-thread group to participate."""
    if vector_backend == "wave" and item_count % 32 != 0:
        raise ValueError(
            f"{label} has {item_count} items; the wave backend requires a multiple of 32 "
            "so no lane exits before a workgroup barrier."
        )
