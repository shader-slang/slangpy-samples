# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Literal

import slangpy as spy


VectorBackend = Literal["inline", "wave"]
NeuralExecutionMode = Literal["training", "inference"]
HERE = Path(__file__).resolve().parent


def create_device(
    vector_backend: VectorBackend = "inline",
    *,
    execution_mode: NeuralExecutionMode = "inference",
) -> spy.Device:
    """Create a Vulkan device specialized for neural training or inference."""
    if vector_backend not in ("inline", "wave"):
        raise ValueError(f"Unknown vector backend: {vector_backend}")
    if execution_mode not in ("training", "inference"):
        raise ValueError(f"Unknown neural execution mode: {execution_mode}")

    device = spy.Device(
        type=spy.DeviceType.vulkan,
        compiler_options={
            "include_paths": [spy.SHADER_PATH, HERE],
            "defines": {
                "NEURAL_VECTOR_WAVE": "1" if vector_backend == "wave" else "0",
                "NEURAL_TARGET_CUDA": "0",
                "NEURAL_EXECUTION_INFERENCE": (
                    "1" if execution_mode == "inference" else "0"
                ),
            },
            "enable_experimental_features": True,
        },
        enable_hot_reload=False,
    )

    if vector_backend == "wave" and spy.Feature.cooperative_matrix not in device.features:
        device.close()
        raise RuntimeError(
            "The wave backend requires Vulkan cooperative-matrix support. "
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
