# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import datetime
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import slangpy as spy

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from materials.ceramic.mdl_ceramic_material import CeramicMaterial

if __package__:
    from .checkpoint import (
        PARAMETER_COUNT,
        ParameterLayoutConverter,
        initialize_parameters,
        load_parameters,
        tensor_from_numpy,
    )
    from .common import VectorBackend, create_device, require_wave_safe_work_size
else:
    from checkpoint import (
        PARAMETER_COUNT,
        ParameterLayoutConverter,
        initialize_parameters,
        load_parameters,
        tensor_from_numpy,
    )
    from common import VectorBackend, create_device, require_wave_safe_work_size


def cosine_schedule(
    iteration: int,
    total_iterations: int,
    maximum: float,
    minimum: float,
) -> float:
    t = min(iteration, total_iterations)
    return minimum + 0.5 * (maximum - minimum) * (1.0 + math.cos(math.pi * t / total_iterations))


def color_augmentation_ratio(iteration: int, enabled: bool) -> float:
    if not enabled or iteration >= 10_000:
        return 0.0
    return 0.5 * (1.0 + math.cos(math.pi * iteration / 10_000.0))


class Trainer:
    def __init__(
        self,
        vector_backend: VectorBackend,
        seed: int,
        resume: Path | None,
    ) -> None:
        self.vector_backend = vector_backend
        self.device = create_device(vector_backend)
        self.module = spy.Module(self.device.load_module("training"))
        self.material = CeramicMaterial(self.device)

        self.parameter_layout = ParameterLayoutConverter(self.device, vector_backend)
        self.parameter_count = self.parameter_layout.storage_parameter_count
        values = load_parameters(resume) if resume else initialize_parameters(seed)
        self.parameters = self.parameter_layout.from_portable(values)
        self.gradients = tensor_from_numpy(
            self.device, np.zeros((self.parameter_count,), dtype=np.float32)
        )
        self.first_moment = tensor_from_numpy(
            self.device, np.zeros((self.parameter_count,), dtype=np.float32)
        )
        self.second_moment = tensor_from_numpy(
            self.device, np.zeros((self.parameter_count,), dtype=np.float32)
        )
        self._batch_size = 0
        self._batch_encoder_input: spy.Tensor | None = None
        self._batch_wi: spy.Tensor | None = None
        self._batch_wo: spy.Tensor | None = None
        self._batch_reference: spy.Tensor | None = None
        self._batch_valid: spy.Tensor | None = None
        self.iteration = self._resume_iteration(resume) if resume else 0

    @property
    def parameter_address(self) -> int:
        return self.parameters.storage.device_address

    @property
    def gradient_address(self) -> int:
        return self.gradients.storage.device_address

    def globals(self) -> dict[str, object]:
        return self.material.to_global_uniforms()

    def train_step(
        self,
        batch_size: int,
        learning_rate: float,
        augmentation_ratio: float,
    ) -> None:
        self._ensure_batch(batch_size)
        assert self._batch_encoder_input is not None
        assert self._batch_wi is not None
        assert self._batch_wo is not None
        assert self._batch_reference is not None
        assert self._batch_valid is not None
        shape = (batch_size, batch_size)
        self.module.prepareTrainingSample.set(self.globals())(
            seed=self.iteration,
            batchIndex=spy.grid(shape),
            batchSize=spy.int2(shape),
            colorAugmentationRatio=augmentation_ratio,
            encoderInputOutput=self._batch_encoder_input,
            wiOutput=self._batch_wi,
            woOutput=self._batch_wo,
            referenceOutput=self._batch_reference,
            validOutput=self._batch_valid,
        )
        self.module.calculateGradients(
            batchIndex=spy.grid(shape),
            encoderInput=self._batch_encoder_input,
            wiInput=self._batch_wi,
            woInput=self._batch_wo,
            referenceInput=self._batch_reference,
            validInput=self._batch_valid,
            parameters=self.parameter_address,
            parameterGradients=self.gradient_address,
        )
        self.iteration += 1
        self.module.optimizerStep(
            index=spy.grid((self.parameter_count,)),
            parameters=self.parameter_address,
            parameterGradients=self.gradient_address,
            firstMoment=self.first_moment.storage.device_address,
            secondMoment=self.second_moment.storage.device_address,
            learningRate=learning_rate,
            iteration=self.iteration,
        )

    def _ensure_batch(self, batch_size: int) -> None:
        if self._batch_size == batch_size:
            return
        self._batch_size = batch_size
        self._batch_encoder_input = spy.Tensor.zeros(
            self.device, (batch_size, batch_size, 29), "float"
        )
        self._batch_wi = spy.Tensor.zeros(self.device, (batch_size, batch_size, 3), "float")
        self._batch_wo = spy.Tensor.zeros(self.device, (batch_size, batch_size, 3), "float")
        self._batch_reference = spy.Tensor.zeros(self.device, (batch_size, batch_size, 3), "float")
        self._batch_valid = spy.Tensor.zeros(self.device, (batch_size, batch_size), "float")

    def validation_loss(self, size: int = 256, seed: int = 0x5EED) -> float:
        shape = (size, size)
        require_wave_safe_work_size(self.vector_backend, size * size, "validation batch")
        output = spy.Tensor.zeros(self.device, shape, "float")
        self.module.computeLoss.set(self.globals())(
            seed=seed,
            batchIndex=spy.grid(shape),
            batchSize=spy.int2(shape),
            parameters=self.parameter_address,
            output=output,
        )
        values = output.to_numpy()
        valid = values >= 0.0
        return float(values[valid].mean()) if np.any(valid) else float("nan")

    def bake_latents(self, resolution: int) -> np.ndarray:
        require_wave_safe_work_size(
            self.vector_backend,
            resolution * resolution,
            "latent texture",
        )
        tensor = spy.Tensor.zeros(self.device, (resolution, resolution, 8), dtype="float")
        self.module.bakeLatentTexture.set(self.globals())(
            pixel=spy.grid((resolution, resolution)),
            resolution=spy.int2(resolution, resolution),
            parameters=self.parameter_address,
            latentTexture=tensor,
        )
        self.device.wait()
        return tensor.to_numpy()

    def save_checkpoint(
        self,
        run_dir: Path,
        latent_resolution: int,
        config: dict[str, object],
    ) -> Path:
        checkpoint_dir = run_dir / f"{self.iteration:07d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        latent_texture = self.bake_latents(latent_resolution)
        parameters = self.parameter_layout.to_portable(self.parameters)
        np.savez(
            checkpoint_dir / "network.npz",
            parameters=parameters,
            latent_texture=latent_texture,
        )
        with (checkpoint_dir / "train_config.json").open("w") as stream:
            json.dump({**config, "iteration": self.iteration}, stream, indent=2)
        print(f"[iter {self.iteration:7d}] checkpoint saved -> {checkpoint_dir}")
        return checkpoint_dir

    def close(self) -> None:
        self.device.wait()
        self.device.close()

    @staticmethod
    def _resume_iteration(resume: Path | None) -> int:
        if resume is None:
            return 0
        directory = resume if resume.is_dir() else resume.parent
        config_path = directory / "train_config.json"
        if config_path.is_file():
            with config_path.open() as stream:
                config = json.load(stream)
            return int(config.get("iteration", config.get("optimize_counter", 0)))
        try:
            return int(directory.name)
        except ValueError:
            return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the neural.slang ceramic encoder/decoder.")
    parser.add_argument("--vector-backend", choices=("inline", "wave"), default="inline")
    parser.add_argument("--iterations", type=int, default=16_384)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--schedule-iterations", type=int, default=100_000)
    parser.add_argument("--validation-interval", type=int, default=64)
    parser.add_argument("--validation-size", type=int, default=256)
    parser.add_argument("--checkpoint-interval", type=int, default=4096)
    parser.add_argument("--latent-resolution", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--no-color-augmentation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations must be positive")
    if args.batch_size < 1 or args.latent_resolution < 1:
        raise ValueError("Batch and latent dimensions must be positive")

    backend: VectorBackend = args.vector_backend
    require_wave_safe_work_size(backend, args.batch_size * args.batch_size, "training batch")

    run_dir = args.run_dir
    if run_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = HERE / "runs" / timestamp
    run_dir = run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")
    print(f"Vector backend: {backend}")

    trainer = Trainer(backend, args.seed, args.resume)
    config = {
        "vector_backend": backend,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "min_learning_rate": args.min_learning_rate,
        "schedule_iterations": args.schedule_iterations,
        "latent_resolution": args.latent_resolution,
        "parameter_count": PARAMETER_COUNT,
        "parameter_storage_count": trainer.parameter_count,
        "parameter_layout": "optimal" if backend == "wave" else "linear",
        "color_augmentation": not args.no_color_augmentation,
    }

    final_iteration = trainer.iteration + args.iterations
    last_saved = -1
    start = time.perf_counter()
    try:
        while trainer.iteration < final_iteration:
            learning_rate = cosine_schedule(
                trainer.iteration,
                args.schedule_iterations,
                args.learning_rate,
                args.min_learning_rate,
            )
            augmentation = color_augmentation_ratio(
                trainer.iteration, not args.no_color_augmentation
            )
            trainer.train_step(args.batch_size, learning_rate, augmentation)

            if args.validation_interval > 0 and trainer.iteration % args.validation_interval == 0:
                loss = trainer.validation_loss(args.validation_size)
                elapsed = time.perf_counter() - start
                print(
                    f"[iter {trainer.iteration:7d}] lr={learning_rate:.2e} "
                    f"loss={loss:.6f} elapsed={elapsed:.1f}s"
                )

            if args.checkpoint_interval > 0 and trainer.iteration % args.checkpoint_interval == 0:
                trainer.save_checkpoint(run_dir, args.latent_resolution, config)
                last_saved = trainer.iteration

        if last_saved != trainer.iteration:
            trainer.save_checkpoint(run_dir, args.latent_resolution, config)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
