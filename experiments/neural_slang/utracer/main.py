# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from pathlib import Path

if __package__:
    from .common import VectorBackend, create_device, require_wave_safe_work_size
    from .renderer import Camera, MicroScene, Viewer, save_tonemapped
else:
    from common import VectorBackend, create_device, require_wave_safe_work_size
    from renderer import Camera, MicroScene, Viewer, save_tonemapped


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the neural-only UTracer.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vector-backend", choices=("inline", "wave"), default="inline")
    parser.add_argument("--mesh", type=Path, default=HERE / "data/shaderball.glb")
    parser.add_argument(
        "--environment",
        type=Path,
        default=HERE / "data/brown_photostudio_02_2k.hdr",
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--spp", type=int, default=1)
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Render this many frames headlessly; 0 opens the interactive viewer.",
    )
    parser.add_argument("--output", type=Path, default=HERE / "render.png")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Compile the selected specialization and load the checkpoint, then exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.width < 1 or args.height < 1 or args.spp < 1 or args.frames < 0:
        raise ValueError("Image dimensions and spp must be positive; frames cannot be negative")
    backend: VectorBackend = args.vector_backend
    require_wave_safe_work_size(backend, args.width * args.height, "render target")

    device = create_device(backend, execution_mode="inference")
    viewer: Viewer | None = None
    try:
        scene = MicroScene(device, args.checkpoint, backend, args.spp)
        print(f"Loaded checkpoint: {scene.checkpoint.path}")
        print(f"Vector backend: {backend}")
        if args.validate_only:
            return

        scene.load_mesh(args.mesh, rescale_to=1.0)
        scene.set_environment(args.environment)
        scene.build()
        camera = Camera(device, args.width, args.height)

        if args.frames > 0:
            camera.reset_accumulator(scene.module)
            output = camera.output
            for _ in range(args.frames):
                output = scene.render(camera)
            device.wait()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            save_tonemapped(output, args.output)
            print(f"Saved render: {args.output.resolve()}")
        else:
            viewer = Viewer(scene, camera)
            viewer.run()
    finally:
        # Viewer.close() already owns the surface/device/window ordering.
        if viewer is None:
            device.close()


if __name__ == "__main__":
    main()
