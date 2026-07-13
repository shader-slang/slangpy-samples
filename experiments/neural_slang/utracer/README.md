# neural.slang encoder + UTracer

This directory is a neural-only reduction of the original `train/` and
`utracer/` examples. It has two entry points:

- `train.py` trains the encoder/decoder and bakes an 8-channel latent texture.
- `main.py` path traces a mesh using only the baked latents and neural decoder.

The analytic/PBR render branch, MDL render branch, material texture plumbing,
and denoiser are intentionally absent. The trainer still uses the ceramic MDL
material as the teacher that supplies encoder features and reference BSDF
values.

![Neural UTracer rendering the ceramic shader ball](screenshot.png)

## Architecture

![Training and inference architecture for the neural.slang UTracer](architecture.svg)

`model.slang` keeps the `Network`, `Encoder`, and `Decoder` structures. Their
layer implementations are `FFLayer` instances from `slang.neural`:

```text
Encoder: 29 -> 64 -> 64 -> 64 -> 8
Decoder: 14 -> 32 -> 32 -> 3
```

The hidden layers use `LeakyReLU<float>(0.01)`, the encoder output is linear,
and the decoder output uses `ExpActivation<float>`.

Training uses two GPU stages. `prepareTrainingSample` runs the generated MDL
teacher and writes compact feature/direction/reference tensors.
`calculateGradients` reads those tensors and runs only the neural model plus
`bwd_diff`. This keeps the large MDL evaluation out of Slang's reverse-mode
tape and makes the neural implementation much easier to inspect.

The model structures are generic over types constrained by `IVector<float>`.
The `NEURAL_VECTOR_WAVE` specialization selects the matching vector and
parameter layout:

- `InlineVector<float, N>` with `LinearLayout`
- `WaveTangledVector<float, ..., N, 32, 1>` with `OptimalLayout`

Checkpoints remain portable row-major arrays. On the wave backend,
`NetworkParameterLayoutConverter` converts the encoder and decoder separately
to the tiled optimal layout when loading, then converts back to the portable
layout when saving. Runtime parameters remain in one float buffer addressed
through `PointerAddress<float>`.

The layout lifecycle is:

1. Load or initialize portable `LinearLayout` parameters.
2. For wave execution, launch the converter once into a separate
   `OptimalLayout` buffer and use that buffer for training or inference.
3. Before saving wave-trained parameters, launch the reverse conversion and
   write only the portable buffer to the checkpoint.

## Install

From the active SlangPy virtual environment:

```bash
python -m pip install -r utracer/requirements.txt
```

`slang.neural` is experimental in the current Slang release. `common.py`
enables the required compiler feature automatically. The sample uses Vulkan
because the same device must support pointers, cooperative matrices, and ray
queries.

### Backend support

The two entry points currently require Vulkan. CUDA is not supported
end-to-end: UTracer's compute path uses `TraceRayInline`, and the ceramic MDL
teacher uses `Texture.SampleLevel`; neither feature is available to these
generated kernels on the CUDA compute target. The neural model itself is not
the blocker, but both `main.py` and `train.py` depend on one of those stages.

## Run a trained material

`train.py` writes checkpoints under the local `runs/` directory and prints the
saved path. For example, if training saves
`runs/20260710_103810/0016384`, run it with:

```bash
cd utracer
python main.py \
  --checkpoint runs/20260710_103810/0016384
```

Use the cooperative-matrix-backed vector implementation with:

```bash
python main.py \
  --vector-backend wave \
  --checkpoint runs/20260710_103810/0016384
```

Mouse-left drag orbits the camera, the wheel zooms, and Escape closes the
viewer. A headless render is also available:

```bash
python main.py \
  --checkpoint runs/20260710_103810/0016384 \
  --width 1280 --height 720 --spp 4 --frames 16 \
  --output render.png
```

## Train

The default command performs 16,384 additional iterations and saves a final
4096x4096x8 latent texture:

```bash
cd utracer
python train.py --vector-backend inline
```

To exercise `WaveTangledVector` during both forward and backward passes:

```bash
python train.py --vector-backend wave
```

A small smoke run is useful before a full bake:

```bash
python train.py \
  --iterations 1 --batch-size 8 \
  --validation-interval 0 --checkpoint-interval 0 \
  --latent-resolution 8
```

Resume parameters from either the old per-layer format or this sample's flat
format:

```bash
python train.py \
  --resume runs/20260710_120000/0004096 \
  --iterations 4096
```

`--iterations` means additional iterations after the restored iteration.
Optimizer moments are reset on resume to keep the checkpoint intentionally
small and simple.

## PointerAddress parameter layouts

The checkpoint uses the portable linear layout. Every layer stores row-major
weights followed immediately by its bias:

| Layer | Shape | Float offset | Float count |
|---|---:|---:|---:|
| Encoder 0 | 29 -> 64 | 0 | 1,920 |
| Encoder 1 | 64 -> 64 | 1,920 | 4,160 |
| Encoder 2 | 64 -> 64 | 6,080 | 4,160 |
| Encoder 3 | 64 -> 8 | 10,240 | 520 |
| Decoder 0 | 14 -> 32 | 10,760 | 480 |
| Decoder 1 | 32 -> 32 | 11,240 | 1,056 |
| Decoder 2 | 32 -> 3 | 12,296 | 99 |
| Total | | | 12,395 |

The wave backend converts that array to the 16-padded tiled optimal layout:

| Block | Portable floats | Optimal floats | Optimal offset |
|---|---:|---:|---:|
| Encoder | 10,760 | 11,472 | 0 |
| Decoder | 1,635 | 2,128 | 11,472 |
| Total | 12,395 | 13,600 | |

The larger optimal buffer exists only at runtime. New checkpoints contain only:

- `parameters`: `(12395,)`, `float32`
- `latent_texture`: `(H, W, 8)`, `float32`

## Wave safety

`WaveTangledVector` uses workgroup barriers. The tracer therefore executes a
fixed five-bounce loop and makes one decoder call per lane per bounce, even for
lanes whose rays have already terminated. Dummy lanes use safe directions and
UV zero. Render targets, training batches, validation batches, and latent
bakes must contain a multiple of 32 invocations; the Python entry points check
this before dispatch.
