# Soft Rasterizer (SlangPy + PyTorch)

This is a port of the slang-torch soft-rasterizer example to SlangPy with PyTorch interop.
- Original example: https://github.com/shader-slang/slang-torch/tree/main/examples/soft-rasterizer-example
- Note: slangpy-samples already includes a soft-rasterizer example without PyTorch interop (see `examples/soft-rasterizer/`). This sample shows the PyTorch-integrated variant.

## Contents
- `soft_rasterizer2d_itensor.slang` — per-pixel soft rasterizer using `ITensor<>` for differentiable inputs (vertices, color).
- `main.py` — host script that renders a target image, optimizes triangle vertices/color with Adam, and saves an animation.

## Requirements
- CUDA-capable GPU
- PyTorch (install the correct wheel for your CUDA/GPU per https://pytorch.org/get-started/locally/)
- Python packages from requirements.txt

Setup (example)
```bash
# 1) Install PyTorch for your environment
#    CUDA 12.9 example:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

# 2) Install sample dependencies
pip install -r requirements.txt
```

## Run
```bash
python main.py
```
This generates `rasterizer2d_itensor.mp4` with three panels: current render, gradient visualization, and target.

## Overview (SlangPy + PyTorch)

This variant mirrors the original soft-rasterizer, but is written in the SlangPy style and integrates with PyTorch. For a deeper walkthrough of the soft rasterization algorithm itself, see `examples/soft-rasterizer/` in this repo.

### 1) Soft rasterization recap
- Replace the discontinuous triangle test with a continuous function based on distance to the triangle and a sigmoid “soft” step.
- Mark helper functions and kernel as `[Differentiable]` so Slang can generate backward code.

### 2) Slang kernel (SlangPy style)
- We write a per-pixel function that SlangPy auto-vectorizes over the image.
- Inputs use `ITensor<>` so shapes are runtime-sized and differentiable.
- We get the pixel coordinate from `spy.call_id()` (bound to a `uint2 grid_cell`).

Example (excerpt) from `soft_rasterizer2d_itensor.slang`:

```c
[Differentiable]
float3 rasterize_pixel(
    no_diff uniform Camera camera,
    ITensor<float, 2> vertices,   // shape [3,2]
    ITensor<float, 1> color,      // shape [3]
    no_diff float sigma,
    no_diff uint2 grid_cell)
{
    float2 pixel = float2(grid_cell) + 0.5;
    float2 uv = pixel / camera.frameDim;
    // compute softTriangle(...) using distanceToTriangle(...)
    // mix foreground/background to produce float3 color
    return result;
}
```

### 3) PyTorch integration
- Create a torch-compatible device via `spy.create_torch_device(...)` (CUDA recommended).
- Call the Slang function from Python with preallocated `_result=img` PyTorch tensor.
  - SlangPy detects the PyTorch tensor and wraps the call to `rasterize_pixel` in a custom autograd function; as a result, `loss.backward()` automatically invokes `module.rasterize_pixel.bwds(...)` to compute gradients.

### 4) Training loop (simplified)
```python
img = torch.empty((W, H, 3), dtype=torch.float32, device='cuda', requires_grad=True)
module.rasterize_pixel(
    camera=camera,
    vertices=vertices, color=color,
    sigma=sigma,
    grid_cell=spy.call_id(),
    _result=img,
)
loss = ((img - target_img)**2).mean()
loss.backward()
optim.step()
```
