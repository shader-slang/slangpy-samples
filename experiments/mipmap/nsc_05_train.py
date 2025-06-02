# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math

import numpy as np
from app import App
import slangpy as spy

# Create the app and load the slang module.
app = App(width=4126, height=1024, title="Mipmap Example")
module = spy.Module.load_from_file(app.device, "nsc_05_train.slang")

# Load some materials.
albedo_map = spy.Tensor.load_from_image(app.device,
                                        "PavingStones070_2K.diffuse.jpg", linearize=True)
normal_map = spy.Tensor.load_from_image(app.device,
                                        "PavingStones070_2K.normal.jpg", scale=2, offset=-1)

def downsample(source: spy.Tensor, steps: int) -> spy.Tensor:
    for i in range(steps):
        dest = spy.Tensor.empty(device=app.device, shape=(source.shape[0] // 2, source.shape[1] // 2), dtype=source.dtype)
        module.downsample(spy.call_id(), source, _result=dest)
        source = dest
    return source

# Downsampled maps
lr_albedo_map = downsample(albedo_map, 2)
lr_normal_map =  downsample(normal_map, 2)

#module.init_normal(lr_normal_map)

# Corresponding gradients
lr_albedo_grad = spy.Tensor.zeros_like(lr_albedo_map)
lr_normal_grad = spy.Tensor.zeros_like(lr_normal_map)

m_normal = spy.Tensor.zeros_like(lr_normal_grad)
v_normal = spy.Tensor.zeros_like(lr_normal_grad)

def getRandomDir():
    r = math.sqrt(np.random.rand())
    phi = np.random.rand() * math.pi * 2
    Lx = r * math.sin(phi)
    Ly = r * math.cos(phi)
    Lz = math.sqrt(max(1 - r ** 2, 0))
    return spy.float3(Lx, Ly, Lz)

optimize_counter = 0

while app.process_events():

    light_dir = spy.math.normalize(spy.float3(0.2, 0.2, 1.0))

    # Full res rendered output BRDF from full res inputs.
    output = spy.Tensor.empty_like(albedo_map)
    module.render(pixel = spy.call_id(),
                  material = {
                        "albedo": albedo_map,
                        "normal": normal_map,
                  },
                  light_dir = light_dir,
                  view_dir = spy.float3(0, 0, 1),
                  _result = output)

    # Downsample the output tensor.
    output = downsample(output, 2)

    # Blit tensor to screen.
    app.blit(output, size=spy.int2(1024, 1024))

    # Quarter res rendered output BRDF from quarter res inputs.
    lr_output = spy.Tensor.empty_like(output)
    module.render(pixel = spy.call_id(),
                  material = {
                        "albedo": lr_albedo_map,
                        "normal": lr_normal_map,
                  },
                  light_dir = light_dir,
                  view_dir = spy.float3(0, 0, 1),
                  _result = lr_output)

    # Blit tensor to screen.
    app.blit(lr_output, size=spy.int2(1024, 1024), offset=spy.int2(2068, 0))

    # Loss between downsampled output and quarter res rendered output.
    loss_output = spy.Tensor.empty_like(output)
    module.loss(pixel = spy.call_id(),
                  material = {
                        "albedo": lr_albedo_map,
                        "normal": lr_normal_map,
                  },
                  reference = output,
                  light_dir = light_dir,
                  view_dir = spy.float3(0, 0, 1),
                  _result = loss_output)


    # Blit tensor to screen.
    app.blit(loss_output, size=spy.int2(1024, 1024), offset=spy.int2(1034, 0), tonemap=False)

    # Blit tensor to screen.
    app.blit(lr_normal_map, size=spy.int2(1024, 1024), offset=spy.int2(3102, 0), tonemap=False)

    # Loss between downsampled output and quarter res rendered output.
    module.calculate_grads(
        seed = spy.wang_hash(seed=optimize_counter, warmup=2),
        pixel = spy.call_id(),
        material = {
                "albedo": lr_albedo_map,
                "normal": lr_normal_map,
                "albedo_grad": lr_albedo_grad,
                "normal_grad": lr_normal_grad,
        },
        reference = output)
    optimize_counter += 1


    # Blit tensor to screen.
    app.blit(lr_normal_grad, size=spy.int2(1024, 1024), offset=spy.int2(1034, 0), tonemap=False)

    module.optimize(lr_normal_map, lr_normal_grad, m_normal, v_normal, 0.01)
    lr_normal_grad.clear()

    # read loss output to numpy tensor and sum abs values
    loss_np = loss_output.to_numpy()
    loss_value = np.sum(np.abs(loss_np))
    print(f"Loss: {loss_value:.6f}")

    # Present the window.
    app.present()


