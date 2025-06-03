# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from app import App
import slangpy as spy
import numpy as np

# Create the app and load the slang module.
app = App(width=3092, height=1024, title="Mipmap Example")
module = spy.Module.load_from_file(app.device, "nsc_06_network.slang")

spy.set_dump_generated_shaders(True)

# Load some materials.
albedo_map = spy.Tensor.load_from_image(app.device,
                                        "PavingStones070_2K.diffuse.jpg", linearize=True)
normal_map = spy.Tensor.load_from_image(app.device,
                                        "PavingStones070_2K.normal.jpg", scale=2, offset=-1)
roughness_map = spy.Tensor.load_from_image(app.device,
                                        "PavingStones070_2K.roughness.jpg",
                                        grayscale=True)

def downsample(source: spy.Tensor, steps: int) -> spy.Tensor:
    for i in range(steps):
        dest = spy.Tensor.empty(device=app.device, shape=(source.shape[0] // 2, source.shape[1] // 2), dtype=source.dtype)
        if dest.dtype.name == 'vector':
            module.downsample3(spy.call_id(), source, _result=dest)
        else:
            module.downsample1(spy.call_id(), source, _result=dest)
        source = dest
    return source

biases = spy.Tensor.from_numpy(app.device, np.random.uniform(0, 1, (7,)).astype('float32'))

#biases = spy.Tensor.from_numpy(app.device, np.array([0.1,0.4,0.7,1,1,1,1]).astype('float32'))


weights = spy.Tensor.from_numpy(app.device, np.random.uniform(0, 1, (7, 2)).astype('float32'))
biases_grad = spy.Tensor.zeros_like(biases)
weights_grad = spy.Tensor.zeros_like(weights)
m_biases = spy.Tensor.zeros_like(biases)
m_weights = spy.Tensor.zeros_like(weights)
v_biases = spy.Tensor.zeros_like(biases)
v_weights = spy.Tensor.zeros_like(weights)

n1 = {
    "_type": "NetworkParameters<2, 7>",
    "biases": biases,
    "weights": weights,
    "biases_grad": biases_grad,
    "weights_grad": weights_grad,
}

n2_biases_0 = spy.Tensor.from_numpy(app.device, np.random.uniform(0, 1, (16,)).astype('float32'))
n2_weights_0 = spy.Tensor.from_numpy(app.device, np.random.uniform(0, 1, (16, 2)).astype('float32'))
n2_biases_1 = spy.Tensor.from_numpy(app.device, np.random.uniform(0, 1, (7,)).astype('float32'))
n2_weights_1 = spy.Tensor.from_numpy(app.device, np.random.uniform(0, 1, (7, 16)).astype('float32'))
n2_biases_grad_0 = spy.Tensor.zeros_like(n2_biases_0)
n2_weights_grad_0 = spy.Tensor.zeros_like(n2_weights_0)
n2_biases_grad_1 = spy.Tensor.zeros_like(n2_biases_1)
n2_weights_grad_1 = spy.Tensor.zeros_like(n2_weights_1)
n2_m_biases_0 = spy.Tensor.zeros_like(n2_biases_0)
n2_m_weights_0 = spy.Tensor.zeros_like(n2_weights_0)
n2_m_biases_1 = spy.Tensor.zeros_like(n2_biases_1)
n2_m_weights_1 = spy.Tensor.zeros_like(n2_weights_1)
n2_v_biases_0 = spy.Tensor.zeros_like(n2_biases_0)
n2_v_weights_0 = spy.Tensor.zeros_like(n2_weights_0)
n2_v_biases_1 = spy.Tensor.zeros_like(n2_biases_1)
n2_v_weights_1 = spy.Tensor.zeros_like(n2_weights_1)

n2_0 = {
    "_type": "NetworkParameters<2, 16>",
    "biases": n2_biases_0,
    "weights": n2_weights_0,
    "biases_grad": n2_biases_grad_0,
    "weights_grad": n2_weights_grad_0,
}
n2_1 = {
    "_type": "NetworkParameters<16, 7>",
    "biases": n2_biases_1,
    "weights": n2_weights_1,
    "biases_grad": n2_biases_grad_1,
    "weights_grad": n2_weights_grad_1,
}

optimize_counter = 0

while app.process_events():

    # Full res rendered output BRDF from full res inputs.
    output = spy.Tensor.empty_like(albedo_map)
    module.render(pixel = spy.call_id(),
                  material = {
                        "albedo": albedo_map,
                        "normal": normal_map,
                        "roughness": roughness_map
                  },
                  light_dir = spy.math.normalize(spy.float3(0.2, 0.2, 1.0)),
                  view_dir = spy.float3(0, 0, 1),
                  _result = output)

    # Downsample the output tensor.
    output = downsample(output, 2)

    # Blit tensor to screen.
    app.blit(output, size=spy.int2(1024, 1024))

    # Quarter res rendered output BRDF from quarter res inputs.
    lr_output = spy.Tensor.empty_like(output)
    module.render_2layer(pixel = spy.call_id(),
                  resolution = spy.int2(512, 512),
                  network_0 = n2_0,
                  network_1 = n2_1,
                  light_dir = spy.math.normalize(spy.float3(0.2, 0.2, 1.0)),
                  view_dir = spy.float3(0, 0, 1),
                  _result = lr_output)

    # Blit tensor to screen.
    app.blit(lr_output, size=spy.int2(1024, 1024), offset=spy.int2(2068, 0))

    # Loss between downsampled output and quarter res rendered output.
    loss_output = spy.Tensor.empty_like(output)
    module.loss_2layer(pixel = spy.call_id(),
                  resolution = spy.int2(512, 512),
                  network_0 = n2_0,
                  network_1 = n2_1,
                  reference = output,
                  light_dir = spy.math.normalize(spy.float3(0.2, 0.2, 1.0)),
                  view_dir = spy.float3(0, 0, 1),
                  _result = loss_output)

    # Blit tensor to screen.
    app.blit(loss_output, size=spy.int2(1024, 1024), offset=spy.int2(1034, 0), tonemap=False)

    # Loss between downsampled output and quarter res rendered output.
    module.calculate_grads_2layer(
        seed = spy.wang_hash(seed=optimize_counter, warmup=2),
        pixel = spy.grid(shape=lr_output.shape),
        resolution = spy.int2(512, 512),
        network_0 = n2_0,
        network_1 = n2_1,
        ref_material = {
                "albedo": albedo_map,
                "normal": normal_map,
                "roughness": roughness_map,
        })
    optimize_counter += 1

    print("Loss:", np.sum(np.abs(loss_output.to_numpy())))

    # Optimize the trained maps using the gradients.
    module.optimize1(biases, biases_grad, m_biases, v_biases, 0.01)
    module.optimize1(weights, weights_grad, m_weights, v_weights, 0.01)
    module.optimize1(n2_biases_0, n2_biases_grad_0, n2_m_biases_0, n2_v_biases_0, 0.01)
    module.optimize1(n2_weights_0, n2_weights_grad_0, n2_m_weights_0, n2_v_weights_0, 0.01)
    module.optimize1(n2_biases_1, n2_biases_grad_1, n2_m_biases_1, n2_v_biases_1, 0.01)
    module.optimize1(n2_weights_1, n2_weights_grad_1, n2_m_weights_1, n2_v_weights_1, 0.01)

    # Present the window.
    app.present()




