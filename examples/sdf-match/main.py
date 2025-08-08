# SPDX-License-Identifier: Apache-2.0

import slangpy as spy
import numpy as np
from app import App
from slangpy.types import call_id
import time
from pathlib import Path


# get weight size of the network
def get_weight_size(network_shape: list[int]):
    total_size = 0
    for i in range(len(network_shape) - 1):
        total_size += network_shape[i] * network_shape[i + 1]
    return total_size


# get bias size of the network
def get_bias_size(network_shape: list[int]):
    total_size = 0
    for i in range(len(network_shape) - 1):
        total_size += network_shape[i + 1]
    return total_size


# get weight size of the layer
def get_layer_weight_size(network_shape: list[int], layer_index: int):
    return network_shape[layer_index] * network_shape[layer_index + 1]


# get bias size of the layer
def get_layer_bias_size(network_shape: list[int], layer_index: int):
    return network_shape[layer_index + 1]


# generate the initial parameters for the network, weight and bias are allocated in the same buffer
# the layout is weight1, bias1, weight2, bias2, ...
# the gradient is allocated in another buffer, and the layout is the same as the params
def generate_init_params(network_shape: list[int]):
    params_size = get_weight_size(network_shape) + get_bias_size(network_shape)
    std = np.sqrt(2.0 / 3.0).astype(np.float32)
    params_data = np.random.randn(params_size).astype(np.float32)
    params_data = params_data * std

    params_grad_data = np.zeros((params_size,), dtype=np.float32)

    params = spy.NDBuffer.from_numpy(app.device, params_data)
    params_grad = spy.NDBuffer.from_numpy(app.device, params_grad_data)
    return params, params_grad, params_data, params_grad_data


# generate the input for the network
def generate_input(samplesSize: int, input_size: int):
    std = np.sqrt(2.0 / 3.0).astype(np.float32)
    input = np.random.randn(samplesSize, input_size).astype(np.float32)
    input = input * std
    return spy.Tensor.from_numpy(app.device, input), input


# construct the network, the network is a tiny MLP with 1 hidden layer
def construct_network(params: spy.NDBuffer, params_grad: spy.NDBuffer):
    tiny_mlp = spy.NDBuffer(app.device, dtype=module.TinyMLP_Params, shape=(1,))
    tiny_mlp_cursor = tiny_mlp.cursor()
    tiny_mlp_cursor[0].write(
        {
            "m_params": params.storage.device_address,
            "m_grads": params_grad.storage.device_address,
        }
    )
    tiny_mlp_cursor.apply()
    return tiny_mlp


def trainMLP(
    iter: int,
    tiny_mlp_params: spy.NDBuffer,
    adam_state: spy.NDBuffer,
    input: spy.Tensor,
    batch_size: int,
):
    result = module.trainMLP(tiny_mlp_params.storage.device_address, input)
    module.updateParams(tiny_mlp_params.storage.device_address, adam_state, batch_size, 0.9, 0.999)
    return result


app = App()
spy.set_dump_generated_shaders(True)
module = spy.Module.load_from_file(app.device, "main.slang")

samplesSize = 256

input_size = 3
hidden_size = 16
output_size = 1
network_shape = [input_size, hidden_size, output_size]

# Generate input, parameters, and gradients for the network
input, input_data = generate_input(samplesSize, input_size)
params, params_grad, params_data, params_grad_data = generate_init_params(network_shape)
tiny_mlp_params = construct_network(params=params, params_grad=params_grad)

# Initialize Adam state, the adam state is per-parameter state, so the size is the same as the params.
# Call slang function `clearAdamState` to clear the adam state.
param_size = get_weight_size(network_shape) + get_bias_size(network_shape)
adam_state = spy.NDBuffer(app.device, dtype=module.AdamState, shape=(param_size,))
module.clearAdamState(adam_state)

# Main loop
total_iter = 0
while app.process_events():
    # we will train the network for 100 iterations, and render the image every 1000 iterations
    iter = 0
    while iter < 100:
        result = trainMLP(iter, tiny_mlp_params, adam_state, input, samplesSize)
        iter += 1

    total_iter += iter
    if total_iter % 1000 == 0:
        result_np = result.to_numpy()
        loss = np.sum(result_np**2)
        print(f"Iteration {total_iter}: Loss = {loss}")

    # Render the image
    windowSize = spy.float2(app._window.width, app._window.height)
    module.RunRayMarch(
        windowSize,
        call_id(),
        tiny_mlp_params.storage.device_address,
        _result=app.output,
    )
    app.present()

exit()
