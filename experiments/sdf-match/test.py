# SPDX-License-Identifier: Apache-2.0

import slangpy as spy
import numpy as np
from app import App
from slangpy.types import call_id
import time
from pathlib import Path


def get_weight_size(network_shape: list[int]):
    total_size = 0;
    for i in range(len(network_shape) - 1):
        total_size += network_shape[i] * network_shape[i + 1];
    return total_size;

def get_bias_size(network_shape: list[int]):
    total_size = 0;
    for i in range(len(network_shape) - 1):
        total_size += network_shape[i + 1];
    return total_size;

def get_layer_weight_size(network_shape: list[int], layer_index: int):
    return network_shape[layer_index] * network_shape[layer_index + 1];

def get_layer_bias_size(network_shape: list[int], layer_index: int):
    return network_shape[layer_index + 1];

def generate_init_params(network_shape: list[int]):
    params_size = get_weight_size(network_shape) + get_bias_size(network_shape);

    params_data = np.random.randn(params_size,).astype(np.float32)
    params_grad_data = np.zeros((params_size,), dtype=np.float32)

    params = spy.NDBuffer.from_numpy(device, params_data)
    params_grad = spy.NDBuffer.from_numpy(device, params_grad_data)
    return params, params_grad, params_data, params_grad_data

def generate_input(samplesSize: int):
    input = np.random.randn(samplesSize, input_size).astype(np.float32)
    return spy.Tensor.from_numpy(device, input), input

def construct_network(params: spy.NDBuffer, params_grad: spy.NDBuffer):
    tiny_mlp = spy.NDBuffer(device, dtype=module.TinyMLP_Params, shape=(1,))
    tiny_mlp_cursor = tiny_mlp.cursor()
    tiny_mlp_cursor[0].write(
        {
            "m_params": params.storage.device_address,
            "m_grads": params_grad.storage.device_address,
        }
    )
    tiny_mlp_cursor.apply()
    return tiny_mlp;

# app = App()

# call trainMLP shader code, and check the result on CPU side
def unit_test_trainMLP():
    # need to implement the MLP on python side
    samplesSize = 1
    input_size = 3;
    hidden_size = 16;
    output_size = 1;

    network_shape = [input_size, hidden_size, output_size];

    input_tensor, X = generate_input(samplesSize)
    print("input:")
    np.set_printoptions(suppress=True, precision=6)
    print(X)

    params, params_grad, params_data, params_grad_data = generate_init_params(network_shape)
    print("weights:")
    print(params_data)
    print("biases:")
    print(params_grad_data)

    tiny_mlp = construct_network(params=params, params_grad=params_grad)
    gpu_loss = module.trainMLP(tiny_mlp.storage.device_address, input_tensor)
    print("GPU Result forward pass loss:")
    print(gpu_loss.to_numpy())

    # check the result on CPU side
    print("\nCPU Verification:")


    # Reshape weights and biases for the 2-layer network
    # Layer 1: input_size -> hidden_size
    weight_size = get_layer_weight_size(network_shape, 0)
    bias_size = get_layer_bias_size(network_shape, 0)
    W1 = params_data[:weight_size].reshape(hidden_size, input_size).transpose()
    b1 = params_data[weight_size:weight_size + bias_size].reshape(1, hidden_size)
    # Layer 2: hidden_size -> output_size

    layer_offset = weight_size + bias_size
    weight_size = get_layer_weight_size(network_shape, 1)
    bias_size += get_layer_bias_size(network_shape, 1)
    W2 = params_data[layer_offset:layer_offset + weight_size].reshape(output_size, hidden_size).transpose()
    b2 = params_data[layer_offset + weight_size:layer_offset + weight_size + bias_size].reshape(1, output_size)

    # Forward pass
    def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Leaky ReLU activation function"""
        return np.where(x > 0, x, alpha * x)

    def leaky_relu_grad(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Gradient of Leaky ReLU activation function"""
        return np.where(x > 0, 1.0, alpha)

    # Layer 1
    Z1 = np.dot(X, W1) + b1  # shape: (samplesSize, hidden_size)
    A1 = leaky_relu(Z1)

    # Layer 2 (with Leaky ReLU activation)
    Z2 = np.dot(A1, W2) + b2  # shape: (samplesSize, output_size)
    A2 = leaky_relu(Z2)
    output = A2

    print("CPU Forward Pass Result:")
    print(output)

    # Simple loss calculation - target is x+y+z (sum of input coordinates)
    target = np.sum(X, axis=1, keepdims=True).astype(np.float32)  # shape: (samplesSize, 1)
    diff = output - target  # shape: (samplesSize, output_size)
    loss = np.linalg.norm(diff)

    # Also add backward pass to check the gradient calculation
    print(f"\nCPU Loss: {loss}")

    # Backward pass - compute gradients
    # Loss = ||output - target||_2, so dL/doutput = (output - target) / ||output - target||_2
    # This is the exact gradient for L2 norm loss

    dL_doutput = diff / loss  # shape: (samplesSize, output_size)

    # Backward pass follows the exact reverse of forward pass:
    # Forward: X -> Z1 = X@W1 + b1 -> A1 = leaky_relu(Z1) -> Z2 = A1@W2 + b2 -> A2 = leaky_relu(Z2) -> output = A2

    # Layer 2 gradients (A2 -> Z2 -> W2, b2, A1)
    # dL/dA2 = dL/doutput (since output = A2)
    dL_dA2 = dL_doutput  # shape: (samplesSize, output_size)
    # dL/dZ2 = dL/dA2 * dA2/dZ2 = dL/dA2 * leaky_relu'(Z2)
    dL_dZ2 = dL_dA2 * leaky_relu_grad(Z2)  # shape: (samplesSize, output_size)
    # dL/dW2 = dL/dZ2 * dZ2/dW2 = dL/dZ2 * A1^T
    dL_dW2 = np.dot(A1.T, dL_dZ2)  # shape: (hidden_size, output_size)
    # dL/db2 = dL/dZ2 * dZ2/db2 = sum(dL/dZ2, axis=0)
    dL_db2 = np.sum(dL_dZ2, axis=0, keepdims=True)  # shape: (1, output_size)
    # dL/dA1 = dL/dZ2 * dZ2/dA1 = dL/dZ2 * W2^T
    dL_dA1 = np.dot(dL_dZ2, W2.T)  # shape: (samplesSize, hidden_size)

    # Layer 1 gradients (A1 -> Z1 -> W1, b1)
    # dL/dZ1 = dL/dA1 * dA1/dZ1 = dL/dA1 * leaky_relu'(Z1)
    dL_dZ1 = dL_dA1 * leaky_relu_grad(Z1)  # shape: (samplesSize, hidden_size)

    # dL/dW1 = dL/dZ1 * dZ1/dW1 = dL/dZ1 * X^T
    dL_dW1 = np.dot(X.T, dL_dZ1)  # shape: (input_size, hidden_size)

    # dL/db1 = dL/dZ1 * dZ1/db1 = sum(dL/dZ1, axis=0)
    dL_db1 = np.sum(dL_dZ1, axis=0, keepdims=True)  # shape: (1, hidden_size)

    # Reshape gradients to match the flat format used in GPU
    cpu_weights_grad = np.concatenate([
        dL_dW1.flatten(),  # Layer 1 weights gradient
        dL_dW2.flatten()   # Layer 2 weights gradient
    ])

    cpu_biases_grad = np.concatenate([
        dL_db1.flatten(),  # Layer 1 biases gradient
        dL_db2.flatten()   # Layer 2 biases gradient
    ])

    print("\nCPU Gradients:")
    print("Weights gradients:")
    print(cpu_weights_grad)
    print("Biases gradients:")
    print(cpu_biases_grad)

    # Get GPU gradients for comparison
    gpu_params_grad = params_grad.to_numpy()

    print("\nGPU Gradients:")
    print("params gradients:")
    print(gpu_params_grad)

    # Compare gradients
    print(f"\nGradient Comparison:")
    print(f"Params gradient max difference:")
    cpu_params_grad = np.concatenate([
        dL_dW1.flatten(),
        dL_db1.flatten(),
        dL_dW2.flatten(),
        dL_db2.flatten()
    ])
    print(np.abs(gpu_params_grad - cpu_params_grad))

    # implement adam optimizer
    param_size = get_weight_size(network_shape) + get_bias_size(network_shape)
    adam_state = spy.NDBuffer(device, dtype=module.AdamState, shape=(param_size,))
    module.clearAdamState(adam_state)
    module.updateParams(tiny_mlp.storage.device_address, adam_state, 0.9, 0.999)

    # CPU Adam optimizer implementation to verify GPU results
    print("\nCPU Adam Optimizer Implementation:")

    # Initialize Adam state (matching shader parameters)
    cpu_adam_state = {
        'mean': 0.0,
        'variance': 0.0,
        'epsilon': 1e-7,
        'iteration': 0,
        'learning_rate': 0.001
    }

    def cpu_adam_update(param: float, grad: float, state: dict, beta1: float, beta2: float) -> tuple[float, float]:
        """CPU implementation of Adam optimizer matching the shader"""
        state['mean'] = beta1 * state['mean'] + (1.0 - beta1) * grad
        state['variance'] = beta2 * state['variance'] + (1.0 - beta2) * grad * grad
        state['iteration'] += 1

        m_hat = state['mean'] / (1.0 - (beta1 ** state['iteration']))
        v_hat = state['variance'] / (1.0 - (beta2 ** state['iteration']))
        denom = np.sqrt(v_hat) + state['epsilon']

        new_param = param - state['learning_rate'] * m_hat / denom
        return new_param, 0.0  # Return updated param and reset grad to 0

    # Apply Adam update to weights and biases
    print("Before Adam update:")
    print("Params:", params_data[:5])  # Show first 5 weights
    print("Params grad:", params_grad_data[:3])    # Show first 3 biases

    # Update params using Adam
    for i in range(len(params_data)):
        params_data[i], _ = cpu_adam_update(params_data[i], cpu_params_grad[i], cpu_adam_state, 0.9, 0.999)


    print("\nAfter CPU Adam update:")
    print("Params:", params_data)
    print("Params grad:", params_grad_data)

    # Get GPU updated parameters for comparison
    gpu_params_after = params.to_numpy()
    gpu_params_grad_after = params_grad.to_numpy()

    print("\nGPU parameters after Adam update:")
    print("Params:", gpu_params_after)
    print("Params grad:", gpu_params_grad_after)

    print("\nAdam Update Comparison:")
    print("Weights difference (CPU vs GPU):")
    print(np.abs(params_data - gpu_params_after))
    print("Params grad difference (CPU vs GPU):")
    print(np.abs(params_grad_data - gpu_params_grad_after))

    return gpu_loss, loss


device = spy.create_device(spy.DeviceType.automatic, include_paths=[Path(__file__).parent])

spy.set_dump_generated_shaders(True)
module = spy.Module.load_from_file(device, "tinymlp.slang")

samplesSize = 50
learningRate = 0.001
iteration = 500

input_size = 3;
hidden_size = 16;
output_size = 1;

network_shape = [input_size, hidden_size, output_size];

input, input_data = generate_input(samplesSize)
params, params_grad, params_data, params_grad_data = generate_init_params(network_shape)
tiny_mlp = construct_network(params=params, params_grad=params_grad)

# Initialize Adam state
param_size = get_weight_size(network_shape) + get_bias_size(network_shape)
adam_state = spy.NDBuffer(device, dtype=module.AdamState, shape=(param_size,))
module.clearAdamState(adam_state)
adam_state_cursor = adam_state.cursor()
print(adam_state_cursor[0].read())


for i in range(iteration):
    result = module.trainMLP(tiny_mlp.storage.device_address, input)
    print(np.max(np.abs(result.to_numpy())))
    print('--------------------------------')

    call_id = module.updateParams(tiny_mlp.storage.device_address, adam_state, 0.9, 0.999)
    adam_state_cursor = adam_state.cursor()
    print(adam_state_cursor[0].read())

eval_result = module.EvalMLP(tiny_mlp.storage.device_address, [1.0,2.0,3.0])
print(eval_result.to_numpy())

# Run the unit test to verify trainMLP implementation
# print("=" * 50)
# print("Running unit test for trainMLP verification")
# print("=" * 50)
# gpu_result, cpu_result = unit_test_trainMLP()
# print("=" * 50)
# print("Unit test completed")
# print("=" * 50)

exit()

def findMachingSDF(iter):
    #  for i in range(iteration):
    module.forward(samplePoint=input, sdf_params=params, _result=forwardResult)

    forwardResult.grad.storage.copy_from_numpy(allOnes)

    module.forward.bwds(samplePoint, params, _result=forwardResult)

    paramArray = params.to_numpy()
    gradArray = params.grad.to_numpy()
    paramArray = paramArray - learningRate * gradArray / samplesSize
    paramArray[3] = np.fmax(paramArray[3], 0.0)  # Clamp the radius to be positive

    params.storage.copy_from_numpy(paramArray)
    params.grad.clear()

    if iter % 50 == 0:
        resultArray = forwardResult.to_numpy()
        loss = np.linalg.norm(resultArray) / samplesSize
        print("Iteration: {}, Loss: {}".format(iter, loss))
        print("parameter {}".format(params.to_numpy()))

    return forwardResult


iter = 0
while app.process_events():
    if iter < iteration:
        forwardResult = findMachingSDF(iter)
        iter += 1
    pylist = params.to_numpy().tolist()
    windowSize = spy.float2(app._window.width, app._window.height)
    module.RunRayMarch(windowSize, call_id(), pylist, _result=app.output)
    time.sleep(0.005)  # Sleep for 10ms to see the evolution of the SDF
    app.present()
