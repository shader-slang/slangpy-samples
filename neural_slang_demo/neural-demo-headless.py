# SPDX-License-Identifier: Apache-2.0
# Neural Network Demo using neural.slang's FFLayer (Headless Version)
#
# Saves results to files instead of displaying in a window.
#
# Demonstrates neural.slang types:
# - InlineVector<T, N>: Fixed-size vectors
# - StructuredBufferStorage<T>: GPU buffer storage
# - FFLayer<T, InVec, OutVec, Storage, Activation, HasBias>: Feed-forward layer
# - LeakyReLU<T>, ExpActivation<T>: Activation functions
# - FFLayer.ParameterCount: Compile-time parameter count

import slangpy as spy
from slangpy.core.calldata import SLANG_PATH
import numpy as np
from pathlib import Path
from PIL import Image

# Setup paths
data_path = Path(__file__).parent
output_path = data_path / "output"
output_path.mkdir(exist_ok=True)

# Create headless device with experimental features for neural.slang
print("Creating device...")
compiler_options = spy.SlangCompilerOptions({
    "include_paths": [data_path, SLANG_PATH],
    "enable_experimental_features": True,
})
device = spy.Device(
    type=spy.DeviceType.vulkan,
    compiler_options=compiler_options,
)

# Load the slang module
print("Loading neural-demo.slang...")
module = spy.Module(device.load_module(str(data_path / "neural-demo.slang")))

# Load reference image and convert to RGB Tensor<float3, 2>
print("Loading reference image...")
image_rgba = spy.Tensor.load_from_image(device, data_path / "slangstars.png", linearize=True)
print(f"  Original image shape: {image_rgba.shape}")

# Convert RGBA to RGB if needed and create proper float3 tensor
image_np = image_rgba.to_numpy()
if len(image_np.shape) == 3 and image_np.shape[2] == 4:
    image_np = image_np[:, :, :3]  # Drop alpha channel
    print(f"  Converted to RGB: {image_np.shape}")

# Create a 2D tensor with float3 dtype (height x width of float3 vectors)
height, width = image_np.shape[0], image_np.shape[1]
image = spy.Tensor.empty(device, shape=(height, width), dtype="float3")
# Copy data - reshape to match the expected layout
image.storage.copy_from_numpy(image_np.astype(np.float32).flatten())
print(f"  Final image shape: {image.shape} (dtype: float3)")


def create_buffer(data: np.ndarray):
    """Create a GPU buffer from numpy array."""
    return device.create_buffer(
        element_count=data.size,
        struct_size=4,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=data.astype("float32").flatten()
    )


def save_tensor_as_image(tensor: spy.Tensor, filename: str):
    """Save a tensor as an image file."""
    data = tensor.to_numpy()
    # Clamp and convert to uint8
    data = np.clip(data * 255, 0, 255).astype(np.uint8)
    if len(data.shape) == 3:
        img = Image.fromarray(data, mode='RGB')
    else:
        img = Image.fromarray(data)
    img.save(filename)
    print(f"  Saved: {filename}")


class LayerParams:
    """Parameters for a single layer."""

    def __init__(self, inputs: int, outputs: int):
        self.inputs = inputs
        self.outputs = outputs
        self.param_count = outputs * inputs + outputs  # weights + biases

        # Xavier initialization
        scale = np.sqrt(6.0 / (inputs + outputs))
        self.weights_np = np.random.uniform(-scale, scale, (outputs, inputs)).astype("float32")
        self.biases_np = np.zeros(outputs, dtype="float32")

    def get_params(self):
        """Return flattened params: [weights, biases]."""
        return np.concatenate([self.weights_np.flatten(), self.biases_np])


class TrainableLayerParams(spy.InstanceList):
    """Trainable layer parameters (tensor-based for gradient accumulation)."""

    def __init__(self, inputs: int, outputs: int, layer_params: LayerParams):
        super().__init__(module[f"TrainableLayer<{inputs},{outputs}>"])
        self.layer_params = layer_params

        # Initialize from layer params
        self.weights = spy.Tensor.from_numpy(device, layer_params.weights_np)
        self.biases = spy.Tensor.from_numpy(device, layer_params.biases_np)
        self.weights_grad = spy.Tensor.zeros_like(self.weights)
        self.biases_grad = spy.Tensor.zeros_like(self.biases)

        # Adam state
        self.m_weights = spy.Tensor.zeros_like(self.weights)
        self.m_biases = spy.Tensor.zeros_like(self.biases)
        self.v_weights = spy.Tensor.zeros_like(self.weights)
        self.v_biases = spy.Tensor.zeros_like(self.biases)

    def optimize(self, learning_rate: float, iteration: int):
        module.optimizer_step(self.weights, self.weights_grad, self.m_weights, self.v_weights, learning_rate, iteration)
        module.optimizer_step(self.biases, self.biases_grad, self.m_biases, self.v_biases, learning_rate, iteration)
        # Sync back to layer params
        self.layer_params.weights_np = self.weights.to_numpy()
        self.layer_params.biases_np = self.biases.to_numpy()


class MLPNetwork(spy.InstanceList):
    """MLP using FFLayer with single buffer for all layers."""

    def __init__(self):
        super().__init__(module["MLPNetwork"])
        # Create layer parameters
        self.layer0 = LayerParams(inputs=4, outputs=32)
        self.layer1 = LayerParams(inputs=32, outputs=32)
        self.layer2 = LayerParams(inputs=32, outputs=3)

        # Create single buffer with all parameters
        self._update_buffer()

    def _update_buffer(self):
        """Concatenate all layer params into single buffer."""
        all_params = np.concatenate([
            self.layer0.get_params(),
            self.layer1.get_params(),
            self.layer2.get_params()
        ])
        self.params = create_buffer(all_params)


class TrainableMLPNetwork(spy.InstanceList):
    """MLP for training (tensor-based with gradient accumulation)."""

    def __init__(self, mlp_network: MLPNetwork):
        super().__init__(module["TrainableMLPNetwork"])
        self.mlp_network = mlp_network

        # Create trainable layers linked to MLPNetwork layer params
        self.layer0 = TrainableLayerParams(4, 32, mlp_network.layer0)
        self.layer1 = TrainableLayerParams(32, 32, mlp_network.layer1)
        self.layer2 = TrainableLayerParams(32, 3, mlp_network.layer2)

    def optimize(self, learning_rate: float, iteration: int):
        self.layer0.optimize(learning_rate, iteration)
        self.layer1.optimize(learning_rate, iteration)
        self.layer2.optimize(learning_rate, iteration)
        # Rebuild single buffer from updated params
        self.mlp_network._update_buffer()


class LatentTexture(spy.InstanceList):
    """Latent texture for spatial feature encoding."""

    def __init__(self, width: int, height: int, num_latents: int):
        super().__init__(module[f"LatentTexture<{num_latents}>"])

        initial = np.random.uniform(0.0, 1.0, (height, width, num_latents)).astype("float32")
        self.texture = spy.Tensor.from_numpy(device, initial)
        self.texture_grads = spy.Tensor.zeros_like(self.texture)

        self.m_texture = spy.Tensor.zeros_like(self.texture)
        self.v_texture = spy.Tensor.zeros_like(self.texture)

    def optimize(self, learning_rate: float, iteration: int):
        module.optimizer_step(self.texture, self.texture_grads, self.m_texture, self.v_texture, learning_rate, iteration)


class Network(spy.InstanceList):
    """Network using FFLayer for rendering."""

    def __init__(self):
        super().__init__(module["Network"])
        self.latent_texture = LatentTexture(32, 32, 4)
        self.mlp = MLPNetwork()


class TrainableNetwork(spy.InstanceList):
    """Network for training with gradient accumulation."""

    def __init__(self, render_network: Network):
        super().__init__(module["TrainableNetwork"])
        self.latent_texture = render_network.latent_texture  # Share latent texture
        self.mlp = TrainableMLPNetwork(render_network.mlp)

    def optimize(self, learning_rate: float, iteration: int):
        self.latent_texture.optimize(learning_rate, iteration)
        self.mlp.optimize(learning_rate, iteration)


# Create networks
print("\nInitializing networks...")
render_network = Network()
train_network = TrainableNetwork(render_network)

print("Using neural.slang: FFLayer, StructuredBufferStorage, InlineVector, LeakyReLU, ExpActivation")
total_params = (32*4+32) + (32*32+32) + (3*32+3)
print(f"Total MLP parameters: {total_params}")

# Training parameters
res = spy.int2(256, 256)
batch_size = (64, 64)
learning_rate = 0.001
total_epochs = 100
steps_per_epoch = 20
save_every = 10  # Save images every N epochs

print(f"\nTraining for {total_epochs} epochs ({steps_per_epoch} steps/epoch)...")
print(f"Saving images every {save_every} epochs to: {output_path}")

# Save reference image
save_tensor_as_image(image, str(output_path / "reference.png"))

# Training loop
for epoch in range(total_epochs):
    # Training steps
    for step in range(steps_per_epoch):
        optimize_counter = epoch * steps_per_epoch + step + 1
        module.calculate_grads(
            seed=spy.wang_hash(seed=optimize_counter, warmup=2),
            batch_index=spy.grid(batch_size),
            batch_size=spy.int2(batch_size),
            reference=image,
            network=train_network,
        )
        train_network.optimize(learning_rate, optimize_counter)

    # Compute loss
    loss_output = spy.Tensor.empty_like(image)
    module.loss(pixel=spy.call_id(), resolution=res, network=render_network, reference=image, _result=loss_output)
    loss_value = np.mean(loss_output.to_numpy())

    print(f"Epoch {epoch + 1:3d}/{total_epochs}: Loss = {loss_value:.6f}")

    # Save images periodically
    if (epoch + 1) % save_every == 0 or epoch == 0:
        # Render current network output
        render_output = spy.Tensor.empty_like(image)
        module.render(pixel=spy.call_id(), resolution=res, network=render_network, _result=render_output)
        
        save_tensor_as_image(render_output, str(output_path / f"output_epoch_{epoch + 1:03d}.png"))
        save_tensor_as_image(loss_output, str(output_path / f"loss_epoch_{epoch + 1:03d}.png"))

# Final render
print("\nTraining complete!")
render_output = spy.Tensor.empty_like(image)
module.render(pixel=spy.call_id(), resolution=res, network=render_network, _result=render_output)
save_tensor_as_image(render_output, str(output_path / "final_output.png"))

device.close()
print(f"\nAll results saved to: {output_path}")
