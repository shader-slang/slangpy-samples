# SPDX-License-Identifier: Apache-2.0
# Neural Network Demo using neural.slang's FFLayer
#
# Demonstrates neural.slang types:
# - InlineVector<T, N>: Fixed-size vectors
# - StructuredBufferStorage<T>: GPU buffer storage
# - FFLayer<T, InVec, OutVec, Storage, Activation, HasBias>: Feed-forward layer
# - LeakyReLU<T>, ExpActivation<T>: Activation functions
# - FFLayer.ParameterCount: Compile-time parameter count

from app import App
import slangpy as spy
import numpy as np
from pathlib import Path

# Create the app and load the slang module
app = App(width=512 * 3 + 10 * 2, height=512, title="Neural Demo (FFLayer)")
module = spy.Module.load_from_file(app.device, "neural-demo.slang")

# Load reference image
data_path = Path(__file__).parent
image = spy.Tensor.load_from_image(app.device, data_path.joinpath("slangstars.png"), linearize=True)


def create_buffer(data: np.ndarray):
    """Create a GPU buffer from numpy array."""
    return app.device.create_buffer(
        element_count=data.size,
        struct_size=4,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=data.astype("float32").flatten()
    )


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
        self.weights = spy.Tensor.from_numpy(app.device, layer_params.weights_np)
        self.biases = spy.Tensor.from_numpy(app.device, layer_params.biases_np)
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
        self.texture = spy.Tensor.from_numpy(app.device, initial)
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
render_network = Network()
train_network = TrainableNetwork(render_network)
optimize_counter = 0

print("Compiling shaders... this may take a while")
print("Using neural.slang: FFLayer, StructuredBufferStorage, InlineVector, LeakyReLU, ExpActivation")
print("[TODO] LatentTexture: Replace with HashGrid from neural.slang when available")
total_params = (32*4+32) + (32*32+32) + (3*32+3)
print(f"Total parameters: {total_params} (single buffer)")

while app.process_events():
    offset = 0

    # Blit reference
    app.blit(image, size=spy.int2(512), offset=spy.int2(offset, 0), tonemap=False, bilinear=True)
    offset += 522

    res = spy.int2(256, 256)
    batch_size = (64, 64)

    # Render using FFLayer-based network
    lr_output = spy.Tensor.empty_like(image)
    module.render(pixel=spy.call_id(), resolution=res, network=render_network, _result=lr_output)
    app.blit(lr_output, size=spy.int2(512), offset=spy.int2(offset, 0), tonemap=False, bilinear=True)
    offset += 522

    # Show loss
    loss_output = spy.Tensor.empty_like(image)
    module.loss(pixel=spy.call_id(), resolution=res, network=render_network, reference=image, _result=loss_output)
    app.blit(loss_output, size=spy.int2(512), offset=spy.int2(offset, 0), tonemap=False)

    learning_rate = 0.001

    # Training
    for _ in range(20):
        module.calculate_grads(
            seed=spy.wang_hash(seed=optimize_counter, warmup=2),
            batch_index=spy.grid(batch_size),
            batch_size=spy.int2(batch_size),
            reference=image,
            network=train_network,
        )
        optimize_counter += 1
        train_network.optimize(learning_rate, optimize_counter)

    print(f"Loss: {np.mean(loss_output.to_numpy()):.5f}")
    app.present()
