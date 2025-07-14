# SPDX-License-Identifier: Apache-2.0

from app import App
import slangpy as spy
import numpy as np

# Create the app and load the slang module.
app = App(width=512 * 3 + 10 * 2, height=512, title="Mipmap Example")
module = spy.Module.load_from_file(app.device, "nsc_nw_02_3layers.slang")

# Load some materials.
image = spy.Tensor.load_from_image(app.device, "slangstars.png", linearize=False)


class NetworkParameters(spy.InstanceList):
    def __init__(self, inputs: int, outputs: int):
        super().__init__(module[f"NetworkParameters<{inputs},{outputs}>"])
        self.inputs = inputs
        self.outputs = outputs

        # Biases and weights for the layer.
        self.biases = spy.Tensor.from_numpy(app.device, np.zeros(outputs).astype("float32"))
        self.weights = spy.Tensor.from_numpy(
            app.device, np.random.uniform(-0.5, 0.5, (outputs, inputs)).astype("float32")
        )

        # Gradients for the biases and weights.
        self.biases_grad = spy.Tensor.zeros_like(self.biases)
        self.weights_grad = spy.Tensor.zeros_like(self.weights)

        # Temp data for Adam optimizer.
        self.m_biases = spy.Tensor.zeros_like(self.biases)
        self.m_weights = spy.Tensor.zeros_like(self.weights)
        self.v_biases = spy.Tensor.zeros_like(self.biases)
        self.v_weights = spy.Tensor.zeros_like(self.weights)

    # Calls the Slang 'optimize' function for biases and weights
    def optimize(self, learning_rate: float, optimize_counter: int):
        module.optimize1(
            self.biases,
            self.biases_grad,
            self.m_biases,
            self.v_biases,
            learning_rate,
            optimize_counter,
        )
        module.optimize1(
            self.weights,
            self.weights_grad,
            self.m_weights,
            self.v_weights,
            learning_rate,
            optimize_counter,
        )


class Network(spy.InstanceList):
    def __init__(self):
        super().__init__(module["Network"])
        self.layer0 = NetworkParameters(2, 32)
        self.layer1 = NetworkParameters(32, 32)
        self.layer2 = NetworkParameters(32, 3)

    # Calls the Slang 'optimize' function for the layer.
    def optimize(self, learning_rate: float, optimize_counter: int):
        self.layer0.optimize(learning_rate, optimize_counter)
        self.layer1.optimize(learning_rate, optimize_counter)
        self.layer2.optimize(learning_rate, optimize_counter)


network = Network()

optimize_counter = 0

while app.process_events():

    # Blit tensor to screen.
    offset = 0
    app.blit(image, size=spy.int2(512), offset=spy.int2(offset, 0), tonemap=False, bilinear=True)
    offset += 512 + 10
    res = spy.int2(256, 256)

    lr_output = spy.Tensor.empty_like(image)
    module.render(pixel=spy.call_id(), resolution=res, network=network, _result=lr_output)

    # Blit tensor to screen.
    app.blit(lr_output, size=spy.int2(512, 512), offset=spy.int2(offset, 0), tonemap=False)
    offset += 512 + 10

    # Loss between downsampled output and quarter res rendered output.
    loss_output = spy.Tensor.empty_like(image)
    module.loss(
        pixel=spy.call_id(), resolution=res, network=network, reference=image, _result=loss_output
    )

    # Blit tensor to screen.
    app.blit(loss_output, size=spy.int2(512, 512), offset=spy.int2(offset, 0), tonemap=False)
    offset += 512 + 10

    learning_rate = 0.001

    for i in range(50):
        # Loss between downsampled output and quarter res rendered output.
        module.calculate_grads(
            seed=spy.wang_hash(seed=optimize_counter, warmup=2),
            pixel=spy.call_id(),
            resolution=res,
            reference=image,
            network=network,
        )
        optimize_counter += 1

        network.optimize(learning_rate, optimize_counter)

    print("Loss:", np.sum(np.abs(loss_output.to_numpy())))

    # Present the window.
    app.present()
